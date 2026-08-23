"""
seam_correction.py - Minimax H3 Seam Correction

分段长视频的段间接缝修正。两级架构：
- 色彩/曝光对齐（基于统计或 MKL）
- 接缝光流 warp + 局部混合

镜头检测用 PySceneDetect（纯 CPU，无模型污染），支持内存流，
若内存不足则自动写临时文件。
"""

import torch
import torch.nn.functional as F
from comfy_api.latest import io
import gc
import os
import cv2
import numpy as np
import traceback
import tempfile          
import time              
from scenedetect import SceneManager, ContentDetector, open_video   

from typing import List, Optional, Tuple


# ---------- 常量 ----------
_LONG_SIDE = 320
_STAT_LONG_SIDE = 256
_LUMA_W = (0.2126, 0.7152, 0.0722)

_STAT_MAX_PIXELS = 400_000
_STAT_MAX_FRAMES = 12
_STAT_BLOCK = 16
_APPLY_BLOCK = 8

_CLIP_LO = 1.0 / 255.0
_CLIP_HI = 254.0 / 255.0

_GAIN_MIN = 0.40
_GAIN_MAX = 2.50
_OFFSET_MAX = 0.12

_HIST_BINS = 32
_CUT_TOL = 2
_CUT_MIN_GAP = 4

_WARMUP_MIN = 0.005
_WARMUP_MAX = 2

_DESPIKE_MIN = 0.004

_RAMP_STRENGTH = 0.7
_RAMP_MIN_DEV = 0.0005

_HL_LO = 0.70
_HL_HI = 1.0

# ---------- 工具函数 ----------
def _work_device(images, use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return images.device

def _downsample_rgb(frames, long_side=_LONG_SIDE):
    rgb = frames[..., :3].clamp(0.0, 1.0).permute(0, 3, 1, 2)
    h, w = rgb.shape[-2:]
    scale = min(1.0, long_side / float(max(h, w)))
    if scale < 1.0:
        rgb = F.interpolate(
            rgb,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )
    return rgb.permute(0, 2, 3, 1)

def _luma(frames):
    w = frames.new_tensor(_LUMA_W)
    return torch.sum(frames[..., :3] * w, dim=-1)

def _segments_from_boundaries(n_frames, boundaries):
    edges = [0] + [int(b) for b in boundaries] + [int(n_frames)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

@torch.no_grad()
def _frame_stats(images, device):
    n = images.shape[0]
    rows = []
    for s in range(0, n, _STAT_BLOCK):
        e = min(s + _STAT_BLOCK, n)
        small = _downsample_rgb(images[s:e].to(device), _STAT_LONG_SIDE)
        k = small.shape[0]
        flat = small.reshape(k, -1, 3).float()
        y = torch.sum(flat * flat.new_tensor(_LUMA_W), dim=-1)
        m = y.shape[1]
        lo = int(m * 0.05)
        hi = max(lo + 1, int(m * 0.95))
        order = y.argsort(dim=1)[:, lo:hi]
        trimmed = torch.gather(
            flat, 1, order.unsqueeze(-1).expand(-1, -1, 3)
        )
        rows.append(
            torch.cat(
                [trimmed.mean(dim=1), y.median(dim=1).values.unsqueeze(-1)],
                dim=1,
            ).cpu()
        )
    return torch.cat(rows, dim=0).float()

def _refine_boundary(stats, t0, radius, n):
    lo = max(1, int(t0) - radius)
    hi = min(int(n), int(t0) + radius + 1)
    if hi <= lo:
        return int(t0)
    best_t, best_v = int(t0), -1.0
    for t in range(lo, hi):
        v = float((stats[t] - stats[t - 1]).abs().sum())
        if v > best_v:
            best_v, best_t = v, t
    return best_t

def _detect_step_boundaries(stats, window, z_thresh, min_gap, warmup=_WARMUP_MAX):
    n = stats.shape[0]
    k = max(2, int(window))
    warmup = max(0, int(warmup))
    if n < 2 * k + warmup + 2:
        return [], None
    raw = torch.zeros(n, dtype=torch.float32)
    for t in range(k, n - k - warmup + 1):
        left = stats[t - k:t].median(dim=0).values
        right = stats[t + warmup:t + warmup + k].median(dim=0).values
        raw[t] = float((right - left).abs().sum())
    valid = raw[k:n - k - warmup + 1]
    med = valid.median()
    mad = (valid - med).abs().median().clamp_min(1e-6)
    z = (raw - med) / (1.4826 * mad)
    cands = []
    for t in range(k, n - k - warmup + 1):
        if float(z[t]) < z_thresh:
            continue
        if t - 1 >= 0 and raw[t] < raw[t - 1]:
            continue
        if t + 1 < n and raw[t] < raw[t + 1]:
            continue
        cands.append((float(z[t]), int(t)))
    cands.sort(reverse=True)
    accepted = []
    for zv, t in cands:
        if all(abs(t - a) >= min_gap for a in accepted):
            accepted.append(_refine_boundary(stats, t, max(2, k // 2), n))
    return sorted(set(accepted)), z

@torch.no_grad()
def _sample_window_pixels(images, start, end, device):
    start = max(0, int(start))
    end = min(int(images.shape[0]), int(end))
    if end - start <= 0:
        return None
    idxs = list(range(start, end))
    if len(idxs) > _STAT_MAX_FRAMES:
        step = (len(idxs) - 1) / float(_STAT_MAX_FRAMES - 1)
        idxs = sorted({start + int(round(i * step))
                       for i in range(_STAT_MAX_FRAMES)})
    small = _downsample_rgb(images[idxs].to(device), _STAT_LONG_SIDE)
    flat = small.reshape(-1, 3).float()
    y = torch.sum(flat * flat.new_tensor(_LUMA_W), dim=-1)
    mask = (y > _CLIP_LO) & (y < _CLIP_HI)
    if int(mask.sum()) >= 2000:
        flat = flat[mask]
    n_pixels = flat.shape[0]
    if n_pixels > _STAT_MAX_PIXELS:
        step = n_pixels / _STAT_MAX_PIXELS
        idxs = torch.arange(0, n_pixels, step, device=flat.device).long()
        flat = flat[idxs[:_STAT_MAX_PIXELS]]
    if flat.shape[0] < 16:
        return None
    return flat

def _spd_pow(mat, power):
    w, v = torch.linalg.eigh(mat)
    w = w.clamp_min(1e-12).pow(power)
    return (v * w.unsqueeze(0)) @ v.transpose(-1, -2)

def _mkl_transform(src, dst, eps=1e-6):
    src = src.double()
    dst = dst.double()
    mx = src.mean(dim=0)
    my = dst.mean(dim=0)
    eye = torch.eye(3, dtype=src.dtype, device=src.device)
    cx = torch.cov(src.transpose(0, 1)) + eps * eye
    cy = torch.cov(dst.transpose(0, 1)) + eps * eye
    cx_h = _spd_pow(cx, 0.5)
    cx_ih = _spd_pow(cx, -0.5)
    mid = _spd_pow(cx_h @ cy @ cx_h, 0.5)
    a = cx_ih @ mid @ cx_ih
    b = my - a @ mx
    return a, b

def _affine_transform(src, dst):
    src = src.double()
    dst = dst.double()
    q = src.new_tensor([0.05, 0.5, 0.95])
    sq = torch.quantile(src, q, dim=0)
    dq = torch.quantile(dst, q, dim=0)
    span_s = (sq[2] - sq[0]).clamp_min(1e-4)
    span_d = (dq[2] - dq[0]).clamp_min(1e-4)
    gain = (span_d / span_s).clamp(0.2, 5.0)
    offset = dq[1] - gain * sq[1]
    return torch.diag(gain), offset

def _guard_transform(a, b, src, dst):
    diag = torch.tensor([float(a[i, i]) for i in range(3)])
    off = torch.tensor([float(b[i]) for i in range(3)])
    ok = (
        bool((diag > _GAIN_MIN).all())
        and bool((diag < _GAIN_MAX).all())
        and bool((off.abs() < _OFFSET_MAX).all())
    )
    if ok:
        return a, b, None
    ms = src.double().median(dim=0).values.clamp_min(1e-4)
    md = dst.double().median(dim=0).values
    gain = (md / ms).clamp(_GAIN_MIN, _GAIN_MAX)
    warn = (
        f"变换过激 (gain={[round(float(v), 3) for v in diag]}, "
        f"offset={[round(float(v), 3) for v in off]}) "
        f"-> 退回纯增益 {[round(float(v), 3) for v in gain]}"
    )
    return (
        torch.diag(gain),
        torch.zeros(3, dtype=gain.dtype, device=gain.device),
        warn,
    )

def _blend_toward_identity(a, b, strength):
    if strength >= 1.0:
        return a, b
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    return eye + (a - eye) * strength, b * strength

@torch.no_grad()
def _apply_transform(images, start, end, a, b, device):
    a32 = a.float().to(device).transpose(0, 1).contiguous()
    b32 = b.float().to(device)
    lw = torch.tensor(_LUMA_W, dtype=torch.float32, device=device)
    for s in range(int(start), int(end), _APPLY_BLOCK):
        e = min(s + _APPLY_BLOCK, int(end))
        view = images[s:e]
        rgb = view[..., :3].to(device)
        corrected = torch.matmul(rgb, a32) + b32
        luma = torch.sum(rgb * lw, dim=-1, keepdim=True)
        hw = ((luma - _HL_LO) / max(_HL_HI - _HL_LO, 1e-6)).clamp(0.0, 1.0)
        hw = hw * hw * (3.0 - 2.0 * hw)
        new = torch.lerp(corrected, rgb, hw).clamp(0.0, 1.0)
        view[..., :3] = new.to(view.device)

def _describe_transform(a, b):
    diag = [float(a[i, i]) for i in range(3)]
    gray = a.new_tensor([0.5, 0.5, 0.5])
    mapped = a @ gray + b
    w = a.new_tensor(_LUMA_W)
    luma_out = float(torch.sum(mapped * w))
    return diag, luma_out

@torch.no_grad()
def _apply_gain_curve(images, gain, device):
    n = int(images.shape[0])
    lw = torch.tensor(_LUMA_W, dtype=torch.float32, device=device)
    for s in range(0, n, _APPLY_BLOCK):
        e = min(s + _APPLY_BLOCK, n)
        view = images[s:e]
        rgb = view[..., :3].to(device)
        g = gain[s:e].to(device).view(-1, 1, 1, 3)
        corrected = rgb * g
        luma = torch.sum(rgb * lw, dim=-1, keepdim=True)
        hw = ((luma - _HL_LO) / max(_HL_HI - _HL_LO, 1e-6)).clamp(0.0, 1.0)
        hw = hw * hw * (3.0 - 2.0 * hw)
        new = torch.lerp(corrected, rgb, hw).clamp(0.0, 1.0)
        view[..., :3] = new.to(view.device)

@torch.no_grad()
def _correct_flatten(images, boundaries, strength, device):
    level = _frame_stats(images, device)[:, :3].clamp_min(1e-4)
    segs = _segments_from_boundaries(images.shape[0], boundaries)
    s0, e0 = segs[0]
    ref = level[s0:e0].median(dim=0).values
    need = ref.unsqueeze(0) / level
    gain = need.clamp(_GAIN_MIN, _GAIN_MAX)
    if strength < 1.0:
        gain = 1.0 + (gain - 1.0) * strength
    clipped = int(
        ((need > _GAIN_MAX) | (need < _GAIN_MIN)).any(dim=1).sum()
    )
    _apply_gain_curve(images, gain, device)
    return level, gain, clipped, ref

@torch.no_grad()
def _correct_exposure(images, seams, edit_points, mode, method, strength,
                      window, device, blend_frames=0):
    estimate = _mkl_transform if method == "mkl" else _affine_transform
    edit = sorted({int(x) for x in edit_points})
    micro = _segments_from_boundaries(images.shape[0], edit)
    seam_set = {int(x) for x in seams}
    logs = []
    if mode == "anchor":
        ref = _sample_window_pixels(images, micro[0][0], micro[0][1], device)
        if ref is None:
            print("[H3-Seam] 锚点段有效像素不足，跳过曝光修正")
            return logs
        for i in range(1, len(micro)):
            s, e = micro[i]
            if edit[i - 1] not in seam_set:
                continue
            cur = _sample_window_pixels(images, s, e, device)
            if cur is None:
                continue
            a, b = estimate(cur, ref)
            a, b, warn = _guard_transform(a, b, cur, ref)
            a, b = _blend_toward_identity(a, b, strength)
            _apply_transform(images, s, e, a, b, device)
            logs.append((edit[i - 1], s, e, a, b, warn))
        return logs
    for i in range(1, len(micro)):
        s, e = micro[i]
        bnd = edit[i - 1]
        if bnd not in seam_set:
            continue
        prev_start = micro[i - 1][0]
        warmup = _head_warmup_frames(images, bnd, window, device)
        left = _sample_window_pixels(
            images, max(prev_start, bnd - window), bnd, device)
        right = _sample_window_pixels(
            images, s + warmup, min(e, s + window + warmup), device)
        if left is None or right is None:
            continue
        a, b = estimate(right, left)
        a, b, warn = _guard_transform(a, b, right, left)
        a, b = _blend_toward_identity(a, b, strength)
        _apply_transform(images, s + warmup, e, a, b, device)
        _fix_warmup_frames(images, bnd, warmup, device)
        if blend_frames > 0:
            _despike_seam(images, bnd, device)
            _smooth_seam_ramp(images, bnd, blend_frames, device)
        logs.append((bnd, s, e, a, b, warn))
    return logs

@torch.no_grad()
def _head_warmup_frames(images, bnd, window, device, max_warmup=_WARMUP_MAX):
    n = int(images.shape[0])
    if bnd + 1 >= n:
        return 0
    vals = _luma_series(images, bnd - 1, min(n, bnd + window + max_warmup), device)
    if vals is None or vals.numel() < 3:
        return 0
    warmup = 0
    prev = float(vals[0])
    for k in range(max_warmup):
        i = 1 + k
        if i + 1 >= vals.numel():
            break
        cur = float(vals[i])
        nxt = float(vals[i + 1])
        jump_in = abs(cur - prev)
        jump_out = abs(nxt - cur)
        net = abs(nxt - prev)
        if jump_in > _WARMUP_MIN and jump_out > _WARMUP_MIN and net < 0.6 * jump_in:
            warmup += 1
            prev = cur
        else:
            break
    return warmup

@torch.no_grad()
def _fix_warmup_frames(images, bnd, warmup, device):
    n = int(images.shape[0])
    if warmup <= 0:
        return
    left_idx = bnd - 1
    right_idx = bnd + warmup
    if left_idx < 0 or right_idx >= n:
        return
    def _med(idx):
        small = _downsample_rgb(images[idx:idx + 1].to(device), _STAT_LONG_SIDE)
        return small[0].reshape(-1, 3).float().median(dim=0).values
    ml = _med(left_idx)
    mr = _med(right_idx)
    for k in range(warmup):
        idx = bnd + k
        alpha = (k + 1) / float(warmup + 1)
        target = ml + (mr - ml) * alpha
        cur = _med(idx)
        gain = (target / cur.clamp_min(1e-4)).clamp(_GAIN_MIN, _GAIN_MAX)
        rgb = images[idx, ..., :3].to(device)
        images[idx, ..., :3] = (rgb * gain.to(device)).clamp(0.0, 1.0).to(images.device)

@torch.no_grad()
def _despike_seam(images, bnd, device, radius=2, min_step=_DESPIKE_MIN):
    n = int(images.shape[0])
    lo = max(1, bnd - radius)
    hi = min(n - 1, bnd + radius)
    if hi - lo < 2:
        return
    prof = _luma_series(images, lo - 1, hi + 2, device)
    if prof is None or prof.numel() < 3:
        return
    for t in range(lo, hi + 1):
        i = t - (lo - 1)
        if i - 1 < 0 or i + 1 >= prof.numel():
            continue
        y = float(prof[i])
        prev = float(prof[i - 1])
        nxt = float(prof[i + 1])
        expect = 0.5 * (prev + nxt)
        dev = abs(y - expect)
        if dev < min_step:
            continue
        if abs(prev - nxt) > dev:
            continue
        scale = expect / max(y, 1e-4)
        scale = min(max(scale, _GAIN_MIN), _GAIN_MAX)
        if abs(scale - 1.0) < 1e-3:
            continue
        rgb = images[t, ..., :3].to(device)
        images[t, ..., :3] = (rgb * scale).clamp(0.0, 1.0).to(images.device)

@torch.no_grad()
def _smooth_seam_ramp(images, bnd, blend, device, strength=_RAMP_STRENGTH):
    n = int(images.shape[0])
    if blend <= 0:
        return
    la = bnd - blend - 1
    ra = bnd + blend
    if la < 0 or ra >= n or ra - la < 3:
        return
    left_rgb = images[la, ..., :3].to(device).float()
    right_rgb = images[ra, ..., :3].to(device).float()
    def luminance(rgb):
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lum_l = luminance(left_rgb).mean()
    lum_r = luminance(right_rgb).mean()
    span = ra - la
    for t in range(la + 1, ra):
        frac = (t - la) / float(span)
        target_lum = (1 - frac) * lum_l + frac * lum_r
        curr_rgb = images[t, ..., :3].to(device).float()
        curr_lum = luminance(curr_rgb).mean()
        if curr_lum < 1e-6:
            continue
        scale = target_lum / curr_lum
        scale = min(max(scale, 0.5), 2.0)
        new_rgb = curr_rgb * scale
        images[t, ..., :3] = new_rgb.clamp(0.0, 1.0).to(images.dtype)

@torch.no_grad()
def _detect_transient_frames(stats, window=3, z_thresh=2.5, min_gap=_CUT_MIN_GAP):
    n = stats.shape[0]
    if n < 2 * window + 3:
        return []
    y = stats[:, 3].float().cpu().numpy()
    out = []
    for t in range(window, n - window):
        window_seq = np.concatenate([y[t-window:t], y[t+1:t+window+1]])
        med = np.median(window_seq)
        mad = np.median(np.abs(window_seq - med))
        if mad < 1e-6:
            continue
        z = (y[t] - med) / (1.4826 * mad)
        if abs(z) > z_thresh:
            if t > 0 and t < n-1:
                avg_diff = (abs(y[t] - y[t-1]) + abs(y[t] - y[t+1])) / 2
                if avg_diff < abs(y[t] - med):
                    continue
            if not out or t - out[-1] >= min_gap:
                out.append(t)
    return out

def _classify_boundary(images, boundary, threshold, device):
    small = _downsample_rgb(images).to(device)
    y = _luma(small)
    if boundary < 1 or boundary >= small.shape[0]:
        return "clean", 0.0
    prev = y[boundary - 1]
    curr = y[boundary]
    delta = float(
        torch.mean(torch.abs(prev - curr)).item()
    )
    if boundary + 3 < small.shape[0]:
        after = [
            float(
                torch.mean(
                    torch.abs(prev - y[boundary + k])
                ).item()
            )
            for k in range(1, 4)
        ]
        if all(a > delta * 0.9 for a in after):
            return "exposure", delta
    if boundary + 2 < small.shape[0]:
        d2 = float(
            torch.mean(
                torch.abs(prev - y[boundary + 1])
            ).item()
        )
        if d2 < delta * 0.6:
            return "flash", delta
    if threshold <= delta < 0.12:
        return "motion", delta
    return "clean", delta

@torch.no_grad()
def _frame_histograms(images, device):
    n = int(images.shape[0])
    hists = []
    for s in range(0, n, _STAT_BLOCK):
        e = min(s + _STAT_BLOCK, n)
        small = _downsample_rgb(images[s:e].to(device), _STAT_LONG_SIDE).cpu()
        for t in range(small.shape[0]):
            flat = small[t].reshape(-1, 3).float()
            hs = []
            for c in range(3):
                idx = (flat[:, c] * _HIST_BINS).long().clamp(0, _HIST_BINS - 1)
                h = torch.bincount(idx, minlength=_HIST_BINS).float()
                hs.append(h)
            h = torch.cat(hs)
            tot = float(h.sum())
            if tot > 0:
                h = h / tot
            hists.append(h)
    return hists

def _hist_pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = float(a.norm() * b.norm())
    if denom < 1e-9:
        return 0.0
    return float((a * b).sum() / denom)

# ---------- SSIM 辅助函数 ----------
def _ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """计算两幅灰度图像的 SSIM (数值范围 [0,1])。"""
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=img1.dtype, device=img1.device) - size//2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g = g / g.sum()
        return g.outer(g)

    kernel = gaussian_window(window_size).unsqueeze(0).unsqueeze(0)
    mu1 = F.conv2d(img1, kernel, padding=window_size//2)
    mu2 = F.conv2d(img2, kernel, padding=window_size//2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size//2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

@torch.no_grad()
def _refine_cuts_by_similarity(cuts, images, device, search_radius=2):
    """
    使用与台阶检测相同的度量（RGB均值绝对差之和）来修正切点。
    在候选切点附近，选择该度量值最大的帧作为修正后的切点。
    """
    n_frames = images.shape[0]
    if n_frames < 2:
        return []
    
    
    stats = _frame_stats(images, device) 
    rgb_means = stats[:, :3]  
    diff = torch.zeros(n_frames, device=device)
    for t in range(1, n_frames):
        diff[t] = torch.sum(torch.abs(rgb_means[t] - rgb_means[t-1]))


    refined = []
    for c in cuts:
        if c < 1 or c >= n_frames - 1:
            refined.append(c)
            continue
        lo = max(1, c - search_radius)
        hi = min(n_frames - 1, c + search_radius)
        best_t = c
        best_val = diff[c].item()
        for t in range(lo, hi + 1):
            val = diff[t].item()
            if val > best_val:
                best_val = val
                best_t = t
        refined.append(best_t)

    refined = sorted(set(refined))
    refined = [c for c in refined if 0 < c < n_frames]
    return refined


# ---------- 镜头检测主函数 ----------
def _detect_shot_cuts(images, threshold, device, min_gap=_CUT_MIN_GAP):
    """
    使用 PySceneDetect 从临时视频文件检测切镜（兼容 Windows 文件锁定）
    """
    n_frames = images.shape[0]
    if n_frames < 2:
        return []

    tmp_path = None
    video = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = images[0].shape[:2]
        writer = cv2.VideoWriter(tmp_path, fourcc, 24.0, (w, h))
        if not writer.isOpened():
            raise IOError("无法创建临时视频文件")

        for img in images.cpu().numpy():
            bgr = (img[..., :3] * 255).astype(np.uint8)[..., ::-1]
            writer.write(bgr)
        writer.release()  

        video = open_video(tmp_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=float(threshold)))
        scene_manager.detect_scenes(video)

        scenes = scene_manager.get_scene_list()
        cuts = [scene[0].frame_num for scene in scenes[1:]]
        cuts = [c for c in cuts if 0 < c < n_frames]
        cuts = _refine_cuts_by_similarity(cuts, images, device)

        if video is not None:
            if hasattr(video, 'close'):
                video.close()
            elif hasattr(video, 'release'):
                video.release()
            video = None

        if tmp_path and os.path.exists(tmp_path):
            for _ in range(3):  
                try:
                    os.unlink(tmp_path)
                    break
                except PermissionError:
                    time.sleep(0.1)  
            else:
                print(f"[H3-Seam] 警告：临时文件 {tmp_path} 未能删除（可手动清理）")

        print(f"[H3-Seam] PySceneDetect 检测到切镜: {cuts}")
        return cuts

    except Exception as e:
        print(f"[H3-Seam] 临时文件检测失败: {e}")
        if video is not None:
            try:
                if hasattr(video, 'close'):
                    video.close()
                elif hasattr(video, 'release'):
                    video.release()
            except:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        cuts = _detect_shot_cuts_histogram(images, threshold, device, min_gap)
        cuts = _refine_cuts_by_similarity(cuts, images, device)
        return cuts

       
def _detect_shot_cuts_histogram(images, threshold, device, min_gap=_CUT_MIN_GAP):
    n = int(images.shape[0])
    if n < 2:
        return []
    hists = _frame_histograms(images, device)
    corr = [1.0] * n
    for t in range(1, n):
        corr[t] = _hist_pearson(hists[t - 1], hists[t])
    cuts = []
    for t in range(1, n):
        if corr[t] >= threshold:
            continue
        left = corr[t - 1] if t - 1 >= 1 else 1.0
        right = corr[t + 1] if t + 1 < n else 1.0
        if corr[t] > left or corr[t] > right:
            continue
        if cuts and t - cuts[-1] < min_gap:
            continue
        cuts.append(t)
    return cuts

def _is_near_cut(b, cuts, tol=_CUT_TOL):
    return any(abs(int(b) - int(c)) <= tol for c in cuts)

# ---------- 光流相关函数 ----------
def _image_to_gray(image):
    return (
        image[..., :3]
        .mul(image.new_tensor(_LUMA_W))
        .sum(dim=-1)
        .unsqueeze(0)
        .unsqueeze(0)
    )

def _make_flow_grid(h, w, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    gx = 2.0 * xx / max(w - 1, 1) - 1.0
    gy = 2.0 * yy / max(h - 1, 1) - 1.0
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)

def _warp(image, flow):
    _, _, h, w = image.shape
    base = _make_flow_grid(
        h,
        w,
        image.device,
        image.dtype,
    )
    dx = flow[:, 0] * 2.0 / max(w - 1, 1)
    dy = flow[:, 1] * 2.0 / max(h - 1, 1)
    grid = base.clone()
    grid[..., 0] += dx
    grid[..., 1] += dy
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

def _gradient_x(x):
    return F.pad(
        x[..., :, 1:] - x[..., :, :-1],
        (0, 1, 0, 0),
    )

def _gradient_y(x):
    return F.pad(
        x[..., 1:, :] - x[..., :-1, :],
        (0, 0, 0, 1),
    )

def _smooth_flow(flow):
    return F.avg_pool2d(
        F.pad(flow, (1, 1, 1, 1), mode="replicate"),
        kernel_size=3,
        stride=1,
    )

@torch.no_grad()
def _gpu_optical_flow(
    source,
    target,
    iterations=12,
    pyramid_levels=3,
):
    src = _image_to_gray(source)
    tgt = _image_to_gray(target)
    original_h, original_w = src.shape[-2:]
    src_pyr = [src]
    tgt_pyr = [tgt]
    for _ in range(max(1, pyramid_levels - 1)):
        h, w = src_pyr[-1].shape[-2:]
        if min(h, w) < 64:
            break
        src_pyr.append(
            F.avg_pool2d(
                src_pyr[-1],
                kernel_size=2,
                stride=2,
            )
        )
        tgt_pyr.append(
            F.avg_pool2d(
                tgt_pyr[-1],
                kernel_size=2,
                stride=2,
            )
        )
    flow = None
    for level in reversed(range(len(src_pyr))):
        s = src_pyr[level]
        t = tgt_pyr[level]
        h, w = s.shape[-2:]
        if flow is None:
            flow = torch.zeros(
                (1, 2, h, w),
                device=s.device,
                dtype=s.dtype,
            )
        else:
            flow = F.interpolate(
                flow,
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            ) * 2.0
        ix = _gradient_x(t)
        iy = _gradient_y(t)
        for _ in range(iterations):
            warped = _warp(s, flow)
            error = t - warped
            scale = error.abs().mean().clamp_min(0.01)
            error = error / (1.0 + 4.0 * scale)
            ix_w = _warp(ix, flow)
            iy_w = _warp(iy, flow)
            denom = (
                ix_w * ix_w
                + iy_w * iy_w
                + 0.01
            )
            du = error * ix_w / denom
            dv = error * iy_w / denom
            update = torch.cat((du, dv), dim=1)
            update = 0.65 * update + 0.35 * _smooth_flow(update)
            flow = flow + update
            flow = flow.clamp(-32.0, 32.0)
    if flow.shape[-2:] != (original_h, original_w):
        flow = F.interpolate(
            flow,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=True,
        )
    return flow

def _local_color_match(
    orig,
    reference,
    grid=8,
    strength=0.65,
):
    if strength <= 0:
        return orig
    x = orig[..., :3].permute(2, 0, 1).unsqueeze(0)
    r = reference[..., :3].permute(2, 0, 1).unsqueeze(0)
    h, w = x.shape[-2:]
    gh = max(1, min(grid, h))
    gw = max(1, min(grid, w))
    xm = F.adaptive_avg_pool2d(x, (gh, gw))
    rm = F.adaptive_avg_pool2d(r, (gh, gw))
    diff = F.interpolate(
        rm - xm,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    corrected = (x + diff * strength).clamp(0.0, 1.0)
    out = orig.clone()
    out[..., :3] = corrected.squeeze(0).permute(1, 2, 0)
    return out

@torch.no_grad()
def _warp_blend_seam(
    images,
    boundary,
    window,
    flow_strength,
    blend_strength,
    flow_iterations,
    flow_pyramid,
    device,
):
    out = images
    if boundary < 1 or boundary >= out.shape[0]:
        return out
    window = max(1, int(window))
    left_idx = boundary - 1
    right_idx = min(
        boundary + window,
        out.shape[0] - 1,
    )
    left = out[left_idx].to(device).clone()
    right = out[right_idx].to(device).clone()
    h, w = left.shape[:2]
    scale = min(
        1.0,
        768.0 / float(max(h, w)),
    )
    if scale < 1.0:
        left_small = F.interpolate(
            left[..., :3].permute(2, 0, 1).unsqueeze(0),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )
        right_small = F.interpolate(
            right[..., :3].permute(2, 0, 1).unsqueeze(0),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )
        left_flow = left_small.squeeze(0).permute(1, 2, 0)
        right_flow = right_small.squeeze(0).permute(1, 2, 0)
        flow_small = _gpu_optical_flow(
            right_flow,
            left_flow,
            iterations=flow_iterations,
            pyramid_levels=flow_pyramid,
        )
        flow = F.interpolate(
            flow_small,
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        ) / max(scale, 1e-6)
    else:
        flow = _gpu_optical_flow(
            right,
            left,
            iterations=flow_iterations,
            pyramid_levels=flow_pyramid,
        )
    flow = flow * float(flow_strength)
    right_rgb = right[..., :3].permute(2, 0, 1).unsqueeze(0)
    warped_right = _warp(
        right_rgb,
        flow,
    )
    warped_right = (
        warped_right.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
    )
    start = max(
        0,
        boundary - window,
    )
    end = min(
        out.shape[0],
        boundary + window + 1,
    )
    for idx in range(start, end):
        orig = out[idx].to(device).clone()
        dist = abs(
            idx - (boundary - 0.5)
        )
        strength = max(
            0.0,
            1.0 - dist / float(window + 0.5),
        )
        if idx < boundary:
            alpha = 0.08 * strength
        else:
            denom = max(
                1,
                right_idx - boundary,
            )
            alpha = (
                float(idx - boundary + 1)
                / float(denom + 1)
            )
            alpha = min(
                1.0,
                alpha,
            )
        if idx < boundary:
            reference = torch.lerp(
                left,
                warped_right,
                0.15 * strength,
            )
        else:
            reference = torch.lerp(
                left,
                warped_right,
                alpha,
            )
        matched = _local_color_match(
            orig,
            reference,
            grid=8,
            strength=blend_strength * strength,
        )
        mix = (
            0.30 * strength
            if idx != boundary
            else 0.60
        )
        result = torch.lerp(
            orig,
            matched,
            mix,
        )
        out[idx] = result.to(out.device)
    return out

@torch.no_grad()
def _luma_series(images, start, end, device):
    start = max(0, int(start))
    end = min(int(images.shape[0]), int(end))
    if end <= start:
        return None
    vals = []
    for s in range(start, end, _STAT_BLOCK):
        e = min(s + _STAT_BLOCK, end)
        small = _downsample_rgb(images[s:e].to(device), _STAT_LONG_SIDE)
        y = _luma(small).flatten(1)
        vals.append(y.median(dim=1).values.float().cpu())
    return torch.cat(vals, dim=0)

@torch.no_grad()
def _boundary_step(images, b, window, device):
    b = int(b)
    lo = max(0, b - window)
    hi = min(int(images.shape[0]), b + window)
    left = _luma_series(images, lo, b, device)
    right = _luma_series(images, b, hi, device)
    if left is None or right is None or left.numel() < 1 or right.numel() < 1:
        return 0.0
    return float(right.median() - left.median())

@torch.no_grad()
def _cell_luma(images, start, end, device, cells=3):
    start = max(0, int(start))
    end = min(int(images.shape[0]), int(end))
    if end <= start:
        return None
    idxs = list(range(start, end))
    if len(idxs) > 8:
        step = (len(idxs) - 1) / 7.0
        idxs = sorted({start + int(round(i * step)) for i in range(8)})
    small = _downsample_rgb(images[idxs].to(device), 192)
    y = _luma(small).unsqueeze(1)
    pooled = F.adaptive_avg_pool2d(y, (cells, cells)).squeeze(1)
    return pooled.mean(dim=0).float().cpu()

@torch.no_grad()
def _diagnose(images, boundaries, window, device, tag):
    pass

@torch.no_grad()
def _spatial_similarity(images, left_start, left_end, right_start, right_end, device):
    cpu_images = images.cpu()
    left_start = max(0, int(left_start))
    left_end = min(int(images.shape[0]), int(left_end))
    right_start = max(0, int(right_start))
    right_end = min(int(images.shape[0]), int(right_end))
    if left_end <= left_start or right_end <= right_start:
        return 0.0
    left_idxs = list(range(left_start, left_end))
    right_idxs = list(range(right_start, right_end))
    if len(left_idxs) > 3:
        step = (len(left_idxs) - 1) / 2.0
        left_idxs = sorted({left_start + int(round(i * step)) for i in range(3)})
    if len(right_idxs) > 3:
        step = (len(right_idxs) - 1) / 2.0
        right_idxs = sorted({right_start + int(round(i * step)) for i in range(3)})
    left_small = _downsample_rgb(cpu_images[left_idxs], 64)
    right_small = _downsample_rgb(cpu_images[right_idxs], 64)
    left_gray = _luma(left_small).flatten()
    right_gray = _luma(right_small).flatten()
    min_len = min(left_gray.numel(), right_gray.numel())
    if min_len < 10:
        return 0.0
    left_gray = left_gray[:min_len]
    right_gray = right_gray[:min_len]
    return _hist_pearson(left_gray, right_gray)

# ---------- 预设与工具提示 ----------
_PHOTO_TOOLTIP = (
    "色彩/曝光处理档位。"
)

_SEAM_TOOLTIP = (
    "接缝连续性处理。基于光流对齐+局部融合，修补运动/结构不连续。"
)

_FLASH_TOOLTIP = (
    "闪帧处理，若画面有闪电、爆炸等合理快速明暗变化，抑制会削平这些效果。"
)

_CUT_DETECT_TOOLTIP = (
    "将视频切为多个镜头，修正仅在镜头内进行，真实切镜处跳过。关闭则处理全部分段边界。"
)

_PHOTO_LEVELS = {
    "off": None,
    "low": {"mode": "de-step", "method": "affine", "strength": 0.5, "window": 3},
    "medium": {"mode": "de-step", "method": "affine", "strength": 1.0, "window": 3},
    "high": {"mode": "de-step", "method": "mkl", "strength": 1.0, "window": 8},
    "max": {"mode": "flatten", "method": "mkl", "strength": 1.0, "window": 3},
}

_SEAM_LEVELS = {
    "off": None,
    "low": {"window": 1, "flow": 0.40, "blend": 0.35, "iters": 8, "pyramid": 3},
    "medium": {"window": 3, "flow": 0.75, "blend": 0.5, "iters": 12, "pyramid": 3},
    "high": {"window": 6, "flow": 1.00, "blend": 0.8, "iters": 16, "pyramid": 4},
    "max": {"window": 9, "flow": 1.20, "blend": 1.0, "iters": 20, "pyramid": 4},
}

_LEVEL_ORDER = ["off", "low", "medium", "high", "max"]

_LOCAL_THRESHOLD = 0.02
_DETECT_Z = 5.5
_DETECT_WINDOW = 6

# ---------- ComfyUI 节点 ----------
class H3SeamCorrection(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3SeamCorrection",
            display_name="Minimax_H3_Seam_Correction",
            category="MinimaxH3_AutoContext",
            description=(
                "Minimax H3 段间接缝修正："
                "分段全局曝光/色彩对齐 (MKL) + "
                "接缝局部光流 warp/blend"
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="解码后的完整视频帧 (接 VAE Decode 输出)",
                ),
                io.Combo.Input(
                    "fix_color_preset",
                    options=_LEVEL_ORDER,
                    default="medium",
                    tooltip=_PHOTO_TOOLTIP,
                ),
                io.Combo.Input(
                    "fix_motion_preset",
                    options=_LEVEL_ORDER,
                    default="off",
                    tooltip=_SEAM_TOOLTIP,
                ),
                io.Boolean.Input(
                    "fix_flash",
                    default=False,
                    tooltip=_FLASH_TOOLTIP,
                ),
                io.Float.Input(
                    "flash_threshold",
                    default=0.30,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="瞬态修正筛选阈值,值越小越激进,推荐 0.20~0.40。",
                ),

                io.Boolean.Input(
                    "cut_detection",
                    default=True,
                    tooltip=_CUT_DETECT_TOOLTIP,
                ),
                
                io.Float.Input(
                    "cut_threshold",
                    default=15.0,
                    min=5.0,
                    max=50.0,
                    step=0.5,
                    tooltip="镜头检测敏感度，值越小越敏感，推荐 10~20",
                ),
                io.Int.Input(
                    "blend_frames",
                    default=2,
                    min=0,
                    max=8,
                    step=1,
                    tooltip=(
                        "接缝处电平渐变窗口 (帧)。值越大过渡越缓、"
                        "但运动大的镜头过大会带来轻微糊感。"
                    ),
                ),
                io.Boolean.Input(
                    "use_gpu",
                    default=True,
                    tooltip="使用 CUDA GPU 进行统计、色彩变换与光流",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        fix_color_preset="medium",
        fix_motion_preset="off",
        fix_flash=False,
        flash_threshold=0.30,
        blend_frames=2,
        use_gpu=True,
        cut_detection=True,
        cut_threshold=15.0,
    ) -> io.NodeOutput:

        if images is None or images.shape[0] < 2:
            return io.NodeOutput(images)

        photo = _PHOTO_LEVELS.get(fix_color_preset)
        seam = _SEAM_LEVELS.get(fix_motion_preset)
        flash_cfg = seam or _SEAM_LEVELS["medium"]

        stat_window = int(photo["window"]) if photo else 3

        n_frames = int(images.shape[0])
        device = _work_device(images, use_gpu)

        if device.type != "cuda":
            print(
                "[H3-Seam] WARNING: GPU disabled or "
                "CUDA unavailable; 统计与光流将在输入设备上运行"
            )

        print(
            "[H3-Seam] "
            f"device={device}, frames={n_frames}, "
            f"color={fix_color_preset}"
            + (f"({photo['mode']}/{photo['method']}, "
               f"strength={photo['strength']:.2f}, window={stat_window})"
               if photo else "")
            + f", motion={fix_motion_preset}, "
            f"flash={'on' if fix_flash else 'off'}, "
            f"flash_threshold={flash_threshold:.2f}, "
            f"blend={blend_frames}"
        )

        min_gap = max(4, _DETECT_WINDOW)
        stats = _frame_stats(images, device)

        steps, z = _detect_step_boundaries(stats, _DETECT_WINDOW, _DETECT_Z, min_gap)
        steps = [s + 1 for s in steps if s + 1 < n_frames]

        boundaries = sorted(set(steps))
        transients = []

        print(
            f"[H3-Seam] 边界来源=内容检测: 台阶={steps} -> 合并={boundaries}"
        )

        if not boundaries:
            print("[H3-Seam] 未检测到接缝，原样输出")
            return io.NodeOutput(images)

        if cut_detection:
            cuts = _detect_shot_cuts(images, cut_threshold, device)
            if cuts:
                seams = []
                edit_points = sorted(cuts)
                for c in cuts:
                    if c < 3 or c >= images.shape[0] - 3:
                        continue
                    corr = _spatial_similarity(
                        images,
                        left_start=max(0, c - 3),
                        left_end=c,
                        right_start=c,
                        right_end=min(images.shape[0], c + 3),
                        device=device
                    )
                    if corr > 0.60:
                        seams.append(c)
                        print(f"[H3-Seam] 边界 {c} 确认为同镜头色差 (空间相关系数={corr:.3f}) -> 启用色彩修正")
                    else:
                        print(f"[H3-Seam] 边界 {c} 为真实切镜 (空间相关系数={corr:.3f}) -> 跳过修正")
                if not seams:
                    print("[H3-Seam] 所有边界均为真实切镜，无色彩修正执行")
            else:
                print("[H3-Seam] 镜头检测无输出，回退到内部台阶检测")
                seams = [b for b in boundaries]
                edit_points = sorted({int(b) for b in boundaries})
        else:
            seams = list(boundaries)
            edit_points = sorted({int(b) for b in boundaries})

        if photo is None and seam is None and not fix_flash:
            print(
                "[H3-Seam] 三项处理均为 off -> 画面原样输出"
            )
            return io.NodeOutput(images)

        out = images.clone()

        if photo is not None and photo["mode"] == "flatten":
            before = _luma_series(out, 0, n_frames, device)
            level, gain, clipped, ref = _correct_flatten(
                out, boundaries, float(photo["strength"]), device)
            after = _luma_series(out, 0, n_frames, device)
            print(
                f"[H3-Seam] color[max] 全片逐帧电平归一化 -> 锚点电平="
                f"({float(ref[0]):.4f},{float(ref[1]):.4f},{float(ref[2]):.4f})"
            )
            print(
                f"[H3-Seam] color[max] 全片亮度极差 "
                f"{float(before.max() - before.min()):.4f} -> "
                f"{float(after.max() - after.min()):.4f}  "
                f"(增益范围 {float(gain.min()):.2f}~{float(gain.max()):.2f})"
            )
            if clipped:
                print(
                    f"[H3-Seam] color[max] 警告: {clipped}/{n_frames} 帧所需增益"
                    f"超出安全上限 {_GAIN_MAX}x，只做了部分提亮。"
                    f"这些帧已被压到近黑、像素信息已丢失，后处理无法恢复 —— "
                    f"需要在生成阶段解决 (重跑该段 / 加大 context_frames / "
                    f"用参考图锚定曝光)"
                )

        elif photo is not None:
            before = {
                int(b): _boundary_step(out, b, stat_window, device)
                for b in seams
            }
            logs = _correct_exposure(
                out,
                seams,
                edit_points,
                photo["mode"],
                photo["method"],
                float(photo["strength"]),
                stat_window,
                device,
                int(blend_frames),
            )
            if not logs:
                print("[H3-Seam] 色彩曝光对齐: 未产生有效修正")
            for bnd, s, e, a, b, warn in logs:
                diag, luma_out = _describe_transform(a, b)
                after = _boundary_step(out, bnd, stat_window, device)
                print(
                    f"[H3-Seam] color 帧{bnd} -> 段[{s},{e}): "
                    f"gain=({diag[0]:.3f},{diag[1]:.3f},{diag[2]:.3f}) "
                    f"亮度台阶 {before.get(int(bnd), 0.0):+.4f} -> {after:+.4f}"
                )
                if warn:
                    print(f"[H3-Seam] 帧{bnd} 安全闸门: {warn}")

        if fix_flash:
            stats_after = _frame_stats(out, device)
            transients_after = _detect_transient_frames(
                stats_after,
                window=1,
                z_thresh=1.2,
                min_gap=min_gap
            )
            if transients_after:
                device = out.device
                lw = torch.tensor(_LUMA_W, device=device)
                area_threshold = max(0.0, min(1.0, float(flash_threshold)))
                for f in transients_after:
                    if 0 < f < out.shape[0] - 1:
                        curr = out[f, ..., :3].float().to(device)
                        prev = out[f-1, ..., :3].float().to(device)
                        nxt = out[f+1, ..., :3].float().to(device)
                        luma_curr = torch.sum(curr * lw, dim=-1, keepdim=True)
                        luma_prev = torch.sum(prev * lw, dim=-1, keepdim=True)
                        luma_nxt = torch.sum(nxt * lw, dim=-1, keepdim=True)
                        luma_target = (luma_prev + luma_nxt) / 2.0
                        diff = (luma_curr - luma_target).abs() / (luma_target + 1e-6)
                        mask = (diff > 0.10).float()
                        abnormal_ratio = mask.mean().item()
                        if abnormal_ratio < area_threshold:
                            continue
                        avg_adj = (prev + nxt) / 2.0
                        result = curr * (1 - mask) + avg_adj * mask
                        out[f, ..., :3] = result.clamp(0.0, 1.0).to(out.dtype)
                        print(f"[H3-Seam] 局部修正: 帧{f}，异常像素占比 {abnormal_ratio:.2%}")
            else:
                print("[H3-Seam] color后未检测到瞬态")

        if seam is not None or fix_flash:
            actions = []
            for b in seams:
                kind, delta = _classify_boundary(
                    out, b, _LOCAL_THRESHOLD, device)
                if delta < _LOCAL_THRESHOLD:
                    continue
                if kind == "flash":
                    if not fix_flash:
                        continue
                    cfg = flash_cfg
                    label = "fix_flash"
                else:
                    if seam is None:
                        continue
                    cfg = seam
                    label = f"motion[{fix_motion_preset}]"
                out = _warp_blend_seam(
                    out,
                    b,
                    window=cfg["window"],
                    flow_strength=cfg["flow"],
                    blend_strength=cfg["blend"],
                    flow_iterations=cfg["iters"],
                    flow_pyramid=cfg["pyramid"],
                    device=device,
                )
                actions.append((b, kind, f"{label} delta={delta:.3f}"))
            for b, kind, action in actions:
                print(f"[H3-Seam] 帧{b}: {kind} -> {action}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        if out.device != images.device or out.dtype != images.dtype:
            out = out.to(images.device).to(images.dtype)
        return io.NodeOutput(out)

NODE_CLASS_MAPPINGS = {
    "H3SeamCorrection": H3SeamCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SeamCorrection":
        "Minimax_H3_Seam_Correction",
}