"""
seam_correction.py - Minimax H3 Seam Correction

分段长视频的段间接缝修正。两级架构：

  [阶段 1] 曝光/色彩 —— 分段全局对齐 (本次新增，主力)
      同一连续镜头下的段间曝光不一致，本质是时间信号里的"阶跃"
      (level shift / DC step)，必须对整段施加校正，而不是在接缝
      附近做局部融合 —— 后者只会把硬台阶变成几帧内的斜坡，产生
      "呼吸/pumping"感，台阶之后整段仍然偏亮或偏暗。

      变换用 MKL (Monge-Kantorovich Linear) 闭式解：匹配均值 +
      协方差的最优 3x3 线性变换，能处理跨通道色偏 (偏暖/偏绿)，
      严格强于逐通道 gain/offset。纯 torch 实现，不引入新依赖。

  [阶段 2] 运动/闪光 —— 接缝局部 warp/blend (原有逻辑，保留)
      GPU 光流对齐 + 局部融合。放在阶段 1 之后执行：直流台阶已被
      扣除，此时边界上的残差才是真正的结构性不连续，局部融合的
      修正量不再掺进曝光差异。

边界来源 (纯内容检测，不依赖采样端 info)：
   - 单帧瞬态 (flash / 段首 warm-up)：跳入跳出都大、净值小，
     精确定位"下一段新增内容的首帧"；
   - 曝光台阶 (level shift)：右窗跳过 warm-up 帧的窗口分位台阶检测。
     连续镜头的内容均值平滑演化，因此"窗口中位数的跳变"是干净的
     曝光台阶信号，而"帧差局部尖峰"在持续运动下必然失效。

镜头闸门 (cut_detection)：修正前对整个视频逐帧做切镜检测
  (相邻帧 RGB 直方图相关度)，把时间线切成镜头；曝光/色彩/运动修正
  只在同一个镜头内部进行，真实切镜处 (包括分段内部的切镜) 一律跳过，
  不会把一个镜头的修正量带到其它镜头。

依赖：PyTorch + ComfyUI。不需要 OpenCV，不需要 color-matcher。
如需非参数化色彩迁移 (hm-mvgd-hm 等)，可在本节点之后串接
KJNodes 的 ColorMatchV2。
"""

import torch
import torch.nn.functional as F
from comfy_api.latest import io

_LONG_SIDE = 320
_STAT_LONG_SIDE = 256
_LUMA_W = (0.2126, 0.7152, 0.0722)

# 统计采样上限：控制显存与耗时
_STAT_MAX_PIXELS = 400_000
_STAT_MAX_FRAMES = 12
_STAT_BLOCK = 16
_APPLY_BLOCK = 8

# 只排除硬截断像素 (纯黑/纯白)。
# [重要] 不能用 0.02/0.98 这种固定区间：暗场素材整帧亮度可能只有 0.003~0.05，
# 固定下限会把几乎整幅画面排除掉，只剩极少量高光参与统计 -> 估出荒谬的变换。
_CLIP_LO = 1.0 / 255.0
_CLIP_HI = 254.0 / 255.0

# 变换合理性闸门：超出范围就退回纯增益 (见 _guard_transform)
_GAIN_MIN = 0.40
_GAIN_MAX = 2.50
_OFFSET_MAX = 0.12

# 切镜检测 (镜头边界)：RGB 直方图分箱数 / 容错帧数 / 最小间隔
_HIST_BINS = 32
_CUT_TOL = 2
_CUT_MIN_GAP = 4

# 瞬态 (flash / 段首 warm-up) 检测与修正
_WARMUP_MIN = 0.02
_WARMUP_MAX = 2

# 接缝残差尖峰修平 (de-step 硬切后) 的判定阈值
_DESPIKE_MIN = 0.004

# 接缝斜坡平滑 (blend_frames) 的拉回强度与最小残差
_RAMP_STRENGTH = 0.7
_RAMP_MIN_DEV = 0.0005

# 高光保护：线性变换会按比例把高光 (车灯等镜面高光) 一起压暗变灰。
# luma 低于 _HL_LO 的像素接受完整校正，高于 _HL_HI 的像素基本保持原样。
_HL_LO = 0.70
_HL_HI = 0.95


# ============================================================
# Basic utilities
# ============================================================

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
    """边界帧号 -> [(start, end), ...] 段区间"""
    edges = [0] + [int(b) for b in boundaries] + [int(n_frames)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


# ============================================================
# 逐帧鲁棒统计量 + 台阶检测
# ============================================================

@torch.no_grad()
def _frame_stats(images, device):
    """逐帧鲁棒统计量 -> [N, 4]

    前 3 维是 RGB 截尾均值 (去掉最亮/最暗 5% 像素，抗高光与死黑干扰)，
    第 4 维是亮度中位数。用中位数/截尾而非全图平均，是为了让统计量
    对局部运动不敏感，只跟随整体曝光电平。
    """
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
    """精定位：窗口中位数只能粗定位 (±window/2)，因为窗口跨过台阶时
    中位数是渐变的。真正的台阶帧上，单帧差 |stats[t]-stats[t-1]| 会
    局部抬高，用它在邻域内取 argmax 把边界钉准。
    """
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
    """窗口分位台阶检测 (warm-up 感知)。

    对每个候选切点 t，比较 median(stats[t-K:t]) 与
    median(stats[t+warmup : t+warmup+K])。右窗跳过紧邻边界的 warmup 帧，
    避免段首单帧瞬态 (生成模型 re-anchor 时的偏亮/偏暗帧) 把台阶信号
    抹平或把边界钉错。连续镜头的内容均值是平滑演化的，因此这个差值在
    正常帧上很小、在曝光台阶处突然变大。用 MAD 归一化成 z-score 自校准。

    粗定位后用 _refine_boundary 做单帧级精定位。
    """
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
        # 只取局部极大值，避免一个台阶被报成一片
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


# ============================================================
# 全局色彩变换估计 (MKL / 逐通道仿射)
# ============================================================

@torch.no_grad()
def _sample_window_pixels(images, start, end, device):
    """取 [start, end) 帧的有效像素 -> [M, 3]

    MKL/仿射都只用分布统计量 (均值/协方差/分位)，不需要逐像素配对，
    所以两侧窗口各自独立采样、各自剔除饱和像素是安全的。
    """
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

    if flat.shape[0] > _STAT_MAX_PIXELS:
        sel = torch.randperm(flat.shape[0], device=flat.device)
        flat = flat[sel[:_STAT_MAX_PIXELS]]

    if flat.shape[0] < 16:
        return None

    return flat


def _spd_pow(mat, power):
    """对称正定矩阵的实数次幂 (0.5 / -0.5)，走特征分解"""
    w, v = torch.linalg.eigh(mat)
    w = w.clamp_min(1e-12).pow(power)
    return (v * w.unsqueeze(0)) @ v.transpose(-1, -2)


def _mkl_transform(src, dst, eps=1e-6):
    """Monge-Kantorovich 线性色彩迁移 (闭式解)。

    解使 src 的均值+协方差匹配到 dst 的最优线性变换：
        A = Cx^(-1/2) (Cx^(1/2) Cy Cx^(1/2))^(1/2) Cx^(-1/2)
        y = A (x - mean_x) + mean_y

    返回 (A[3,3], b[3])，使 y ≈ x @ A.T + b
    """
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
    """逐通道鲁棒仿射：用 5/50/95 分位定 gain/offset。

    比 MKL 保守 —— 不做跨通道混合，因此不会引入色相偏移，
    适合只有整体亮度差、担心 MKL 过度校正的场景。
    """
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
    """变换合理性闸门。

    MKL 匹配的是均值+协方差，而协方差在很大程度上由**画面内容构成**决定，
    不是曝光决定。当边界两侧内容差异较大 (运动、遮挡、明暗区域进出画面) 时，
    MKL 会把内容差异当成色彩差异，解出荒谬的"大增益 + 大负偏移"对比度拉伸。

    这里检测这种情况并退回纯增益 (逐通道中位数比值)：只改亮度不改对比度，
    宁可修不足也不要修坏。

    返回 (a, b, warn_or_None)
    """
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
    """把变换按 strength 插值回单位变换。

    等价于像素域 lerp(x, Ax+b, s)，但只需算一次矩阵：
        lerp(x, Ax+b, s) = (I + s(A-I)) x + s*b
    """
    if strength >= 1.0:
        return a, b
    eye = torch.eye(3, dtype=a.dtype, device=a.device)
    return eye + (a - eye) * strength, b * strength


@torch.no_grad()
def _apply_transform(images, start, end, a, b, device):
    """在 [start, end) 帧上原地施加 y = x @ A.T + b (分块搬运，控制显存)。

    带高光保护：线性变换按比例压暗整帧，会把车灯等镜面高光一起压暗变灰。
    这里按像素亮度把结果向原图回退 —— 暗部接受完整校正，接近白色的
    高光几乎不动。
    """
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
        hw = hw * hw * (3.0 - 2.0 * hw)  # smoothstep，过渡更柔和
        new = torch.lerp(corrected, rgb, hw).clamp(0.0, 1.0)

        view[..., :3] = new.to(view.device)


def _describe_transform(a, b):
    """把变换翻译成人看得懂的量：中灰点的亮度变化 + 通道增益"""
    diag = [float(a[i, i]) for i in range(3)]
    gray = a.new_tensor([0.5, 0.5, 0.5])
    mapped = a @ gray + b
    w = a.new_tensor(_LUMA_W)
    luma_out = float(torch.sum(mapped * w))
    return diag, luma_out


# ============================================================
# 阶段 1：分段全局曝光/色彩对齐
# ============================================================

@torch.no_grad()
def _apply_gain_curve(images, gain, device):
    """逐帧施加 per-channel 增益 gain[N,3] (原地，分块搬运)，带高光保护。

    线性增益按比例压暗整帧，会把车灯等镜面高光一起压暗变灰。按像素
    亮度向原图回退：暗部接受完整增益，接近白色的高光几乎不动。
    """
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
        hw = hw * hw * (3.0 - 2.0 * hw)  # smoothstep
        new = torch.lerp(corrected, rgb, hw).clamp(0.0, 1.0)

        view[..., :3] = new.to(view.device)


@torch.no_grad()
def _correct_flatten(images, boundaries, strength, device):
    """全片逐帧电平归一化：把每帧的逐通道电平拉到第 1 段的电平。

    [必读] 这是"最强"档，代价明确：
    1. 它会消除所有亮度变化 —— 包括画面本身合理的明暗变化 (天黑/灯灭/进隧道)。
       因为"生成漂移"和"剧情要的变暗"在像素里是同一个信号，无法区分。
    2. 对已经被压到近黑的帧 (电平 < 约 0.01)，所需增益会超出安全上限。
       那些帧的像素信息已经被压掉了 —— 后处理无法恢复，强行拉只会得到
       噪点和色带。这类帧会被统计并在日志中报告。

    返回 (level, gain, n_clipped, ref)
    """
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
                      window, device, debug, blend_frames=0):
    """分段全局对齐 (镜头感知)。images 会被原地修改。

    seams:      同镜头内的段间接缝 (应修正的 chunk 边界)。
    edit_points: seams ∪ cuts 的全部边界，用于把时间线切成
                 "单镜头微段"，保证修正量绝不会越过切镜。

    de-step: 逐 seam 顺序处理，左窗取"已修正"的上一微段尾部 ->
             校正量天然累积 (仅在同一镜头内)，只消除台阶、保留
             内容本身的明暗趋势。
    anchor:  每个微段整体对齐到第 1 段 -> 抗累积漂移，但会压制
             内容合理的明暗变化。
    """
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

    # de-step
    for i in range(1, len(micro)):
        s, e = micro[i]
        bnd = edit[i - 1]
        if bnd not in seam_set:
            continue
        prev_start = micro[i - 1][0]

        # 段首 warm-up 帧 (re-anchor 瞬态) 不算稳定内容：
        # 从变换估计窗口剔除，再单独修平。
        warmup = _head_warmup_frames(images, bnd, window, device)

        left = _sample_window_pixels(
            images, max(prev_start, bnd - window), bnd, device)
        right = _sample_window_pixels(
            images, s + warmup, min(e, s + window + warmup), device)

        if left is None or right is None:
            if debug:
                print(f"[H3-Seam] 帧{bnd}: 窗口有效像素不足，跳过")
            continue

        a, b = estimate(right, left)
        a, b, warn = _guard_transform(a, b, right, left)
        a, b = _blend_toward_identity(a, b, strength)
        _apply_transform(images, s + warmup, e, a, b, device)
        _fix_warmup_frames(images, bnd, warmup, device)
        _despike_seam(images, bnd, device)
        _smooth_seam_ramp(images, bnd, blend_frames, device)
        logs.append((bnd, s, e, a, b, warn))

    return logs


# ============================================================
# 段首 warm-up 帧处理 (阶段 1 用)
# ============================================================

@torch.no_grad()
def _head_warmup_frames(images, bnd, window, device, max_warmup=_WARMUP_MAX):
    """右段头部 warm-up 帧数 (0~max_warmup)。

    判据是「跳入、跳出都大，但两帧净值小」：帧 t 相对 t-1 剧烈跳变，
    t+1 又跳回接近 t-1 的水平 —— 单帧 re-anchor 瞬态签名。曝光台阶则
    相反 (跳入大、跳出小)，不会被误判。

    用于把段首瞬态帧从变换估计窗口剔除，再交给 _fix_warmup_frames 修平。
    """
    n = int(images.shape[0])
    if bnd + 1 >= n:
        return 0

    vals = _luma_series(images, bnd - 1, min(n, bnd + window + max_warmup), device)
    if vals is None or vals.numel() < 3:
        return 0

    warmup = 0
    prev = float(vals[0])  # 帧 bnd-1
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
    """把段首 warm-up 帧修成左右稳定帧之间的时域过渡。

    对每个 warm-up 帧施加逐通道中位数增益，使其亮度/颜色插值到
    「左尾帧 bnd-1」与「右稳定帧 bnd+warmup」之间。只改电平、不重采样
    像素，保留帧自身结构与运动细节，避免光流插值的糊感/鬼影。
    须在 _apply_transform 之后调用 (此时 bnd+warmup 已是对齐后的稳定帧)。
    """
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
    """修平接缝邻域里由硬切变换留下的残差单帧尖峰。

    de-step 的全局变换是硬切在边界处的：若原始接缝是 2~3 帧斜坡，
    斜坡里的"半过渡帧"在变换后会变成单帧尖峰 (比两侧都亮或都暗)，
    而且这帧可能在边界左侧，是 _fix_warmup_frames 覆盖不到的。

    这里对 [bnd-radius, bnd+radius] 逐帧检测：帧 t 的亮度中位数相对
    左右两帧的局部趋势偏离超过 min_step、且左右两帧彼此接近时，用
    单个亮度增益把帧 t 拉回趋势。只改电平、保留结构与运动细节。
    """
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
        # 两侧本身差异大：是台阶/运动，不是单帧尖峰，不碰
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
    """接缝处电平斜坡平滑：把边界 ±blend 帧的亮度过渡拉成平滑斜坡。

    de-step 是硬切 + 单帧修平，残差仍可能是 1~2 帧的快速台阶。这里以
    左右锚点 (bnd-blend-1 与 bnd+blend) 为端点做线性插值，把中间
    [bnd-blend, bnd+blend) 的每帧用部分增益拉向理想斜坡。blend 越大
    过渡越缓；strength<1 只部分拉回，避免过度平滑产生呼吸感。
    只改电平、保留结构与运动细节。
    """
    n = int(images.shape[0])
    if blend <= 0:
        return
    la = bnd - blend - 1
    ra = bnd + blend
    if la < 0 or ra >= n or ra - la < 3:
        return

    prof = _luma_series(images, la, ra + 1, device)  # 帧 [la, ra]
    if prof is None or prof.numel() != ra - la + 1:
        return

    left_lvl = float(prof[0])    # 帧 la
    right_lvl = float(prof[-1])  # 帧 ra
    span = ra - la

    for t in range(la + 1, ra):
        i = t - la
        frac = i / float(span)
        target = left_lvl + (right_lvl - left_lvl) * frac
        y = float(prof[i])
        dev = y - target
        if abs(dev) < _RAMP_MIN_DEV:
            continue
        new_y = y - dev * strength
        scale = new_y / max(y, 1e-4)
        scale = min(max(scale, _GAIN_MIN), _GAIN_MAX)
        if abs(scale - 1.0) < 1e-3:
            continue
        rgb = images[t, ..., :3].to(device)
        images[t, ..., :3] = (rgb * scale).clamp(0.0, 1.0).to(images.device)


# ============================================================
# 接缝分类 (阶段 2 用)
# ============================================================

@torch.no_grad()
def _detect_transient_frames(stats, threshold, min_gap=_CUT_MIN_GAP):
    """单帧瞬态 (flash / 段首 warm-up) 检测。

    判据是「跳入、跳出都大，但两帧净值小」：帧 t 相对 t-1 剧烈跳变，
    t+1 又跳回接近 t-1 的水平 —— 单帧闪光 / 段首 re-anchor 帧的签名。
    它比旧的"帧差局部尖峰"更准：当瞬态帧前后都有大帧差时，相邻两个
    尖峰会互相顶掉，导致漏检。

    返回瞬态帧号列表 (升序)。瞬态帧常是"下一段新增内容的首帧"，
    可作接缝边界的精定位依据。
    """
    n = int(stats.shape[0])
    if n < 3:
        return []

    y = stats[:, 3].float().cpu()  # 逐帧亮度中位数

    out = []
    for t in range(1, n - 1):
        jump_in = abs(float(y[t] - y[t - 1]))
        jump_out = abs(float(y[t + 1] - y[t]))
        net = abs(float(y[t + 1] - y[t - 1]))
        if jump_in > threshold and jump_out > threshold and net < 0.6 * jump_in:
            if not out or t - out[-1] >= min_gap:
                out.append(t)
    return out


def _classify_boundary(images, boundary, threshold, device):
    """边界形态分类：exposure (持续差异) / flash (瞬态) / motion / clean。

    [已移除 scene_cut] 边界来自真实段账目或台阶检测，把它判成"转场"再
    特殊对待没有意义 —— 幅度是否要处理由调用方的阈值闸门决定。
    """
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


# ============================================================
# 切镜检测：全片逐帧镜头边界 (shot detection)
# ============================================================

@torch.no_grad()
def _frame_histograms(images, device):
    """逐帧 RGB 直方图 -> list[N] of [_HIST_BINS*3] (CPU)。分块处理控制显存。"""
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
    """两个直方图向量的 Pearson 相关系数 -> [-1, 1]。

    先减去均值再归一化，因此只比较直方图的"形状"，对整体的
    平移/缩放 (即全局曝光变化) 不敏感 —— 曝光跳变不会被误判成切镜。
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = float(a.norm() * b.norm())
    if denom < 1e-9:
        return 0.0
    return float((a * b).sum() / denom)


@torch.no_grad()
def _detect_shot_cuts(images, threshold, device, min_gap=_CUT_MIN_GAP):
    """全片逐帧镜头边界检测：返回切镜帧号列表 (升序)。

    逐帧算 RGB 直方图，比较相邻帧的直方图相关度；相关度低于 threshold
    且为局部最小处判为切镜。同镜头内的曝光跳变只平移/压缩直方图、
    相关度仍高，因此不会被误判；切镜是内容改变、相关度骤降。
    """
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
        # 局部最小，避免运动导致的连续低相关被反复报告
        left = corr[t - 1] if t - 1 >= 1 else 1.0
        right = corr[t + 1] if t + 1 < n else 1.0
        if corr[t] > left or corr[t] > right:
            continue
        if cuts and t - cuts[-1] < min_gap:
            continue
        cuts.append(t)

    return cuts


def _is_near_cut(b, cuts, tol=_CUT_TOL):
    """判断帧号 b 是否落在切镜附近 (tol 帧内)，用于容错。"""
    return any(abs(int(b) - int(c)) <= tol for c in cuts)


# ============================================================
# GPU optical flow
# ============================================================

def _image_to_gray(image):
    """
    [H,W,C] -> [1,1,H,W]
    """
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
    """
    image: [1,C,H,W]
    flow:  [1,2,H,W] pixel displacement

    output(x) = image(x + flow(x))
    """
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
    """
    Lightweight dense optical flow.

    source: frame A [H,W,C]
    target: frame B [H,W,C]

    Returns flow [1,2,H,W] mapping source -> target.

    This is deliberately conservative: the flow is used to align
    seam frames, not to perform full video optical-flow generation.
    """

    src = _image_to_gray(source)
    tgt = _image_to_gray(target)

    original_h, original_w = src.shape[-2:]

    # Build a small pyramid.
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

        # Image gradients of target.
        ix = _gradient_x(t)
        iy = _gradient_y(t)

        for _ in range(iterations):
            warped = _warp(s, flow)

            error = t - warped

            # Robust normalization prevents large exposure jumps
            # from generating excessive flow.
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

            # Spatial regularization.
            update = 0.65 * update + 0.35 * _smooth_flow(update)

            flow = flow + update

            # Prevent unstable jumps.
            flow = flow.clamp(-32.0, 32.0)

    # At this point flow is at original resolution.
    if flow.shape[-2:] != (original_h, original_w):
        flow = F.interpolate(
            flow,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=True,
        )

    return flow


# ============================================================
# Local color matching
# ============================================================

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


# ============================================================
# 阶段 2：接缝局部 warp + blend
# ============================================================

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
    """
    接缝局部修正策略：

        left frame
           \
            optical flow
             \
              aligned right reference
                   |
             local color match
                   |
             temporal seam blend

    只修改边界附近的小窗口。images 会被原地修改。
    """

    out = images

    if boundary < 1 or boundary >= out.shape[0]:
        return out

    window = max(1, int(window))

    left_idx = boundary - 1
    right_idx = min(
        boundary + window,
        out.shape[0] - 1,
    )

    # 必须 clone：下面的循环会写回 left_idx 附近的帧，
    # 若 device 与 out.device 相同，.to() 返回的是视图，会被写坏。
    left = out[left_idx].to(device).clone()

    right = out[right_idx].to(device).clone()

    h, w = left.shape[:2]

    # Flow is calculated at reduced resolution to save VRAM.
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

    # Limit flow influence.
    flow = flow * float(flow_strength)

    # Convert right reference to NCHW.
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

        # 0 at the outer edge, 1 close to seam.
        dist = abs(
            idx - (boundary - 0.5)
        )

        strength = max(
            0.0,
            1.0 - dist / float(window + 0.5),
        )

        # Temporal position.
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

        # Construct aligned reference.
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

        # Spatial color correction.
        matched = _local_color_match(
            orig,
            reference,
            grid=8,
            strength=blend_strength * strength,
        )

        # Keep original detail.
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


# ============================================================
# 诊断：区分"台阶 / 段内渐变 / 段首瞬态 / 空间不均"
# ============================================================

@torch.no_grad()
def _luma_series(images, start, end, device):
    """[start,end) 每帧的亮度中位数 -> [k]"""
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
    """边界处的带符号亮度台阶 = median(右窗) - median(左窗)。

    带符号很关键：它只反映直流电平差。而 _classify_boundary 的
    delta 是逐像素绝对差的均值，会把运动和空间不均一起算进来。
    """
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
    """[start,end) 帧在 cells x cells 网格上的平均亮度 -> [cells,cells]"""
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
    """打印一组能区分四种成因的诊断量。纯读取，不修改 images。"""
    n = int(images.shape[0])
    print(f"[H3-Seam] ===== 诊断 ({tag}) =====")

    prof = _luma_series(images, 0, n, device)
    if prof is None:
        return

    # 1) 全片亮度曲线 -> 区分"台阶"与"段内渐变漂移"
    stride = max(1, n // 26)
    pts = [f"{i}:{float(prof[i]):.3f}" for i in range(0, n, stride)]
    print(f"[H3-Seam] [诊断] 逐帧亮度(每{stride}帧) {' '.join(pts)}")

    # 2) 各段头/中/尾亮度 -> 段内是否单调漂移
    segs = _segments_from_boundaries(n, boundaries)
    for i, (s, e) in enumerate(segs):
        if e - s < 3:
            continue
        k = max(1, min(5, (e - s) // 3))
        head = float(prof[s:s + k].median())
        mid = float(prof[(s + e) // 2 - k // 2:(s + e) // 2 + k // 2 + 1].median())
        tail = float(prof[e - k:e].median())
        print(
            f"[H3-Seam] [诊断] 段{i} [{s},{e}) 头={head:.4f} "
            f"中={mid:.4f} 尾={tail:.4f} 段内漂移={tail - head:+.4f}"
        )

    # 3) 边界邻域逐帧 -> 是否只是段首 1~2 帧瞬态
    for b in boundaries:
        lo = max(0, b - 4)
        hi = min(n, b + 5)
        seq = " ".join(
            f"{'|' if i == b else ''}{i}:{float(prof[i]):.4f}"
            for i in range(lo, hi)
        )
        print(f"[H3-Seam] [诊断] 帧{b} 邻域 {seq}   ('|'=边界首帧)")

    # 4) 3x3 分块亮度差 -> 是否空间不均 (全局变换原理上修不了)
    for b in boundaries:
        cl = _cell_luma(images, max(0, b - window), b, device)
        cr = _cell_luma(images, b, min(n, b + window), device)
        if cl is None or cr is None:
            continue
        d = (cr - cl).flatten()
        vals = " ".join(f"{float(v):+.4f}" for v in d)
        spread = float(d.max() - d.min())
        mean = float(d.mean())
        # 用相对判据：只有"跨块变化量超过平均偏移本身"才是真正的空间不均。
        # 均匀台阶/瞬态会让 9 个值整体平移 (spread 远小于 |mean|)，不该告警。
        uneven = spread > max(0.015, abs(mean))
        print(
            f"[H3-Seam] [诊断] 帧{b} 3x3分块亮度差 [{vals}] "
            f"均值={mean:+.4f} 极差={spread:.4f}"
            f"{'  <- 空间不均, 全局变换无法完全消除' if uneven else ''}"
        )

    print("[H3-Seam] ===== 诊断结束 =====")


# ============================================================
# ComfyUI Node
# ============================================================

_PHOTO_TOOLTIP = (
    "色彩/曝光处理档位。"
    "off: 不处理; "
    "low: 逐通道亮度增益消除边界台阶，修正量减半，最保守、绝不引入色相偏移; "
    "medium: MKL 线性色彩迁移消除边界台阶 (推荐起点)。只修接缝处的电平跳变，"
    "完整保留画面本身的明暗变化; "
    "high: 同 medium 但统计窗口更长，边界两侧运动大时更稳; "
    "max: 全片逐帧电平归一化 —— 把每帧亮度都拉到第 1 段的水平。"
    "这是唯一能消除'段内渐变漂移'的档位，但代价是会同时削平画面本身合理的"
    "明暗变化 (天黑/灯灭/进隧道)，因为二者在像素里是同一个信号、无法区分。"
    "另外对已被压成近黑的帧无效 (信息已丢失，日志会报告受影响帧数)"
)

_SEAM_TOOLTIP = (
    "接缝处连续性处理档位 —— 在边界前后若干帧做光流对齐 + 局部融合，"
    "修补运动/结构性不连续。档位越高，参与融合的帧数与融合强度越大，"
    "接缝越平滑但越容易带来轻微糊感或 pumping。"
    "off: 不处理 (推荐默认，先只用色彩档位看效果)"
)

_FLASH_TOOLTIP = (
    "单独处理闪帧 (边界处 1~2 帧的瞬时亮度突跳)。"
    "独立成开关的原因：闪帧走的是接缝局部时域融合逻辑，"
    "与整段色彩变换完全不同；而且当画面本身存在合理的快速明暗变化 "
    "(闪电、爆炸、灯光切换) 时，抑制闪帧会把这些效果一起削平。"
    "fix_motion_preset=off 时本项仍然生效，使用 medium 档参数"
)

_CUT_DETECT_TOOLTIP = (
    "镜头检测闸门。开启后，对整个视频逐帧做切镜检测 (相邻帧 RGB 直方图相关度)，"
    "把时间线切成多个镜头；曝光/色彩/运动修正只在同一个镜头内部进行，"
    "真实切镜处 (包括分段内部的切镜) 直接跳过，不把不同镜头强行对齐。"
    "关闭则回到旧行为：处理全部分段边界"
)

_CUT_THRESHOLD_TOOLTIP = (
    "切镜判定阈值 (相邻帧 RGB 直方图相关度，0~1)。相关度低于该值判为切镜。"
    "越高越激进 (更容易把同镜头的快速运动误判成切镜)，"
    "越低越保守 (更容易漏掉切镜)。默认 0.6"
)

_DEBUG_TOOLTIP = (
    "打印诊断：全片亮度曲线、各段头/中/尾漂移、边界邻域逐帧亮度、"
    "3x3 分块亮度差，用于判断成因 (台阶 / 段内渐变 / 段首瞬态 / 空间不均)。"
    "把三个处理项都设为 off 并开启本项，即为「只诊断不修改」"
)

# 色彩/曝光档位 -> 内部参数
_PHOTO_LEVELS = {
    "off": None,
    "low": {"mode": "de-step", "method": "affine", "strength": 0.5, "window": 3},
    "medium": {"mode": "de-step", "method": "mkl", "strength": 1.0, "window": 3},
    "high": {"mode": "de-step", "method": "mkl", "strength": 1.0, "window": 8},
    "max": {"mode": "flatten", "method": "mkl", "strength": 1.0, "window": 3},
}

# 接缝连续性档位 -> 内部参数
_SEAM_LEVELS = {
    "off": None,
    "low": {"window": 1, "flow": 0.40, "blend": 0.35, "iters": 8, "pyramid": 3},
    "medium": {"window": 3, "flow": 0.75, "blend": 0.65, "iters": 12, "pyramid": 3},
    "high": {"window": 6, "flow": 1.00, "blend": 0.80, "iters": 16, "pyramid": 4},
    "max": {"window": 9, "flow": 1.20, "blend": 1.00, "iters": 20, "pyramid": 4},
}

_LEVEL_ORDER = ["off", "low", "medium", "high", "max"]

# 内部固定参数
_LOCAL_THRESHOLD = 0.05
_DETECT_Z = 4.0
# 台阶检测专用窗口：固定小窗，与 color 档位的修正窗口 (stat_window) 解耦。
# high 档修正窗口=8，若也用于检测，8 帧中位数会把台阶 z 分压到阈值以下而漏检。
_DETECT_WINDOW = 3



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

                io.Int.Input(
                    "blend_frames",
                    default=2,
                    min=0,
                    max=8,
                    step=1,
                    tooltip=(
                        "接缝处电平渐变窗口 (帧)。曝光对齐后，把边界前后各 "
                        "blend_frames 帧的亮度过渡按平滑斜坡拉平；值越大过渡越缓、"
                        "越自然，但运动大的镜头过大会带来轻微糊感/呼吸感。0=关闭"
                    ),
                ),

                io.Boolean.Input(
                    "cut_detection",
                    default=True,
                    tooltip=_CUT_DETECT_TOOLTIP,
                ),

                io.Float.Input(
                    "cut_threshold",
                    default=0.6,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=_CUT_THRESHOLD_TOOLTIP,
                ),

                io.Boolean.Input(
                    "use_gpu",
                    default=True,
                    tooltip="使用 CUDA GPU 进行统计、色彩变换与光流",
                ),

                io.Boolean.Input(
                    "debug",
                    default=False,
                    tooltip=_DEBUG_TOOLTIP,
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
        blend_frames=2,
        use_gpu=True,
        debug=False,
        cut_detection=True,
        cut_threshold=0.6,
    ) -> io.NodeOutput:

        if images is None or images.shape[0] < 2:
            return io.NodeOutput(images)

        photo = _PHOTO_LEVELS.get(fix_color_preset)
        seam = _SEAM_LEVELS.get(fix_motion_preset)
        # 闪帧走接缝局部逻辑；fix_motion_preset=off 时用 medium 档参数
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
            f"blend={blend_frames}"
        )

        # ---- 边界确定：纯内容自动检测 (不依赖 info) ----
        # 注意：检测用固定小窗 _DETECT_WINDOW，不能用 stat_window ——
        # stat_window 随 color 档位变大 (high=8)，会让台阶 z 分被稀释而漏检。
        min_gap = max(4, _DETECT_WINDOW)
        stats = _frame_stats(images, device)

        # 1) 单帧瞬态 (flash / 段首 warm-up)：跳入跳出都大、净值小，
        #    精确定位"下一段新增内容的首帧"。
        transients = _detect_transient_frames(stats, _WARMUP_MIN, min_gap)

        # 2) 曝光台阶 (level shift)：右窗跳过 warm-up 帧，避免瞬态抹平台阶。
        steps, z = _detect_step_boundaries(
            stats, _DETECT_WINDOW, _DETECT_Z, min_gap)

        # 瞬态帧附近的台阶属同一接缝，以瞬态帧的精确定位为准
        steps = [
            s for s in steps
            if all(abs(s - f) >= min_gap for f in transients)
        ]

        boundaries = sorted(set(transients) | set(steps))

        print(
            f"[H3-Seam] 边界来源=内容检测: 瞬态={transients} "
            f"台阶={steps} -> 合并={boundaries}"
        )

        if debug and z is not None:
            top = torch.topk(z, min(8, z.numel()))
            pairs = [
                f"帧{int(i)}:z={float(v):.1f}"
                for v, i in zip(top.values.tolist(), top.indices.tolist())
            ]
            print(f"[H3-Seam] [debug] 台阶得分 top: {', '.join(pairs)}")

        if not boundaries:
            print("[H3-Seam] 未检测到接缝，原样输出")
            return io.NodeOutput(images)

        # ---- 镜头检测：全片逐帧切镜，只修同镜头接缝 ----
        if cut_detection:
            cuts = _detect_shot_cuts(
                images, float(cut_threshold), device)

            seams = [
                b for b in boundaries
                if not _is_near_cut(b, cuts)
            ]
            edit_points = sorted(
                {int(b) for b in boundaries} | {int(c) for c in cuts}
            )

            print(f"[H3-Seam] 镜头检测: 切镜帧 {cuts}")
            skipped = [b for b in boundaries if b not in seams]
            if skipped:
                print(
                    f"[H3-Seam] 跳过切镜边界 {skipped}，"
                    f"仅处理同镜头接缝 {seams}"
                )
        else:
            seams = list(boundaries)
            edit_points = sorted({int(b) for b in boundaries})
            print("[H3-Seam] 镜头检测已关闭，处理全部分段边界")

        # ---- 诊断 (修正前) ----
        if debug:
            _diagnose(images, seams, stat_window, device, "修正前")

        if photo is None and seam is None and not fix_flash:
            print(
                "[H3-Seam] 三项处理均为 off -> 画面原样输出"
                + ("" if debug else " (需要诊断请开启 debug)")
            )
            return io.NodeOutput(images)

        out = images.clone()

        # ---- 阶段 1：色彩/曝光对齐 ----
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
                debug,
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
                if debug:
                    off = [float(b[i]) for i in range(3)]
                    print(
                        f"[H3-Seam] [debug] 帧{bnd} offset="
                        f"({off[0]:+.4f},{off[1]:+.4f},{off[2]:+.4f}) "
                        f"中灰亮度 0.500 -> {luma_out:.3f}"
                    )

        # ---- 阶段 2：接缝局部光流 warp/blend ----
        if seam is not None or fix_flash:
            actions = []

            for b in seams:
                kind, delta = _classify_boundary(
                    out, b, _LOCAL_THRESHOLD, device)

                # 幅度闸门：_classify_boundary 只看形态不看幅度，会把微小
                # 噪声也归类。低于阈值视为已经干净，别白做融合 (只会引入糊感)。
                if delta < _LOCAL_THRESHOLD:
                    actions.append(
                        (b, "clean",
                         f"跳过 delta={delta:.3f} < {_LOCAL_THRESHOLD}")
                    )
                    continue

                if kind == "flash":
                    if not fix_flash:
                        actions.append(
                            (b, kind,
                             f"跳过 (fix_flash=off) delta={delta:.3f}"))
                        continue
                    cfg = flash_cfg
                    label = "fix_flash"
                else:
                    if seam is None:
                        actions.append(
                            (b, kind,
                             f"跳过 (fix_motion_preset=off) delta={delta:.3f}"))
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

        # ---- 诊断 (修正后) ----
        if debug:
            _diagnose(out, seams, stat_window, device, "修正后")

        if device.type == "cuda":
            torch.cuda.synchronize()

        return io.NodeOutput(out)

        # ---- 诊断 (修正后) ----
        if debug:
            _diagnose(out, seams, stat_window, device, "修正后")

        if device.type == "cuda":
            torch.cuda.synchronize()

        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {
    "H3SeamCorrection": H3SeamCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SeamCorrection":
        "Minimax_H3_Seam_Correction",
}
