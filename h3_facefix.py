"""h3_facefix.py — H3 高分辨率局部人脸修复 (逐帧稳定裁剪 v3)

v3 变更 (废除聚簇架构):
- 删 _cluster_tokens / _token_bboxes / _interp / _slice_audio_time / _build_upscaled_canvas
  —— 簇/窗口抽象整体退役。历史 bug (22↔73 崩溃、并集框歪斜、窗口snap、6×平均糊脸) 全部
  源于该抽象, 新架构不再存在这些问题。
- 新增 build_stable_track: 缺失填补 + 滑动中值平滑 + 固定裁剪边长 (Wan-Animate 系
  stabilized face crop 范式: 脸在裁剪序列中恒定居中、恒定等大)。
- 新增 upscale_and_encode: 尺度缩放一律 像素域lanczos + VAE编码器 (零 latent 插值)。
- 保留: token↔帧网格换算 (probe_groups/_pixels_for_tokens)、检测、羽化权重、
  encode_frames_adaptive、normalize_x0 (兼容外部引用)。
"""

import json
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

try:
    import comfy.utils
    _HAS_COMFY = True
except ImportError:
    _HAS_COMFY = False

SPATIAL_COMPRESSION = 16
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
AUDIO_LATENTS_PER_SEC = 40


# ================= 参数指纹 =================
# def face_fix_hash(face_fix):
    # if not face_fix or not face_fix.get("enable"):
        # return "off"
    # payload = {k: face_fix.get(k) for k in ("res", "steps", "model", "conf", "expand")}
    # raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    # return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ================= x0 规范化 (兼容保留) =================
def normalize_x0(model, x0, samples=None):
    if x0 is None:
        return None
    try:
        if getattr(x0, "is_nested", False):
            return x0
        x0 = x0.detach().clone()
        if samples is not None and getattr(samples, "is_nested", False):
            latent_shapes = [t.shape for t in samples.unbind()]
            try:
                import comfy.nested_tensor
                import comfy.utils
                x0 = comfy.nested_tensor.NestedTensor(
                    comfy.utils.unpack_latents(x0, latent_shapes))
            except Exception:
                return None
        try:
            x0 = model.model.process_latent_out(x0.cpu())
        except Exception:
            pass
        return x0
    except Exception as e:
        print(f"[H3-FaceFix] x0 normalization failed: {e}\n[H3-FaceFix] x0 规范化失败: {e}")
        return None


# ================= token ↔ 帧网格 =================
def probe_groups(n_tokens):
    sizes = [FRAME_PER_TOKEN[j % 5] for j in range(n_tokens)]
    starts, c = [], 0
    for s in sizes:
        starts.append(c)
        c += s
    return starts, sizes


def _pixels_for_tokens(n_tokens):
    """前 n_tokens 个 latent token 解码产出的像素帧数 (17k+5 型)。"""
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(max(0, int(n_tokens))))


def decode_probe_frames(vae, v_lat, probe):
    pixels = vae.decode(v_lat[:, :, list(probe)])
    if pixels.dim() == 4:
        pixels = pixels.unsqueeze(1)
    frames = (pixels[0].clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    return frames


# ================= 检测 =================
_yolo_model = None
_yolo_path = None


def detect_faces_yolo(frames_u8, conf=0.3, model_path=""):
    global _yolo_model, _yolo_path
    from ultralytics import YOLO
    if not model_path:
        raise ValueError("face_model 为空: 需要 YOLO 人脸权重路径 (如 face_yolov9c.pt)")
    if _yolo_model is None or _yolo_path != model_path:
        _yolo_model = YOLO(model_path)
        _yolo_path = model_path
    bgr_frames = [np.ascontiguousarray(f[..., ::-1]) for f in frames_u8]
    results = _yolo_model.predict(source=bgr_frames, conf=conf, verbose=False)
    out = []
    for r in results:
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(xyxy, confs):
                boxes.append((float(x1), float(y1), float(x2), float(y2), float(s)))
        out.append(boxes)
    return out


def detect_faces(frames_u8, opt):
    conf = float(opt.get("conf", 0.3))
    model_path = (opt.get("model") or "").strip()
    return detect_faces_yolo(frames_u8, conf, model_path)


# ================= 稳定轨迹 (核心新增) =================
def build_stable_track(per_frame, expand_f):
    """逐帧检测框 → 单条稳定轨迹 (初版单脸: 每帧取最高分脸)。
    缺失帧: 端点最近邻 / 中间线性插值 → 滑动中值平滑(窗5, 对离群误检免疫)
    → 固定边长 S = 中位脸尺寸 × (1+expand), 16 对齐。
    返回 (boxes[F][4] float, S int) 或 None (全帧无检出)。"""
    T = len(per_frame)
    raw = [max(fr, key=lambda x: x[4])[:4] if fr else None for fr in per_frame]
    idx = [i for i, b in enumerate(raw) if b is not None]
    if not idx:
        return None
    filled = []
    for i in range(T):
        if raw[i] is not None:
            filled.append(list(raw[i]))
            continue
        prev = next((j for j in range(i - 1, -1, -1) if raw[j] is not None), None)
        nxt = next((j for j in range(i + 1, T) if raw[j] is not None), None)
        if prev is None:
            filled.append(list(raw[nxt]))
        elif nxt is None:
            filled.append(list(raw[prev]))
        else:
            r = (i - prev) / float(nxt - prev)
            filled.append([a + (b - a) * r for a, b in zip(raw[prev], raw[nxt])])
    sm = []
    for i in range(T):
        win = filled[max(0, i - 2): i + 3]
        sm.append([float(np.median([w[k] for w in win])) for k in range(4)])
    ws = sorted(b[2] - b[0] for b in sm)
    hs = sorted(b[3] - b[1] for b in sm)
    med = max(ws[len(ws) // 2], hs[len(hs) // 2])
    S = int(round(med * (1.0 + expand_f)))
    S = max(64, (min(S, 1024) // 16) * 16)
    return sm, S


# ================= 像素域缩放 + 编码 (唯一合法的尺度变换) =================
def upscale_and_encode(vae, frames, out_h, out_w, want_t=None, tag=""):
    """[F,H,W,C] float 0-1 → lanczos (out_w,out_h) → latent [1,C,T,h,w]。
    尺度缩放只由 像素lanczos+VAE编码 完成, 全管线零 latent 插值。"""
    up = comfy.utils.common_upscale(frames.movedim(-1, 1).contiguous(),
                                    int(out_w), int(out_h),
                                    "lanczos", "disabled").movedim(1, -1)
    return encode_frames_adaptive(vae, up.contiguous(), want_t=want_t, tag=tag)


def encode_frames_adaptive(vae, frames, want_t=None, tag=""):
    """[F,H,W,C] 像素 → [1,C,T,h,w] latent。布局自适应; want_t 不匹配视为失败。"""
    cands = [
        ("BHWC5", frames.unsqueeze(0)),
        ("BCTHW", frames.unsqueeze(0).permute(0, 4, 1, 2, 3)),
        ("BHWC4", frames),
    ]
    errs = []
    for t_, f in cands:
        try:
            lat = vae.encode(f)
            if lat.dim() == 4:
                lat = lat.unsqueeze(2)
            if lat.dim() == 5 and (want_t is None or int(lat.shape[2]) == want_t):
                print(f"[H3-Enc]{tag} layout={t_} → {tuple(lat.shape)} ok\n[H3-Enc]{tag} 布局={t_} → {tuple(lat.shape)} ok")
                return lat
            errs.append(f"{t_}→{tuple(lat.shape)}")
        except Exception as e:
            errs.append(f"{t_}→{type(e).__name__}: {e}")
    print(f"[H3-Enc]{tag} all layouts failed (want_t={want_t}, input {tuple(frames.shape)}): {errs}\n[H3-Enc]{tag} 全部布局失败 (want_t={want_t}, 输入{tuple(frames.shape)}): {errs}")
    return None


# ================= 羽化权重 =================
def _edge_weight(n, f, device, dtype):
    w = torch.ones(n, device=device, dtype=torch.float32)
    if f > 0 and n > 2 * f:
        r = torch.linspace(1.0 / (f + 1), 1.0, f, device=device)
        w[:f] = r
        w[-f:] = r.flip(0)
    return w.to(dtype)


def _rect_weight(h, w, f, device, dtype):
    return _edge_weight(h, f, device, dtype)[:, None] \
        * _edge_weight(w, f, device, dtype)[None, :]
