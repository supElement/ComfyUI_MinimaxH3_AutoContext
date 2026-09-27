"""h3_facefix.py — H3 高分辨率局部人脸修复 (逐帧稳定裁剪 v4, SeC-4B 多人支持)

v4 变更:
- 新增内置 SeC-4B 引擎加载 (sec_engine/ 子目录, 随本扩展发布, 无外部节点依赖):
  load_sec_model / unload_sec_model / list_sec_models / sec_track_identities。
- sec_track_identities: YOLO 多框检测 → SeC-4B 逐身份跨帧追踪 → 每个身份一条
  逐帧紧致 bbox 轨迹 (解决多人面部交叉 / 人物离场回归的归因)。
- 原有全部接口保留 (face_split_blocks / token_blocks / 检测 / 羽化 / 编码等)。

v4.1 变更 (日志分级):
- 新增共享日志函数, 供全部 H3 节点复用:
    log  = 主干信息 (每节点入口/出口/关键摘要, 始终输出)
    vlog = 逐子轨/逐块/逐镜头调试 (仅 _VERBOSE=True 时输出)
    warn = 警告 (始终输出, 黄色)
- 需要排查问题时, 把本文件顶部 _VERBOSE 改为 True 即可, 无需改任何节点代码。
"""
import gc
import os
import re
import shutil
import tempfile
import numpy as np
import torch
import folder_paths

try:
    import comfy.utils
    _HAS_COMFY = True
except ImportError:
    _HAS_COMFY = False

_VERBOSE = False


def log(*args, **kwargs):
    """主干信息: 每节点入口/出口摘要、关键统计 — 始终输出。"""
    print(*args, **kwargs)


def vlog(*args, **kwargs):
    """逐子轨 / 逐块 / 逐镜头 / 逐条缓存条目的调试细节 — 仅 _VERBOSE=True。"""
    if _VERBOSE:
        print(*args, **kwargs)


def warn(*args, **kwargs):
    """警告 / 异常但可继续的情形 — 始终输出, 黄色。"""
    print("\033[33m" + " ".join(str(a) for a in args) + "\033[0m", **kwargs)


SPATIAL_COMPRESSION = 16
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
AUDIO_LATENTS_PER_SEC = 40


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
        warn(f"[H3-FaceFix] x0 normalization failed: {e}\n[H3-FaceFix] x0 规范化失败: {e}")
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
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(max(0, int(n_tokens))))


def decode_probe_frames(vae, v_lat, probe):
    pixels = vae.decode(v_lat[:, :, list(probe)])
    if pixels.dim() == 4:
        pixels = pixels.unsqueeze(1)
    frames = (pixels[0].clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
    return frames


def video_latent_frames(pixel_frames):
    return 2 if pixel_frames <= 5 else ((pixel_frames - 5) // 17) * 5 + 2


def face_split_blocks(seg_sizes, effective_context, expect_tokens=None, boundaries=None, decoded_frames=None):
    if not seg_sizes or not isinstance(seg_sizes, (list, tuple)):
        return None
    ctx_v = video_latent_frames(int(effective_context)) if (effective_context or 0) > 5 else 2
    ctx_v = max(2, ctx_v)
    st = []
    for s in seg_sizes:
        s = int(s)
        if s < 5:
            warn("[H3-FaceFix] split: segment < 5 frames -> fallback\n[H3-FaceFix] 分段账目: 存在 <5 帧的段 → 回退")
            return None
        st.append(video_latent_frames(s))
    blocks, cum = [], 0
    for i, st_i in enumerate(st):
        n_i = 0 if i == 0 else min(ctx_v, st_i - 2)
        c0, c1 = cum, cum + (st_i - n_i)
        blocks.append((c0 - (c0 % 5), c1, _pixels_for_tokens(c0), _pixels_for_tokens(c1)))
        cum = c1
    if expect_tokens is not None and cum != int(expect_tokens):
        warn(f"[H3-FaceFix] split: token accounting {cum} != latent tokens {int(expect_tokens)} -> fallback\n"
             f"[H3-FaceFix] 分段账目: token 账目 {cum} ≠ latent 实际 token 数 {int(expect_tokens)} → 回退")
        return None
    if decoded_frames is not None and blocks[-1][3] != int(decoded_frames):
        warn(f"[H3-FaceFix] split: frames {blocks[-1][3]} != expected {int(decoded_frames)} -> fallback\n"
             f"[H3-FaceFix] 分段账目: 帧数账目 {blocks[-1][3]} ≠ 期望 {int(decoded_frames)} → 回退")
        return None
    if boundaries:
        bnd = [int(x) for x in boundaries]
        if len(bnd) != len(blocks) - 1 or [b[2] for b in blocks[1:]] != bnd:
            warn("[H3-FaceFix] split: boundaries mismatch -> fallback\n[H3-FaceFix] 分段账目: 与 info.boundaries 不一致 → 回退")
            return None
    return blocks


def token_blocks(n_tokens, chunk_tokens=22):
    blocks, k0, keep0 = [], 0, 0
    total = _pixels_for_tokens(int(n_tokens))
    while keep0 < total:
        k1 = min(k0 + int(chunk_tokens), int(n_tokens))
        keep1 = _pixels_for_tokens(k1)
        blocks.append((k0, k1, keep0, keep1))
        keep0 = keep1
        k0 = max(k1 - 2, k0 + 1)
    return blocks


# ================= 检测 =================
_yolo_model = None
_yolo_path = None


def detect_faces_yolo(frames_u8, conf=0.3, model_path=""):
    global _yolo_model, _yolo_path
    from ultralytics import YOLO
    if not model_path:
        raise ValueError("face_model is empty: a YOLO face-weights path is required (e.g. face_yolov9c.pt)"
                         "\nface_model 为空: 需要 YOLO 人脸权重路径 (如 face_yolov9c.pt)")
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


# ================= 稳定轨迹 (旧单脸路径兼容保留) =================
def build_stable_track(per_frame, expand_f):
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


# ================= 像素域缩放 + 编码 =================
def upscale_and_encode(vae, frames, out_h, out_w, want_t=None, tag=""):
    up = comfy.utils.common_upscale(frames.movedim(-1, 1).contiguous(), int(out_w), int(out_h),
                                    "lanczos", "disabled").movedim(1, -1)
    return encode_frames_adaptive(vae, up.contiguous(), want_t=want_t, tag=tag)


def encode_frames_adaptive(vae, frames, want_t=None, tag=""):
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
                vlog(f"[H3-Enc]{tag} layout={t_} → {tuple(lat.shape)} ok\n[H3-Enc]{tag} 布局={t_} → {tuple(lat.shape)} ok")
                return lat
            errs.append(f"{t_}→{tuple(lat.shape)}")
        except Exception as e:
            errs.append(f"{t_}→{type(e).__name__}: {e}")
    warn(f"[H3-Enc]{tag} all layouts failed (want_t={want_t}, input {tuple(frames.shape)}): {errs}\n"
         f"[H3-Enc]{tag} 全部布局失败 (want_t={want_t}, 输入{tuple(frames.shape)}): {errs}")
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


# ============ SeC-4B 多人身份追踪 ============

_SEC_MODEL_CACHE = {}  


def list_sec_models():
    """扫描 models/sams 下的 SeC 单文件权重 (排除已废弃的 fp8)。"""
    out, seen = [], set()
    prefer = ["SeC-4B-fp16.safetensors", "SeC-4B-bf16.safetensors", "SeC-4B-fp32.safetensors"]
    try:
        sams_dirs = folder_paths.get_folder_paths("sams")
    except Exception:
        sams_dirs = [os.path.join(folder_paths.models_dir, "sams")]
    for d in sams_dirs:
        if not os.path.isdir(d):
            continue
        for fn in prefer + sorted(os.listdir(d)):
            if fn in seen or "fp8" in fn.lower():
                continue
            p = os.path.join(d, fn)
            if fn.endswith(".safetensors") and "sec" in fn.lower() and os.path.isfile(p):
                out.append(fn)
                seen.add(fn)
    return out


def _resolve_sec_model_path(model_file):
    try:
        sams_dirs = folder_paths.get_folder_paths("sams")
    except Exception:
        sams_dirs = [os.path.join(folder_paths.models_dir, "sams")]
    if model_file:
        for d in sams_dirs:
            p = os.path.join(d, model_file)
            if os.path.isfile(p):
                return p, model_file
        raise FileNotFoundError(f"[H3-FaceFix] SeC model not found: {model_file}\n"
                                f"[H3-FaceFix] 未找到 SeC 模型: {model_file}")
    models = list_sec_models()
    if not models:
        raise RuntimeError("[H3-FaceFix] no SeC-4B weights in ComfyUI/models/sams/ "
                           "(download SeC-4B-fp16.safetensors)\n"
                           "[H3-FaceFix] models/sams 下没有 SeC-4B 权重 (请下载 SeC-4B-fp16.safetensors)")
    for d in sams_dirs:
        p = os.path.join(d, models[0])
        if os.path.isfile(p):
            return p, models[0]
    raise FileNotFoundError("[H3-FaceFix] SeC weights missing")


def load_sec_model(model_file="", device="auto", use_flash_attn=True, allow_mask_overlap=True):
    """加载内置 SeC-4B (fp16/bf16/fp32, fp8 上游已废弃), 进程级缓存。
    加载流程与上游加载器一致: 空权重初始化 → safetensors 逐张量搬入 → eval →
    preparing_for_generation → fp16 数值稳定性 forward_pre_hook。"""
    from . import sec_engine
    SeCConfig, SeCModel = sec_engine.core()
    model_config_dir = sec_engine.config_dir()
    path, fname = _resolve_sec_model_path(model_file)
    precision = ("fp32" if "-fp32" in fname else "bf16" if "-bf16" in fname else "fp16")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("gpu"):
        device = f"cuda:{int(device[3:])}"
    dev = torch.device(device)
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[precision]
    if dev.type == "cpu" and torch_dtype != torch.float32:
        vlog("[H3-FaceFix] CPU requires fp32, converting\n[H3-FaceFix] CPU 模式需 fp32, 已转换")
        torch_dtype = torch.float32
    if torch_dtype == torch.float32 and use_flash_attn:
        vlog("[H3-FaceFix] fp32 incompatible with flash-attn, disabled\n[H3-FaceFix] fp32 与 flash-attn 不兼容, 已关闭")
        use_flash_attn = False
    key = (path, str(dev), bool(use_flash_attn), bool(allow_mask_overlap))
    if key in _SEC_MODEL_CACHE:
        return _SEC_MODEL_CACHE[key]
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    config = SeCConfig.from_pretrained(model_config_dir)
    config.hydra_overrides_extra = [
        f"++model.non_overlap_masks={'false' if allow_mask_overlap else 'true'}"]
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
        with init_empty_weights():
            model = SeCModel(config, use_flash_attn=use_flash_attn)
        state_dict = load_file(path)
        for name, param in state_dict.items():
            set_module_tensor_to_device(model, name, device="cpu", value=param)
        del state_dict
        model = model.eval()
    except ImportError:
        model = SeCModel(config, use_flash_attn=use_flash_attn)
        state_dict = load_file(path)
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model = model.eval()
    model = model.to(device=dev, dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_config_dir, trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)
    if dev.type == "cuda" and torch_dtype != torch.float32:
        import torch.nn as nn
        int_dtypes = (torch.long, torch.int, torch.int32, torch.int64)

        def _dtype_hook(module, args, kwargs):
            pd = None
            for p_ in module.parameters():
                pd = p_.dtype
                break
            if pd is None or isinstance(module, nn.Embedding):
                return args, kwargs
            na = tuple(a if (not isinstance(a, torch.Tensor) or a.dtype in int_dtypes or a.dtype == pd)
                       else a.to(pd) for a in args)
            nk = {k: (v if (not isinstance(v, torch.Tensor) or v.dtype in int_dtypes or v.dtype == pd)
                      else v.to(pd)) for k, v in kwargs.items()}
            return na, nk

        for m in model.modules():
            if len(list(m.parameters(recurse=False))) > 0:
                m.register_forward_pre_hook(_dtype_hook, with_kwargs=True)
    _SEC_MODEL_CACHE[key] = model
    log(f"[H3-FaceFix] SeC-4B loaded (built-in engine): {fname} [{precision}] on {dev}\n"
        f"[H3-FaceFix] SeC-4B 已加载 (内置引擎): {fname} [{precision}] @ {dev}")
    return model


def unload_sec_model():
    """释放 SeC-4B 显存 (追踪完成后调用, 给 H3/VAE 让位)。"""
    for k, m in list(_SEC_MODEL_CACHE.items()):
        try:
            m.to("cpu")
        except Exception:
            pass
        del m
    _SEC_MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("[H3-FaceFix] SeC-4B unloaded, VRAM freed\n[H3-FaceFix] SeC-4B 已卸载, 显存已释放")


def _bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _overlap_ratio(a, b):
    """重叠率 = 交集面积 / 较小框面积。修正 IoU 对包含关系的失真:
    紧致脸框完全落在头框内时 IoU=(w1/w2)² 可能 <0.3 被误判。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    amin = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / amin if amin > 0 else 0.0


def _mask2d(mask_or_logits):
    """兼容 torch/numpy 的 [H,W] / [1,H,W] / [1,1,H,W] 形状, 阈值>0 → 2D uint8。"""
    m = mask_or_logits
    if hasattr(m, "detach"):
        m = m.detach().float().cpu().numpy()
    m = np.asarray(m)
    while m.ndim > 2:
        m = m[0]
    return (m > 0.0).astype(np.uint8)


def _mask_extract(out):
    """从 SeC/SAM2 返回的第三元素中提取 mask 张量 (tensor / tuple / list / dict 兼容)。"""
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for k in ("video_res_masks", "pred_masks", "masks", "low_res_masks"):
            v = out.get(k)
            if torch.is_tensor(v):
                return v
        return None
    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v):
                return v


def _mask_to_box(mask):
    """任意掩码形状 → 2D → 紧致 bbox; 空 mask 返回 None。"""
    m = _mask2d(mask)
    if m.ndim != 2:
        m = m.reshape(-1, m.shape[-2], m.shape[-1])[0]
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def sec_track_identities(model, frames_u8, per_frame, tracking_direction="bidirectional",
                         mllm_memory_size=12, offload_video_to_cpu=True,
                         iou_thr=0.3, max_ids=6, attempt_gap=16, max_runs=None, tag=""):
    """YOLO 多框检测结果 → SeC-4B 逐身份跨帧追踪。
    返回: [{"id", "anchor", "boxes": [F] bbox|None, "masks": [F] uint8[0/1] [H,W]|None}, ...]
    masks 供 H3FaceCut 裁窗口 mask、H3FaceBlend 做 mask 贴回 (box 仅用于几何账目)。
    注意 masks 为全分辨率, 内存 ≈ F × H × W 字节 (1080p/192帧 ≈ 400MB, 瞬时, 用完即释放)。
    修复记录:
    - max_frame_num_to_track 必须 None (SeC 不处理负数, -1 → 空区间 → 0 帧)
    - init_mask 传锚点帧 mask (来自 add_new_points_or_box 返回值, 语义记忆需要)
    - mask 经 _mask2d 压 2D (修复 np.nonzero 对 [1,H,W] 的 unpack 错误)
    - box 以 numpy float32 传入; 锚点 bbox 直接用 YOLO 原框 (全分辨率, 不反推)
    - 种子全败即放弃 / 锚点最小间隔 attempt_gap / 总尝试预算 max_runs
    """
    from PIL import Image
    F = int(frames_u8.shape[0])
    det_counts = [len(b) for b in per_frame]
    if not any(det_counts):
        return []
    if max_runs is None:
        max_runs = max(4, int(max_ids) * 2)
    runs_left = [int(max_runs)]
    state = None
    tmpdir = tempfile.mkdtemp(prefix="h3_sec_")
    try:
        for i in range(F):
            Image.fromarray(frames_u8[i]).save(os.path.join(tmpdir, f"{i:07d}.jpg"))
        state = model.grounding_encoder.init_state(
            video_path=tmpdir,
            offload_video_to_cpu=bool(offload_video_to_cpu),
            offload_state_to_cpu=False)

        def _track_one(anchor, box, obj_id, occupied=None):
            if runs_left[0] <= 0:
                warn(f"[H3-FaceFix]{tag} SeC run budget exhausted, skip anchor@{anchor}\n"
                     f"[H3-FaceFix]{tag} SeC 尝试次数已达上限, 跳过锚点@{anchor}")
                return None
            runs_left[0] -= 1
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            boxes = [None] * F
            masks = [None] * F
            struct_logged = [False]

            def _prompt_at(frame_idx, bx):
                """reset + box prompt → init_mask。
                ⚠️ 必须返回 CPU numpy: SeC 在场景切换时做概念记忆回放
                (label_img_with_mask 内部 np.uint8(mask)), CUDA 张量必崩;
                引擎自己追加的记忆项也是 (video_res_masks[0]>0).cpu().numpy(), 同款约定。"""
                model.grounding_encoder.reset_state(state)
                bx1, by1, bx2, by2 = [float(v) for v in bx[:4]]
                r = model.grounding_encoder.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(frame_idx),
                    obj_id=int(obj_id),
                    points=None,
                    labels=None,
                    box=np.asarray([bx1, by1, bx2, by2], dtype=np.float32))
                raw = _mask_extract(r[2] if isinstance(r, (tuple, list)) else r)
                if raw is None or not torch.is_tensor(raw):
                    raise RuntimeError(f"prompt output has no mask tensor (got {type(r).__name__})\n"
                                       f"打点输出中未找到 mask 张量")
                init_np = _mask2d(raw)
                if init_np.max() == 0:
                    raise RuntimeError("empty anchor mask from box prompt\n锚点 box 产出空 mask")
                boxes[int(frame_idx)] = [int(bx1), int(by1), int(bx2), int(by2)]
                masks[int(frame_idx)] = init_np
                return init_np.copy()  

            def _run(reverse, init_mask, start_idx):
                for f_idx, _obj_ids, out in model.propagate_in_video(
                        state,
                        start_frame_idx=int(start_idx),
                        max_frame_num_to_track=None,  
                        reverse=reverse,
                        init_mask=init_mask,
                        mllm_memory_size=int(mllm_memory_size)):
                    if not struct_logged[0]:
                        struct_logged[0] = True
                        shp = tuple(out.shape) if torch.is_tensor(out) else "-"
                    if f_idx is None or int(f_idx) < 0 or int(f_idx) >= F:
                        continue
                    if boxes[int(f_idx)] is None:
                        raw = _mask_extract(out)
                        if raw is None:
                            continue
                        m = _mask2d(raw)
                        if m.max() <= 0:
                            continue
                        nb = _mask_to_box(m)
                        if nb is None:
                            continue
                        if occupied is not None and any(
                                _overlap_ratio(nb, cb) >= float(iou_thr) for cb in occupied[int(f_idx)]):
                            continue  
                        boxes[int(f_idx)] = nb
                        masks[int(f_idx)] = m

            def _safe(direction_name, reverse, start_idx, init_mask):
                """单向传播 + 断点续传 (最多 3 次): 中断后用最后一个有效帧的紧致框
                重新打点、从下一帧接着跑 — 轨迹不再被一次异常截断。"""
                cur_init, cur_start = init_mask, int(start_idx)
                for attempt in range(3):
                    try:
                        _run(reverse, cur_init, cur_start)
                        return
                    except Exception as e:
                        import traceback
                        warn(f"[H3-FaceFix]{tag} SeC identity#{obj_id} anchor@{anchor} "
                             f"{direction_name} attempt{attempt + 1}: propagate error: {e}\n"
                             f"[H3-FaceFix]{tag} SeC 身份#{obj_id} 锚点@{anchor} "
                             f"{direction_name} 第{attempt + 1}次传播异常: {e}")
                        vlog(traceback.format_exc())
                        good = [f for f, b in enumerate(boxes) if b is not None]
                        if reverse:
                            seg = [f for f in good if f < cur_start] or good
                            anchor_f = min(seg) if seg else None
                            nxt = (anchor_f - 1) if anchor_f is not None else None
                        else:
                            seg = [f for f in good if f > cur_start] or good
                            anchor_f = max(seg) if seg else None
                            nxt = (anchor_f + 1) if anchor_f is not None else None
                        if anchor_f is None or nxt is None or nxt < 0 or nxt >= F or attempt == 2:
                            return
                        try:
                            cur_start = nxt
                            cur_init = _prompt_at(anchor_f, boxes[anchor_f])
                        except Exception as e2:
                            warn(f"[H3-FaceFix]{tag} resume prompt failed: {e2}\n"
                                 f"[H3-FaceFix]{tag} 续传打点失败: {e2}")
                            return

            try:
                init = _prompt_at(anchor, box)
            except Exception as e:
                warn(f"[H3-FaceFix]{tag} SeC identity#{obj_id} anchor@{anchor}: prompt failed: {e}\n"
                     f"[H3-FaceFix]{tag} SeC 身份#{obj_id} 锚点@{anchor}: 打点失败: {e}")
                return None
            if tracking_direction == "forward":
                _safe("forward", False, anchor, init)
            elif tracking_direction == "backward":
                _safe("backward", True, anchor, init)
            else:  # bidirectional
                _safe("forward", False, anchor, init)
                _safe("backward", True, anchor, _prompt_at(anchor, box))
            n_hit = sum(1 for b in boxes if b is not None)
            n_msk = sum(1 for m in masks if m is not None)
            vlog(f"[H3-FaceFix]{tag} SeC identity#{obj_id} anchor@{anchor}: {n_hit}/{F} frames, masks {n_msk}/{F}\n"
                 f"[H3-FaceFix]{tag} SeC 身份#{obj_id} 锚点@{anchor}: {n_hit}/{F} 帧, mask {n_msk}/{F}")
            return (boxes, masks) if n_hit >= 2 else None

        tracks = []

        def _yolo_dedup(fx):
            """帧 fx 的 YOLO 检出互相去重 (同脸重复检出只算一个)。"""
            out, used = [], []
            for d in per_frame[fx]:
                if any(_bbox_iou(d[:4], u) > 0.5 for u in used):
                    continue
                used.append(d[:4])
                out.append(d)
            return out

        def _distinct_sec(fx):
            """帧 fx 上 SeC 传播框的独立人数: 互相重叠率 >= iou_thr 的框视为同一人。
            去重是必须的 — 否则重复/垃圾身份虚增人数, 把真人永久堵在门外。"""
            uniq = []
            for t in tracks:
                b = t["boxes"][fx]
                if b is None:
                    continue
                if not any(_overlap_ratio(b, u) >= float(iou_thr) for u in uniq):
                    uniq.append(b)
            return uniq

        def _frame_covered(fx):
            dets = _yolo_dedup(fx)
            return (not dets) or len(dets) <= len(_distinct_sec(fx))

        def _coverage_of(fx, det):
            """检出框被现有 SeC 框覆盖的最大重叠率 (越小 → 越可能是新人)。"""
            c = 0.0
            for t in tracks:
                b = t["boxes"][fx]
                if b is not None:
                    c = max(c, _overlap_ratio(det[:4], b))
            return c

        def _occ_snapshot():
            return [[t["boxes"][i] for t in tracks if t["boxes"][i] is not None] for i in range(F)]

        anchor0 = max(range(F), key=lambda i: det_counts[i])
        seeds, used = [], []
        for b in per_frame[anchor0]:
            if any(_bbox_iou(b[:4], u) > 0.5 for u in used):
                continue
            used.append(b[:4])
            seeds.append((anchor0, b[:4]))
            if len(seeds) >= int(max_ids):
                break

        for anchor, box in seeds:
            if len(tracks) >= int(max_ids) or runs_left[0] <= 0:
                break
            res = _track_one(anchor, box, obj_id=len(tracks) + 1, occupied=_occ_snapshot())
            if res is None:
                continue
            tracks.append({"id": len(tracks), "anchor": int(anchor),
                           "boxes": res[0], "masks": res[1]})

        BRIDGE = 3
        for _pass in range(3):
            created = 0
            f = 0
            while f < F:
                if len(tracks) >= int(max_ids) or runs_left[0] <= 0:
                    break
                if _frame_covered(f):
                    f += 1
                    continue
                a = f
                miss = 0
                f += 1
                while f < F and miss <= BRIDGE:
                    miss = miss + 1 if _frame_covered(f) else 0
                    f += 1
                f -= miss
                gap = (a, f)
                if gap[1] - gap[0] < 3:
                    continue
                best = max(range(gap[0], gap[1]),
                           key=lambda i: len(_yolo_dedup(i)) - len(_distinct_sec(i)))
                quota = max(0, len(_yolo_dedup(best)) - len(_distinct_sec(best)))
                if quota <= 0:
                    continue
                picked, pboxes = [], []
                for d in sorted(_yolo_dedup(best), key=lambda d: _coverage_of(best, d)):
                    if len(picked) >= quota:
                        break
                    if any(_bbox_iou(d[:4], pb) > 0.5 for pb in pboxes):
                        continue
                    picked.append(d)
                    pboxes.append(d[:4])
                seeded = 0
                for d in picked:
                    if len(tracks) >= int(max_ids) or runs_left[0] <= 0:
                        break
                    res = _track_one(best, d[:4], obj_id=len(tracks) + 1, occupied=_occ_snapshot())
                    if res is None:
                        continue
                    tracks.append({"id": len(tracks), "anchor": int(best),
                                   "boxes": res[0], "masks": res[1]})
                    seeded += 1
                created += seeded
                vlog(f"[H3-FaceFix]{tag} uncovered gap [{gap[0]},{gap[1]}) -> "
                     f"{seeded} new identity(ies) anchored@{best} "
                     f"(yolo {len(_yolo_dedup(best))} vs sec {len(_distinct_sec(best))})\n"
                     f"[H3-FaceFix]{tag} 覆盖缺口 [{gap[0]},{gap[1]}) → {seeded} 个新身份 "
                     f"打点@{best} (检出 {len(_yolo_dedup(best))} vs SeC {len(_distinct_sec(best))})")
            if created == 0 or len(tracks) >= int(max_ids) or runs_left[0] <= 0:
                break

        if tracks:
            n_cov = sum(1 for i in range(F) if any(t["boxes"][i] is not None for t in tracks))
            n_det_notrack = sum(1 for i in range(F)
                                if per_frame[i] and not any(t["boxes"][i] is not None for t in tracks))
            log(f"[H3-FaceFix]{tag} coverage: {n_cov}/{F} frames covered "
                f"(YOLO dets on {sum(det_counts)} frames, "
                f"{n_det_notrack} det-frames without track)\n"
                f"[H3-FaceFix]{tag} 覆盖统计: SeC 覆盖 {n_cov}/{F} 帧 "
                f"(YOLO 检出 {sum(det_counts)} 帧, 其中 {n_det_notrack} 检出帧无轨迹)")
        else:
            log(f"[H3-FaceFix]{tag} SeC produced no identities\n"
                f"[H3-FaceFix]{tag} SeC 未产出任何身份")
        return tracks
    finally:
        if state is not None:
            try:
                model.grounding_encoder.reset_state(state)
            except Exception:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)
