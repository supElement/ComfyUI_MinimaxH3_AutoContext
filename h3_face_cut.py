"""h3_face_cut.py — 修脸第 1 步: 分镜优先的检测与稳定裁剪 (v20.4, 逐帧平滑窗口)

v20.4 变更 (GPU 批量化 + pre_blur, 检测/几何语义不变):
- 采样/缩放 GPU 批量化: _sample_windows 逐帧 CPU grid_sample → 整批上卡
  (uint8 上传 + 批量 grid_sample, OOM 自动减半重试); 无 SR 模型路径的
  common_upscale(lanczos, CPU) → interpolate bicubic+antialias(GPU)。
  → win_v 2→3: 缩放算法变化使旧缓存值与新产物不一致, 升级后旧缓存失效一次。
- 输出 crop_images 统一 fp16 量化 (.to(fp16).float()): 新鲜输出与缓存命中输出
  位精确一致 (缓存本体即 fp16), 下游 rows_hash 不再漂移。
- 新参数 pre_blur (默认 0=关): 裁剪窗在 SR/bicubic 放大前可选高斯预模糊 —
  压制源噪声/插值锯齿, 提高小脸 SR 稳定性。已入缓存键。
- _track_work 精确进度条总量 (原 F_expect 高估, 进度条到不了头)。

v20 变更 (同一身份裁剪平滑化, 远景小脸占比不受影响):
- 窗口边长 S 从"每个 DP 分段一个常数"改为"逐帧一条平滑序列":
    S_t = 该帧脸尺寸 × (1+余量) → 中值滤波 (窗5) → 相邻变化率限速 (_S_RATE)
    → 包含性地板 (快速推镜头时限速让位, 脸必须完整在窗内) → 钳制 [_MIN_WIN, min(W,H)]。
  小脸仍得到小 S → res² 画布上占比恒为 ≈1/(1+余量), 与脸的绝对大小无关 —
  远景小脸不会被稀释 (修脸强度不因远景而打折)。
- 窗口中心按整条出现区间统一中值平滑; DP 分段只划定 skip 边界, 不再产生几何接缝 —
  相邻分段的 (S_t, center_t) 序列天然连续, 贴回后同一身份无直切感。
- 尺寸超比罚 8→2: 逐帧 S 已让段内小脸保持占比, 分段只剩 skip 划分与极端变焦防护。
- 日志收编: 常规运行只保留入口/出口/警告; 逐子轨细节走 h3ff.vlog
  (h3_facefix.py 顶部 _VERBOSE=True 打开)。
- 缓存指纹加 win_v=2: 升级后旧缓存自动失效 (无需手动 clear_cache)。

v19.3 — 分镜检测仅保留官方临时文件路径; 窗口下限提为 _MIN_WIN。
v19.2 — 删除窗口撑大残留, 窗口只由真实检测框决定。
v19 — 全帧保留 (17n+5 网格约束移出本节点, 由 Face_Resample 编码期补齐)。
v18 — 分镜优先 (shot-aware), 官方 PySceneDetect 管线; v17.2 — latent 端口 optional;
v12 — 尺寸信号 YOLO 实测优先; v11 — multi_sec 内置 SeC-4B 身份追踪 (pack v7)。
"""
import os
import shutil
import tempfile
import numpy as np
import torch
import folder_paths
import comfy.utils
import hashlib
import comfy.model_management


try:
    from . import latent_cache
except ImportError:
    import latent_cache
from comfy_api.latest import io
try:
    from . import h3_conditioning
    from . import h3_facefix as h3ff
except ImportError:
    import h3_conditioning
    import h3_facefix as h3ff

MODEL_DIR = os.path.join(folder_paths.models_dir, "elementEasy")
_MODEL_EXTS = {".pt", ".pth", ".onnx", ".engine", ".torchscript"}

# ---- 可调常量: 窗口下限 (px) ----
# 裁剪窗口 S 的最小值 (自动 16 对齐)。远景小脸时 S 会被钳在此地板上 —
# 调小 → 远景占比更高 (脸在画布上更大)
_MIN_WIN = 48

# ---- 可调常量: 窗口边长时序平滑 ----
# 相邻两帧 S 的最大变化率 (0.20 = 每帧最多 ±20%)。调小 → 更平滑但快速推镜头时
_S_RATE = 0.20

# ---- 可调常量: SR 放大的尺寸桶宽 ----
_SR_BUCKET = 32

# ---- 可调常量: DP 单段帧数上限  ----
# 只用于把 DP 复杂度从 O(n²) 压到 O(n·上限); 切段无几何接缝 (逐帧窗口跨段连续)。
_SEG_MAX = 240


def _win_floor():
    """窗口下限 (16 对齐后的 _MIN_WIN) — 唯一来源, 供窗口规划与 DP 估计共用。"""
    return ((int(_MIN_WIN) + 15) // 16) * 16


# ================= 内置常量=================
_GAP_TOL = 24        # 检测缺失多少帧内视为同一次出现 (绝不跨镜头)
_SCALE_SPLIT = 1.2   # 片内容许的最大脸尺寸比 (软目标, 进入 DP 分段代价)
_SKIP_RATIO = 0.8    # 脸 >= res × 此比例 → 跳过重采样
_SEC_MEM_SIZE = 12   # SeC-4B 记忆库槽位数 (上游默认; 逐镜头单元独立调用, 12 足够)


def _has_scenedetect():
    """scenedetect 是否可用 (shot_detect 自动降级依据)。"""
    try:
        import scenedetect  # noqa: F401
        return True
    except Exception:
        return False


def _flash_attn_available():
    """flash-attn 自动检测: 包已安装 + CUDA 可用 + 设备算力 >= sm_80 (Ampere+)。
    fp32 权重与 flash-attn 不兼容的情形由 load_sec_model 内部自动关闭, 此处不处理 dtype。"""
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(0)
        return int(major) >= 8
    except Exception:
        return False


def _list_models():
    out = []
    if os.path.isdir(MODEL_DIR):
        for root, _dirs, files in os.walk(MODEL_DIR):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in _MODEL_EXTS:
                    rel = os.path.relpath(os.path.join(root, fn), MODEL_DIR)
                    out.append(rel.replace("\\", "/"))
    return sorted(out)


def _size(bx):
    return max(bx[2] - bx[0], bx[3] - bx[1])


def _median_smooth(seqs):
    """逐坐标中值平滑 (窗口5, 端点收缩)。全流水线唯一平滑实现 (测量去噪用)。"""
    sm = []
    for i in range(len(seqs)):
        win = seqs[max(0, i - 2): i + 3]
        sm.append([float(np.median([w[k] for w in win])) for k in range(len(seqs[i]))])
    return sm


def _yolo_size_track(sec_boxes, per_frame, iou_thr=0.35):
    """逐帧用 YOLO 实测尺寸覆盖 SeC 紧致框尺寸 (测量交叉校验)。
    v20.2: SeC 漏检帧的"单检出兜底"必须先过时间连续性校验 (与本身份最近的有效框
    比重叠率), 否则多人场景会把别人的脸错归因到本轨迹。"""
    F = len(sec_boxes)
    out = [None] * F

    def _neighbor_ref(i):
        for d in range(1, F):
            for j in (i - d, i + d):
                if 0 <= j < F and sec_boxes[j] is not None:
                    return sec_boxes[j]
        return None

    for i in range(F):
        dets = per_frame[i]
        if not dets:
            continue
        sb = sec_boxes[i]
        if sb is None:
            if len(dets) == 1:
                ref = _neighbor_ref(i)
                d = dets[0]
                if ref is None or h3ff._overlap_ratio(d[:4], ref[:4]) >= float(iou_thr):
                    out[i] = max(d[2] - d[0], d[3] - d[1])
            continue
        best, best_r = None, float(iou_thr)
        for d in dets:
            r = h3ff._overlap_ratio(sb[:4], d[:4])
            if r > best_r:
                best_r, best = r, d
        if best is not None:
            out[i] = max(best[2] - best[0], best[3] - best[1])
    
    
    _valid = [i for i, b in enumerate(sec_boxes) if b is not None]
    if _valid:
        _lo, _hi = _valid[0], _valid[-1]
        out = [v if _lo <= i <= _hi else None for i, v in enumerate(out)]
    
    return out



def _images_digest(img):
    """外部 images 的轻量内容摘要 (抽样 ≤16 帧 × 空间 1/64, fp16 md5) — 缓存键。"""
    t = img.detach()
    step = max(1, int(t.shape[0]) // 16)
    arr = t[::step, ::8, ::8].contiguous().cpu().numpy().astype(np.float16)
    return hashlib.md5(arr.tobytes()).hexdigest()


def _fps_eff(a_lat, F_expect, seg):
    """有效帧率: 有 a_lat 按音频账本换算 (latent 模式); images 模式 (无 a_lat)
    取 info.h3_runtime.fps 或默认 24。"""
    if a_lat is not None:
        return max(1.0, F_expect * h3ff.AUDIO_LATENTS_PER_SEC / max(int(a_lat.shape[-1]), 1))
    rt = seg.get("h3_runtime") if isinstance(seg.get("h3_runtime"), dict) else {}
    return max(1.0, float(rt.get("fps") or seg.get("fps") or 24.0))

# ---- 裁剪窗口放大: 可选放大模型 (upscale_models), 默认 bicubic ----
def _list_upscale_models():
    try:
        return folder_paths.get_filename_list("upscale_models") or []
    except Exception:
        return []


def _load_sr_model(rel):
    """加载放大模型: 直接走 spandrel (ComfyUI 官方 UpscaleModelLoader 的实际加载器,
    requirements 自带依赖)。仅接受图像超分模型, 视频/插帧类模型报错回退 bicubic。"""
    try:
        import spandrel
        from spandrel import ImageModelDescriptor
    except ImportError:
        raise RuntimeError("spandrel not installed — ComfyUI 2024.08+ 自带, 请升级 ComfyUI")
    try:   
        import spandrel_extra_arches
        spandrel_extra_arches.install()
    except Exception:
        pass
    path = folder_paths.get_full_path("upscale_models", rel)
    if not path:
        raise FileNotFoundError(rel)
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if "module.layers.0.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    desc = spandrel.ModelLoader().load_from_state_dict(sd)
    if not isinstance(desc, ImageModelDescriptor):
        raise ValueError(f"{rel}: not an image upscale model ({type(desc).__name__})")
    return desc.eval()

def _sr_pass(sr_model, x, batch_size=4):
    """单次放大: [N,H,W,3] float 0..1 → [N,H*s,W*s,3]。分批执行, OOM 自动减半批。
    batch_size: 每次前向帧数上限 — SR 走 spandrel 直进显存、不经 ComfyUI 模型管理,
    峰值显存 ≈ batch × 窗口² 特征图; 16GB 卡建议 2~4, 24GB+ 可 8~16。"""
    dev = comfy.model_management.get_torch_device()
    sr_model.to(dev)
    out, n, bs, i = [], int(x.shape[0]), max(1, min(int(batch_size), int(x.shape[0]))), 0
    while i < n:
        try:
            b = x[i:i + bs].movedim(-1, 1).to(dev)
            with torch.no_grad():
                o = sr_model(b)
            out.append(o.movedim(1, -1).clamp(0.0, 1.0).cpu())
            i += bs
            del b, o
        except comfy.model_management.OOM_EXCEPTION:
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            comfy.model_management.soft_empty_cache()
    return torch.cat(out, dim=0)

# ================= 分镜检测 (先分镜, 后面部 — 官方 PySceneDetect 管线) =================
def _detect_shot_cuts_official_file(frames_rgb, threshold, fps_hint):
    """官方用法 (与 Element_scene_detection.detect_scenes_direct 逐字相同):
    临时视频文件 + open_video + SceneManager + ContentDetector + get_scene_list。
    fps 只影响临时文件的时长元数据, 不影响切点帧号 (量化到 1/1001 无副作用)。"""
    from scenedetect import SceneManager, ContentDetector, open_video
    import av as _av
    F = int(frames_rgb.shape[0])
    H, W = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    fps = float(fps_hint) if fps_hint and float(fps_hint) > 0 else 24.0
    He, We = H - (H & 1), W - (W & 1)  
    tmpdir = tempfile.mkdtemp(prefix="h3_shot_")
    tmppath = os.path.join(tmpdir, "frames.mp4")
    try:
        container = _av.open(tmppath, mode="w")
        from fractions import Fraction
        fps_q = Fraction(int(round(fps * 1001)), 1001)
        vstream = container.add_stream("libx264", rate=fps_q)
        vstream.width, vstream.height = We, He
        vstream.pix_fmt = "yuv420p"
        vstream.options = {"crf": "16"}
        for i in range(F):
            img = np.ascontiguousarray(frames_rgb[i][:He, :We])
            for pkt in vstream.encode(_av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(pkt)
        for pkt in vstream.encode():
            container.mux(pkt)
        container.close()
        video = open_video(tmppath)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=float(threshold)))
        scene_manager.detect_scenes(video, show_progress=False)
        scenes = scene_manager.get_scene_list()
        return [scene[0].get_frames() for scene in scenes[1:]]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _detect_shot_cuts(frames_rgb, threshold, fps_hint=24.0):
    """分镜切点 (官方 PySceneDetect 管线): 临时视频文件 + open_video + SceneManager。
    切点 = 新镜头第一帧 (scene[0].get_frames())。未安装 scenedetect → 仅警告,
    按无切点处理 (保留上游分段隔离)。"""
    F = int(frames_rgb.shape[0])
    if F < 4:
        return []
    try:
        cuts = _detect_shot_cuts_official_file(frames_rgb, threshold, fps_hint)
        h3ff.log(f"[H3-FaceCut] PySceneDetect (official open_video+SceneManager): "
                 f"{len(cuts)} cut(s) @ threshold={float(threshold):g}\n"
                 f"[H3-FaceCut] PySceneDetect (官方 open_video+SceneManager): "
                 f"阈值={float(threshold):g} 检出 {len(cuts)} 个切点")
        return sorted({int(c) for c in cuts if 0 < int(c) < F})
    except Exception as e:
        h3ff.warn(f"[H3-FaceCut] PySceneDetect unavailable ({e}) — pip install scenedetect; "
                  f"shot cuts disabled, upstream-segment isolation only\n"
                  f"[H3-FaceCut] PySceneDetect 不可用 ({e}) — pip install scenedetect; "
                  f"镜头切点不可用, 仅保留上游分段隔离")
        return []


def _shot_bounds(blocks, cuts, F):
    """★ 镜头单元 = 仅按真实检测切点划分, 无缝覆盖 [0, F)。
    上游分段边界 (主采样生成接缝) 不再充当镜头边界: 接缝是生成账本, 不是内容边界 —
    主采样跨接缝锚定续接, 解码内容在接缝处连续, 面部轨迹必须同样连续。
    在接缝处切开会产生 [141,144) 这类孤立 3 帧单元: SeC 只有 3 帧上下文、
    窗口平滑被截断、漏检帧被迫跨切点插值 → 贴回错位。"""
    shots = []
    cur = 0
    for c in cuts:
        c = int(c)
        if cur < c < int(F):
            shots.append((cur, c))
            cur = c
    shots.append((cur, int(F)))
    return shots if shots else [(0, int(F))]



# ==================================================================
# ============ 逐帧平滑窗口 (同一身份不再直切) ============
# ==================================================================
def _smooth_window_sizes(face_sizes, expand_f, W, H, rate=_S_RATE):
    """逐帧窗口边长序列 (v20.3): S_t = 该帧脸尺寸×(1+余量) → 中值滤波(窗5) →
    变化率限速 → 包含性地板 → 钳制 [_win_floor(), min(W,H)]。
    合批改由放大期的"尺寸桶 + pad"承担, 本函数几何与 v20 完全一致。"""
    fl = _win_floor()
    cap = max(fl, min(int(W), int(H)))
    n = len(face_sizes)
    raw = []
    for s in face_sizes:
        s = float(s) * (1.0 + float(expand_f))
        raw.append(max(fl, min(cap, s)))
    med = [float(np.median(raw[max(0, i - 2): i + 3])) for i in range(n)]
    out = [med[0]]
    for i in range(1, n):
        prev, tgt = out[-1], med[i]
        max_d = prev * float(rate)
        nxt = prev + max(-max_d, min(max_d, tgt - prev))
        need = raw[i]
        if nxt < need:
            nxt = need
        out.append(min(cap, nxt))
    return [max(fl, int(round(s))) for s in out]

def _window_centers(boxes, S_seq, W, H):
    """逐帧钳制窗口中心 (v20): 脸完整在窗内 ∩ 窗完整在帧内 (构造性包含)。
    boxes 已按整条出现区间中值平滑、S_seq 已平滑 → 相邻帧中心天然连续;
    仅脸贴近画面边缘时钳制生效。面部宽于窗口 (宽幅帧) 退化为帧内跟随。"""
    n = len(boxes)
    px, py = [], []
    for k in range(n):
        b = boxes[k]
        half = S_seq[k] * 0.5
        cx = (b[0] + b[2]) * 0.5
        cy = (b[1] + b[3]) * 0.5
        lo_x = max(b[2] - half, half)        
        hi_x = min(b[0] + half, W - half)     
        lo_y = max(b[3] - half, half)
        hi_y = min(b[1] + half, H - half)
        if lo_x > hi_x:   
            lo_x = hi_x = min(max(cx, half), W - half)
        if lo_y > hi_y:
            lo_y = hi_y = min(max(cy, half), H - half)
        px.append(float(min(max(cx, lo_x), hi_x)))
        py.append(float(min(max(cy, lo_y), hi_y)))
    return [[a, b] for a, b in zip(px, py)]

def _blur_batch(crops, sigma, batch=32):
    if float(sigma) <= 0.0 or not len(crops):
        return crops
    dev = comfy.model_management.get_torch_device()
    s, r = float(sigma), max(1, int(round(3.0 * float(sigma))))
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32, device=dev) ** 2) / (2.0 * s * s))
    k = k / k.sum()
    out = [None] * len(crops)
    groups = {}
    for i, c in enumerate(crops):
        groups.setdefault(int(c.shape[1]), []).append(i)
    for _S, idxs in groups.items():
        i0, bs = 0, max(1, int(batch))
        while i0 < len(idxs):
            try:
                sel = idxs[i0:i0 + bs]
                x = torch.stack([crops[j] for j in sel]).to(dev)
                B, C = int(x.shape[0]), int(x.shape[1])
                kw = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)   # (K,1) 垂直核
                kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)   # (1,K) 水平核
                # pad 维度必须与核方向配对: 先 W-pad + 水平核, 后 H-pad + 垂直核。
                xp = torch.nn.functional.pad(x, (r, r, 0, 0), mode="replicate")
                xp = torch.nn.functional.conv2d(xp, kh, groups=C)
                xp = torch.nn.functional.pad(xp, (0, 0, r, r), mode="replicate")
                x = torch.nn.functional.conv2d(xp, kw, groups=C).cpu()

                for jj, j in enumerate(sel):
                    out[j] = x[jj]
                i0 += len(sel)
                del x
            except comfy.model_management.OOM_EXCEPTION:
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                comfy.model_management.soft_empty_cache()
    return out



def _sample_windows(frames, masks_full, use, centers, S_seq, dev=None):
    """GPU 批量版: 按 S 分组, 帧以 uint8 上卡、卡上转 float, 整批 grid_sample。
    mask 轻量, 保留 CPU 单帧路径。接口与 CPU 版完全一致。"""
    if dev is None:
        dev = comfy.model_management.get_torch_device()
    H, W = int(frames.shape[1]), int(frames.shape[2])
    crops = [None] * len(use)
    win_masks = [None] * len(use)
    groups = {}
    for k in range(len(use)):
        groups.setdefault(int(S_seq[k]), []).append(k)
    ax_cache = {}
    for S, ks in groups.items():
        i0, bs = 0, max(1, min(16, len(ks)))
        while i0 < len(ks):
            try:
                sub = ks[i0:i0 + bs]
                grids = []
                for k in sub:
                    if S not in ax_cache:
                        ax_cache[S] = torch.arange(S, dtype=torch.float32)
                    ax = ax_cache[S]
                    cx, cy = float(centers[k][0]), float(centers[k][1])
                    xs = ((cx - S * 0.5 + 0.5 + ax) / W) * 2.0 - 1.0
                    ys = ((cy - S * 0.5 + 0.5 + ax) / H) * 2.0 - 1.0
                    grids.append(torch.stack([xs[None, :].expand(S, S),
                                              ys[:, None].expand(S, S)], dim=-1))
                g = torch.stack(grids, dim=0).to(dev)                     # [B,S,S,2]
                fr = torch.from_numpy(
                    frames[[use[k] for k in sub]]                         # numpy 花式索引取 B 帧
                ).to(dev).permute(0, 3, 1, 2).float()                     # [B,3,H,W]
                c = torch.nn.functional.grid_sample(
                    fr, g, mode="bilinear", padding_mode="border",
                    align_corners=False).cpu()                            # [B,3,S,S]
                for jj, k in enumerate(sub):
                    crops[k] = c[jj]
                i0 += len(sub)
                del fr, c
            except comfy.model_management.OOM_EXCEPTION:
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                comfy.model_management.soft_empty_cache()
    for k, j in enumerate(use):                                           # masks: CPU, 轻量
        m = masks_full[j] if (masks_full is not None and j < len(masks_full)) else None
        if m is not None:
            S = int(S_seq[k]); cx, cy = float(centers[k][0]), float(centers[k][1])
            ax = torch.arange(S, dtype=torch.float32)
            xs = ((cx - S * 0.5 + 0.5 + ax) / W) * 2.0 - 1.0
            ys = ((cy - S * 0.5 + 0.5 + ax) / H) * 2.0 - 1.0
            grid = torch.stack([xs[None, :].expand(S, S),
                                ys[:, None].expand(S, S)], dim=-1).unsqueeze(0)
            mt = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
            win_masks[k] = torch.nn.functional.grid_sample(
                mt, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    return crops, win_masks


def _partition_appearance(sm_sz, S_seq, res, scale_split, skip_thr):
    """一次出现内的最优分段 (DP)。v20.2: gsum 改前缀和 O(1) 取段代价 + 段长上限 _SEG_MAX,
    复杂度 O(n·_SEG_MAX)。其余语义与 v20 一致。"""
    n = len(sm_sz)
    if n < 5:
        return []
    INF = float("inf")
    pre = [0.0] * (n + 1)
    for k in range(n):
        pre[k + 1] = pre[k] + res / float(max(1, int(S_seq[k])))
    dp = [0.0] + [INF] * n
    prev = [-1] * (n + 1)
    for i in range(1, n + 1):
        mx, mn = 0.0, INF
        for a in range(i - 1, max(0, i - _SEG_MAX) - 1, -1):
            s = sm_sz[a]
            if s > mx:
                mx = s
            if s < mn:
                mn = s
            L = i - a
            if L < 5:
                continue
            gsum = pre[i] - pre[a]
            if mn >= skip_thr:
                c = 0.0
            else:
                c = 22.0 + gsum
                if mx > mn * float(scale_split):
                    c += (mx / (mn * float(scale_split)) - 1.0) * gsum * 2.0
            if dp[a] + c < dp[i]:
                dp[i] = dp[a] + c
                prev[i] = a
    if dp[n] == INF:
        return []
    out, i = [], n
    while i > 0:
        out.append((prev[i], i))
        i = prev[i]
    return out[::-1]


def _build_subtracks_for_track(boxes_seq, frames, H, W, res, expand_f, scale_split,
                               skip_ratio, gap_tol, tid, masks_full=None,
                               size_override=None, shot_id=None, sr_model=None, sr_batch=4, pre_blur=0.0, pbar=None):
    """单条身份轨迹 → (subtracks, crop_parts, n_rows)。
    分工: SeC-4B 身份/mask, YOLO 逐帧实测尺寸; 本函数只做应用层几何:
    DP 分段 (skip 边界) → 逐帧平滑窗口 (S_t 与中心均按整条出现区间计算并时序平滑)
    → 采样 → 账本。检测帧全部分配, 不丢弃。
    镜头单元约束: appearance 合并与框插值禁止跨镜头单元 (切镜头处强制断开, 防跳变)。
    v20.1: 修复逐帧缩放的 BCHW 布局错误; 相邻分段共享同一套连续的 (S_t, center_t)
    序列 → 段边界不再直切; 小脸占比恒为 1/(1+余量), 与脸绝对大小无关。
    v19: 不按 17n+5 掐尾 — 全部帧保留, 网格约束由 Face_Resample 编码期补齐。"""
    F = len(boxes_seq)
    det_idx = [i for i, b in enumerate(boxes_seq) if b is not None]
    subtracks, crop_parts, n_rows = [], [], 0
    if not det_idx:
        return subtracks, crop_parts, 0

    intervals = []
    for i in det_idx:
        pe = intervals[-1][1] if intervals else -1
        if intervals and i - pe - 1 <= int(gap_tol):
            intervals[-1][1] = i
            intervals[-1][2].append(i)
        else:
            intervals.append([i, i, [i]])


    filled = [None] * F
    for a, b, dets in intervals:
        for j in dets:
            filled[j] = list(boxes_seq[j])
        for j in range(a, b + 1):
            if filled[j] is None:
                if shot_id is not None:
                    p = max((d for d in dets if d < j and int(shot_id[d]) == int(shot_id[j])), default=None)
                    n = min((d for d in dets if d > j and int(shot_id[d]) == int(shot_id[j])), default=None)
                else:
                    p = max((d for d in dets if d < j), default=None)
                    n = min((d for d in dets if d > j), default=None)
                if p is not None and n is not None:
                    r = (j - p) / float(n - p)
                    filled[j] = [x + (y - x) * r for x, y in zip(boxes_seq[p], boxes_seq[n])]
                elif p is not None:
                    filled[j] = list(boxes_seq[p])  
                elif n is not None:
                    filled[j] = list(boxes_seq[n])


    def _fs(i):
        if size_override is not None and i < len(size_override) and size_override[i]:
            return float(size_override[i])
        if filled[i] is not None:
            return _size(filled[i])
        return 0.0


    pieces = []
    for a, b, _dets in intervals:
        seg_starts = [a]
        for i in range(a + 1, b):
            if shot_id is not None and int(shot_id[i]) != int(shot_id[i - 1]):
                seg_starts.append(i)
        seg_starts.append(b + 1)
        for ra, rb in zip(seg_starts[:-1], seg_starts[1:]):
            raw_boxes = _median_smooth([filled[j] for j in range(ra, rb)])
            raw_sz = [_fs(i) for i in range(ra, rb)]
            sm_sz = [float(np.median(raw_sz[max(0, k - 2): k + 3])) for k in range(len(raw_sz))]
            S_seq = _smooth_window_sizes(sm_sz, expand_f, W, H)
            if (rb - ra) < 5:
                pieces.append((ra, rb, S_seq, raw_boxes))
                continue
            for pa, pb in _partition_appearance(sm_sz, S_seq, res, float(scale_split), res * float(skip_ratio)):
                pieces.append((ra + pa, ra + pb, S_seq[pa:pb], raw_boxes[pa:pb]))
    if not pieces:
        return subtracks, crop_parts, 0

    for si, (pa, pb, S_seq, boxes) in enumerate(pieces):
        idxs = list(range(pa, pb))
        med = float(np.median([_fs(j) for j in idxs]))
        if med <= 0:
            med = float(np.median([_size(b) for b in boxes]))
        f0_all, f1_all = pa, pb
        if med >= res * float(skip_ratio):
            h3ff.vlog(f"[H3-FaceCut] id{tid} sub{si+1} [{f0_all},{f1_all}) skipped: "
                      f"face {med:.0f}px >= {float(skip_ratio):.2f}xres({res})\n"
                      f"[H3-FaceCut] 身份{tid} 子轨{si+1} [{f0_all},{f1_all}) 跳过: "
                      f"脸 {med:.0f}px 已够大")
            subtracks.append({"track_id": tid, "f0": f0_all, "f1": f1_all, "S": 0,
                              "face_med": round(med, 1), "skip": True,
                              "centers": [], "crop_off": None})
            if pbar is not None:
                pbar.update(len(idxs))
            continue

        centers = _window_centers(boxes, S_seq, W, H)
        crops, win_masks = _sample_windows(frames, masks_full, idxs, centers, S_seq)

        have = [i for i, m in enumerate(win_masks) if m is not None]
        if have:
            for i in range(len(win_masks)):
                if win_masks[i] is None:
                    win_masks[i] = win_masks[min(have, key=lambda k: abs(k - i))]

        # crops, win_masks = _sample_windows(frames, masks_full, idxs, centers, S_seq)
        if float(pre_blur) > 0.0:
            crops = _blur_batch(crops, float(pre_blur))
        have = [i for i, m in enumerate(win_masks) if m is not None]

        proc = [None] * len(crops)
        if sr_model is not None:
            TOL = max(8, int(_SR_BUCKET))
            buckets = {}
            for _ci, _c in enumerate(crops):
                _K = -(-int(_c.shape[1]) // TOL) * TOL
                buckets.setdefault(_K, []).append(_ci)
            h3ff.vlog(f"[H3-FaceCut] id{tid} sub{si+1}: SR buckets={len(buckets)} "
                      f"(TOL={TOL}px), batch<={int(sr_batch)}")
            for _K in sorted(buckets):
                _idxs_g = buckets[_K]
                frames_p = []
                for _i in _idxs_g:
                    c_ = crops[_i]
                    p = _K - int(c_.shape[1])
                    frames_p.append(
                        torch.nn.functional.pad(c_, (0, p, 0, p), mode="replicate")
                        if p > 0 else c_)
                x = torch.stack(frames_p, dim=0).permute(0, 2, 3, 1) / 255.0
                n_pass = 0
                while int(x.shape[1]) < res and n_pass < 2:
                    x = _sr_pass(sr_model, x, batch_size=sr_batch)
                    n_pass += 1
                ratio = float(int(x.shape[1])) / float(_K)
                _dev = comfy.model_management.get_torch_device()
                for _j, _i in enumerate(_idxs_g):
                    S_i = int(crops[_i].shape[1])
                    S2 = int(round(S_i * ratio))
                    f_i = x[_j, :S2, :S2].contiguous()          # [S2,S2,3] HWC, CPU
                    if (int(f_i.shape[0]), int(f_i.shape[1])) != (res, res):
                        ft = f_i.permute(2, 0, 1).unsqueeze(0).to(_dev)   # [1,3,S2,S2]
                        ft = torch.nn.functional.interpolate(
                            ft, size=(res, res), mode="bicubic", antialias=True)
                        f_i = ft.clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu()
                    proc[_i] = f_i

                if pbar is not None:
                    pbar.update(len(_idxs_g))
        else:
            proc = _gpu_resize_batch(crops, res)
            if pbar is not None:
                pbar.update(len(crops))

        crop_t = torch.stack(proc, dim=0).contiguous()        

        entry = {"track_id": tid, "f0": int(idxs[0]), "f1": int(idxs[-1]) + 1,
                 "S": int(round(float(np.median(S_seq)))),
                 "S_list": [int(s) for s in S_seq],   
                 "face_med": round(med, 1), "skip": False,
                 "centers": centers, "crop_off": n_rows}
        if have:
            m_t = torch.stack([
                torch.nn.functional.interpolate(
                    m[None], size=(res, res), mode="bilinear", align_corners=False)[0]
                for m in win_masks
            ], dim=0)  # [K,1,res,res]


            raw_max = float(max(float(m.max()) for m in win_masks))
            entry["masks"] = m_t[:, 0].clamp(0.0, 1.0).mul(255.0).to(torch.uint8).contiguous()
            if int(entry["masks"].max()) == 0:
                h3ff.warn(f"[H3-FaceCut] id{tid} sub{si+1}: WARNING masks all zero after scale "
                          f"(raw window max={int(raw_max * 255)})\n"
                          f"[H3-FaceCut] 身份{tid} 子轨{si+1}: 警告 缩放后 mask 全零 "
                          f"(缩放前窗口最大值={int(raw_max * 255)})")
        else:
            h3ff.vlog(f"[H3-FaceCut] id{tid} sub{si+1}: no SeC mask (blend will use feathered box)\n"
                      f"[H3-FaceCut] 身份{tid} 子轨{si+1}: 无 SeC mask (贴回将回退羽化框)")
        subtracks.append(entry)
        crop_parts.append(crop_t)
        n_rows += len(idxs)
        S_med = int(np.median(S_seq))
        occ_log = (med / float(S_med)) if S_med > 0 else 0.0
        h3ff.vlog(f"[H3-FaceCut] id{tid} sub{si+1} [{f0_all},{f1_all}) "
                  f"S={S_med}px (逐帧平滑 S∈[{min(S_seq)},{max(S_seq)}]) face~{med:.0f}px "
                  f"occ≈{occ_log:.0%} follow-cam -> {res}² rows(local) "
                  f"[{n_rows - len(idxs)},{n_rows}) ({len(idxs)} frames kept)"
                  f"{', mask: on' if have else ''}\n"
                  f"[H3-FaceCut] 身份{tid} 子轨{si+1} [{f0_all},{f1_all}) "
                  f"S={S_med}px (逐帧平滑, 本段范围 [{min(S_seq)},{max(S_seq)}]) "
                  f"脸~{med:.0f}px 占比≈{occ_log:.0%} 跟随窗口 → {res}² 局部行 "
                  f"[{n_rows - len(idxs)},{n_rows}) (保留 {len(idxs)} 帧)"
                  f"{', mask: 开' if have else ''}")
    return subtracks, crop_parts, n_rows

def _fill_mask_rows(subtracks, cursor, res):
    """按子轨账目把 pack 内置 masks 铺成端口行张量 (缓存命中/未命中共用)。"""
    mask_rows = torch.zeros(max(1, int(cursor)), res, res, dtype=torch.float32)
    for st in subtracks:
        m = st.get("masks")
        if m is None or st.get("skip") or st.get("crop_off") is None:
            h3ff.vlog(f"[H3-FaceCut] port skip: tid={st.get('track_id', '?')} "
                      f"masks={'None' if m is None else 'ok'} skip={st.get('skip')} "
                      f"crop_off={st.get('crop_off')}\n"
                      f"[H3-FaceCut] 端口跳过: 身份={st.get('track_id', '?')} "
                      f"masks={'无' if m is None else '有'} skip={st.get('skip')} "
                      f"crop_off={st.get('crop_off')}")
            continue
        off, K = int(st["crop_off"]), int(st["f1"]) - int(st["f0"])
        fits = (off + K <= int(mask_rows.shape[0])) and (int(m.shape[0]) == K)
        h3ff.vlog(f"[H3-FaceCut] port fill: [{st['f0']},{st['f1']}) off={off} K={K} "
                  f"m.max={int(m.max())} fits={fits}\n"
                  f"[H3-FaceCut] 端口填充: 子轨 [{st['f0']},{st['f1']}) 偏移={off} "
                  f"行数={K} m最大值={int(m.max())} 适配={fits}")
        if fits:
            mask_rows[off:off + K] = m.to(torch.float32).mul(1.0 / 255.0)
    if int(cursor) == 0:
        mask_rows = mask_rows[:1]
    return mask_rows

def _track_work(boxes_seq, gap_tol):
    """一条身份轨迹的实际处理帧数 (与 _build_subtracks_for_track 的 gap_tol 区间合并
    逐字一致): 每个出现区间内所有帧 (含插值帧) 恰好计一次进度。"""
    det_idx = [i for i, b in enumerate(boxes_seq) if b is not None]
    if not det_idx:
        return 0
    total, a, b = 0, det_idx[0], det_idx[0]
    for i in det_idx[1:]:
        if i - b - 1 <= int(gap_tol):
            b = i
        else:
            total += b - a + 1
            a = b = i
    return total + (b - a + 1)

def _gpu_resize_batch(crops, res, batch=16):
    """[3,S,S] float 0..255 → [res,res,3] float 0..1。按 S 分组(S_t 逐帧可变),
    GPU bicubic(antialias) 分批, OOM 减半批, 结果回 CPU。"""
    if not len(crops):
        return []
    dev = comfy.model_management.get_torch_device()
    out = [None] * len(crops)
    groups = {}
    for i, c in enumerate(crops):
        groups.setdefault(int(c.shape[1]), []).append(i)
    for _S, idxs in groups.items():
        i0, bs = 0, max(1, int(batch))
        while i0 < len(idxs):
            try:
                sel = idxs[i0:i0 + bs]
                x = torch.stack([crops[j] for j in sel]).to(dev) / 255.0
                x = torch.nn.functional.interpolate(
                    x, size=(int(res), int(res)),
                    mode="bicubic", antialias=True).clamp(0.0, 1.0)
                for jj, j in enumerate(sel):
                    out[j] = x[jj].permute(1, 2, 0).cpu()
                i0 += len(sel)
                del x
            except comfy.model_management.OOM_EXCEPTION:
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                comfy.model_management.soft_empty_cache()
    return out



def _build_pack(a_lat, subtracks, cursor, multi_track_flag, meta, fps_eff):
    return {"version": 7, "multi_track": bool(multi_track_flag), "a_lat": a_lat,
            "subtracks": subtracks, "fps_eff": float(fps_eff),
            "n_crop_rows": int(cursor), "meta": meta}

class H3FaceCut(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        choices = _list_models() or [""]
        sec_choices = h3ff.list_sec_models()
        return io.Schema(
            node_id="H3FaceCut",
            display_name="Minimax_H3_Face_Cut",
            category="MinimaxH3_AutoContext/FaceFix",
            description=("Face fix step 1: shot-aware"
                         "latent is OPTIONAL: fully untouched in images mode. Outputs "
                         "shot_info for the parameter node.\n"
                         "auto-detected; tracking mode is chosen by the sec_model dropdown "
                         "(None = single face).\n"
                         "修脸第 1 步: 分镜优先 (官方 PySceneDetect, 依赖缺失自动降级) + YOLO 检测 + 可选内置 "
                         "latent 为可选输入: images 模式完全不处理。输出 shot_info 供 parameter 节点分配提示词。\n"
                         "追踪模式由 sec_model 下拉框决定 (None = 单脸)。"),
            inputs=[
                io.Combo.Input("face_model", options=choices, tooltip="YOLO face model (ComfyUI/models/elementEasy)\n人脸检测模型"),
                io.Latent.Input("latent", optional=True, tooltip="Optional. Required in latent mode (no images connected). In images mode: "
                                 "NOT unpacked / NOT validated / NOT hashed / NOT decoded — completely "
                                 "untouched (audio ledger is provided via Face_Resample audio port)\n"
                                 "可选。latent 模式 (未连 images) 必须提供。images 模式下完全不处理 "
                                 "(不解包/不校验/不 hash/不解码) — 音频账本由 Face_Resample 的 audio 端口提供"),
                io.Image.Input("images", optional=True, tooltip="Optional external detection source: when connected, shot detection / "
                               "detection / SeC / crop run on these frames at NATIVE size \n "
                               "可选外部检测画面: 连接后 分镜检测/检测/SeC/裁剪 直接作用于这些帧 "),
                io.Vae.Input("vae", tooltip="Video VAE (latent mode only; ignored in images mode)\n视频 VAE (仅 latent 模式使用, images 模式忽略)"),
                io.Float.Input("yolo_threshold", default=0.3, min=0.05, max=0.9, step=0.05, tooltip="YOLO detection confidence threshold\nYOLO 检测置信度阈值"),

                io.Int.Input("yolo_batch", default=16, min=1, max=128, step=1,
                             tooltip="YOLO detection batch size (frames per forward pass). Use 4~8 on low-VRAM GPUs to avoid OOM; "
                                     "raise to 32~64 on high-VRAM GPUs for speed. Results are identical regardless of batch size\n"
                                     "YOLO 检测批大小 (每次前向的帧数)。低显存卡 (8~12GB) 建议降到 4~8 防 OOM；"
                                     "高显存卡可升到 32~64 提速。批大小不影响检测结果"),

                io.Float.Input("shot_threshold", default=40.0, min=5.0, max=100.0, step=0.5, tooltip="PySceneDetect ContentDetector threshold (higher = fewer cuts).\n"
                               "PySceneDetect ContentDetector 阈值 (越高切点越少)。"),
                io.Combo.Input("upscale_model", options=(["None"] + _list_upscale_models()), default="None",
                               tooltip="Optional upscale model (ComfyUI/models/upscale_models) for face crops. None = bicubic only. \n"
                                       "可选放大模型 (upscale_models 目录), None = 仅 bicubic (原行为)。"),
                io.Float.Input("pre_blur", default=0.0, min=0.0, max=8.0, step=0.5,
                    tooltip="Gaussian blur (sigma, source-window px) applied to each crop BEFORE "
                            "SR/bicubic upscale; 0 = off. Softens source noise & interpolation "
                            "jaggies\n放大/SR 前对裁剪窗口施加高斯模糊 (σ, 源窗口像素); 0 = 关闭"),

                io.Int.Input("res", default=512, min=256, max=2048, step=32, tooltip="Canvas side length\n画布边长"),
                io.Int.Input("expand", default=20, min=0, max=100, tooltip="Crop window margin % around the detected face box\n围绕检测面部外框的裁剪窗口余量%"),
                io.Float.Input("skip_ratio", default=_SKIP_RATIO, min=0.3, max=1.0, step=0.05, tooltip="Skip resampling when face >= res x this\n脸够大时跳过重采样"),
                io.Combo.Input("sec_model", options=(["None"] + sec_choices) if sec_choices else ["None"], default="None",
                               tooltip="SeC-4B weights in ComfyUI/models/sams (fp16 recommended). None = single-face "
                                       "mode (one face per frame); picking a weight enables multi-person identity \n"
                                       "models/sams 下的 SeC-4B 权重 (建议 fp16)。None = 单脸模式 (每帧单脸); "),
                io.Float.Input("sec_threshold", default=0.3, min=0.1, max=0.8, step=0.05, tooltip="IoU/overlap threshold to claim a YOLO box to an identity (multi_sec only)\n"
                               "检测框认领给某身份的重叠率阈值 (仅 multi_sec 模式)"),
                io.Int.Input("max_identities", default=6, min=1, max=12, tooltip="Max tracked identities per shot unit\n每个镜头单元最多追踪身份个数"),

                io.Boolean.Input("unload_main_models", default=True, tooltip="Unload H3/VAE/CLIP from VRAM after detection & SeC tracking, before the crop/SR stage "
                               "(recommended on 12~16GB cards \n"
                               "检测与 SeC 追踪完成后、裁剪/SR 放大前，把 H3 主模型/VAE/CLIP 移出显存 "
                               "(12~16GB 显存建议开启；SR 模型经 spandrel 直进显存、不受 ComfyUI 模型管理调度"),
                io.Int.Input("sr_batch", default=4, min=1, max=16, step=1, tooltip="Frames per SR forward pass for crop upscale. Peak VRAM scales with this; "
                               "4 on 16GB, 8~16 on 24GB+. Does not affect results, only speed/VRAM\n"
                               "裁剪放大阶段每次 SR 前向的帧数。显存峰值随此值增大；16GB 用 4，24GB+ 可 8~16。"
                               "不影响结果，只影响速度与显存"),
                               

                io.Boolean.Input("enable_cache", default=True, tooltip="Clear manually after changing model/VAE/SeC weights or CODE (use clear_cache)\n"
                                 "更换模型/VAE/SeC 权重或修改代码后请勾 clear_cache"),
                io.Boolean.Input("clear_cache", default=False, tooltip="Delete this node's cache directory before running\n运行前删除本节点的缓存目录"),
                io.Dict.Input("info", optional=True, tooltip="Main sampler info (h3_runtime), optional\n主采样 info (可选)"),
            ],
            outputs=[
                io.Image.Output(display_name="crop_images", tooltip="Uniform res^2 crops (row order = subtrack order, ALL frames kept)\n所有待采样子轨的统一 res² 裁剪 (全部帧保留)"),
                io.Dict.Output(display_name="face_pack", tooltip="Subtrack geometry pack (masks also kept inside for legacy wiring)\n子轨几何包 (mask 同时保留在包内, 兼容旧接线)"),
                io.Mask.Output(display_name="masks", tooltip="SeC masks [rows,res,res], rows 1:1 with crop_images/canvas rows, 1=face\n"
                               "SeC mask [行数,res,res], 行与 crop_images/画布 1:1 对齐, 1=人脸"),
                io.Dict.Output(display_name="shot_info", tooltip="Shot map for Minimax_H3_AutoContext_parameter: shots/cuts/fps\n"
                               "供 parameter 节点分配分段提示词的镜头表 (shots/cuts/fps)"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, latent=None, vae=None, face_model="", yolo_threshold=0.3, yolo_batch=16,
                shot_threshold=40.0, upscale_model="None", res=512, expand=20, skip_ratio=_SKIP_RATIO,
                sec_model="None", sec_threshold=0.3, max_identities=6,
                unload_main_models=True, pre_blur=0.0, sr_batch=4, enable_cache=True, clear_cache=False,
                images=None, info=None) -> io.NodeOutput:
        shot_detect = True   
        shot_active = shot_detect and _has_scenedetect()
        gap_tol = _GAP_TOL
        scale_split = _SCALE_SPLIT
        mllm_memory_size = _SEC_MEM_SIZE
        use_flash_attn = _flash_attn_available()
        face_tracking = ("multi_sec" if str(sec_model) not in (None, "", "None") else "single")

        use_images = images is not None
        seg = info if isinstance(info, dict) else {}
        v_lat = a_lat = None
        imgs = None
        T = LH = LW = 0

        if use_images:
            imgs = images
            if imgs.dim() != 4 or int(imgs.shape[-1]) != 3:
                raise ValueError(
                    f"[H3-FaceCut] images must be [F,H,W,3], got {tuple(imgs.shape)}\n"
                    f"[H3-FaceCut] images 必须为 [F,H,W,3], 实际 {tuple(imgs.shape)}")
            H, W = int(imgs.shape[1]), int(imgs.shape[2])
            F_expect = int(imgs.shape[0])
            if F_expect < 4:
                raise ValueError(
                    f"[H3-FaceCut] external images frame count {F_expect} is too small\n"
                    f"[H3-FaceCut] 外部画面帧数 {F_expect} 过短")
            h3ff.log(f"[H3-FaceCut] images mode: canvas {W}x{H}, {F_expect} frames — "
                     f"latent NOT touched\n[H3-FaceCut] images 模式: 画布 {W}x{H}, "
                     f"{F_expect} 帧 — latent 完全不处理")
        else:
            if latent is None:
                raise ValueError(
                    "[H3-FaceCut] latent is required when images is not connected\n"
                    "[H3-FaceCut] 未连接 images 时必须提供 latent")
            v_lat, a_lat = h3_conditioning.unpack_nested_latent(latent)
            if v_lat is None or v_lat.dim() != 5 or a_lat is None:
                raise ValueError("[H3-FaceCut] latent must contain both video+audio\n"
                                 "[H3-FaceCut] latent 必须同时含 video+audio")
            B, C, T, LH, LW = v_lat.shape
            if T < 4:
                raise ValueError(f"[H3-FaceCut] video token count {T} is too small\n"
                                 f"[H3-FaceCut] 视频 token 数 {T} 过短")
            if vae is None:
                raise ValueError("[H3-FaceCut] vae is required in latent mode (images not connected)\n"
                                 "[H3-FaceCut] latent 模式 (未连接 images) 必须提供 vae")
            H, W = LH * 16, LW * 16
            F_expect = h3ff._pixels_for_tokens(int(T))

        model_rel = (face_model or "").strip()
        if not model_rel:
            raise ValueError(f"[H3-FaceCut] no detection model selected — put YOLO weights into: {MODEL_DIR}\n"
                             f"[H3-FaceCut] 未选择检测模型 — 请将 YOLO 权重放入: {MODEL_DIR}")
        model_path = os.path.normpath(os.path.join(MODEL_DIR, model_rel))
        if not os.path.isfile(model_path):
            raise ValueError(f"[H3-FaceCut] model not found: {model_path}\n"
                             f"[H3-FaceCut] 模型不存在: {model_path}")

        blocks = None
        if use_images:
            blocks = h3ff.face_split_blocks(
                seg.get("seg_sizes"), seg.get("effective_context"), decoded_frames=F_expect)
        else:
            blocks = h3ff.face_split_blocks(
                seg.get("seg_sizes"), seg.get("effective_context"),
                expect_tokens=int(T), boundaries=seg.get("boundaries"), decoded_frames=F_expect)
        if blocks is None:
            blocks = h3ff.token_blocks(int(T), 22)

        try:
            unique_id = cls.hidden.unique_id
        except AttributeError:
            unique_id = None
        if clear_cache and unique_id is not None:
            try:
                target = os.path.join(folder_paths.get_output_directory(), "cache", f"node_{unique_id}")
                if latent_cache.clear_cache_dir(target):
                    h3ff.log(f"[H3-FaceCut] 🗑️ Cache cleared: {target}\n[H3-FaceCut] 🗑️ 缓存已清除: {target}")
                else:
                    h3ff.vlog("[H3-FaceCut] ℹ️ Cache dir does not exist, nothing to clear\n[H3-FaceCut] ℹ️ 缓存目录不存在，无需清除")
            except Exception as e:
                h3ff.warn(f"[H3-FaceCut] ⚠️ Failed to clear cache: {e}\n[H3-FaceCut] ⚠️ 清除缓存失败: {e}")
        cache_dir = ""
        if enable_cache and unique_id is not None:
            try:
                cache_dir = os.path.join(folder_paths.get_output_directory(), "cache", f"node_{unique_id}")
            except Exception:
                cache_dir = ""

        def _file_stat_sig(p):
            """同名模型文件被替换 (重新下载/更新) 的廉价检测: 大小+mtime。"""
            try:
                if p and os.path.isfile(p):
                    st = os.stat(p)
                    return f"{int(st.st_size)}:{int(st.st_mtime)}"
            except Exception:
                pass
            return None

        _sr_fp_path = None
        try:
            if upscale_model and str(upscale_model) != "None":
                _sr_fp_path = folder_paths.get_full_path("upscale_models", str(upscale_model))
        except Exception:
            _sr_fp_path = None
        _sec_fp_path = None
        if face_tracking == "multi_sec":
            try:
                _sec_fp_path, _ = h3ff._resolve_sec_model_path(str(sec_model))
            except Exception:
                _sec_fp_path = None

        # 指纹补全: VAE 权重 (解码出的裁剪行直接受其影响) + 模型文件实体
        _fp_common = {"win_v": 3,
                      "face_model": model_rel, "yolo_threshold": float(yolo_threshold),
                      "yolo_file": _file_stat_sig(model_path),
                      "shot_active": bool(shot_active), "shot_threshold": float(shot_threshold),
                      "upscale_model": str(upscale_model or "None"),
                      "pre_blur": float(pre_blur),
                      "sr_file": _file_stat_sig(_sr_fp_path),
                      "sec_file": _file_stat_sig(_sec_fp_path),
                      "vae_fp": latent_cache.vae_fingerprint(vae) if vae is not None else None,
                      "res": int(res), "expand": int(expand),
                      "gap_tol": int(gap_tol), "scale_split": float(scale_split),
                      "skip_ratio": float(skip_ratio), "face_tracking": str(face_tracking),
                      "sec_model": str(sec_model), "sec_threshold": float(sec_threshold),
                      "mllm_memory_size": int(mllm_memory_size), "use_flash_attn": bool(use_flash_attn),
                      "max_identities": int(max_identities)}
        if use_images:
            fp = {"src": "images", "images_digest": _images_digest(imgs),
                  "F": int(F_expect), "W": int(W), "H": int(H), **_fp_common}
        else:
            fp = {"src": "latent", "latent_hash": latent_cache.compute_input_hash(latent),
                  "T": int(T), "LH": int(LH), "LW": int(LW),
                  "blocks_hash": hashlib.md5(str(blocks).encode()).hexdigest(), **_fp_common}

        blob = "cut_result_ext.pt" if use_images else "cut_result.pt"
        cached = (latent_cache.load_blob(cache_dir, blob, fp, sensitive_keys=list(fp.keys()))
                  if cache_dir else None)
        if isinstance(cached, dict) and cached.get("subtracks") is not None \
                and cached.get("crop_images") is not None:
            try:
                crop_images = cached["crop_images"].float()
                subtracks = cached["subtracks"]
                cursor = int(cached["n_crop_rows"])
                mask_rows = _fill_mask_rows(subtracks, cursor, res)
                fps_eff = _fps_eff(a_lat, F_expect, seg)
                _cmeta = cached.get("meta") or {}
                pack = _build_pack(a_lat, subtracks, cursor, bool(cached.get("multi_track")), _cmeta, fps_eff)
                shot_info = {"shots": [[int(s), int(e)] for s, e in (_cmeta.get("shots") or [[0, int(F_expect)]])],
                             "shot_cuts": [int(c) for c in (_cmeta.get("shot_cuts") or [])],
                             "n_shots": int(_cmeta.get("n_shots") or len(_cmeta.get("shots") or [])) or 1,
                             "fps": float(fps_eff)}
                mz = float(mask_rows.max()) if mask_rows.numel() else 0.0
                print("\033[33m" + f"[H3-FaceCut] cache hit: {len(subtracks)} subtracks, "
                      f"crop {tuple(crop_images.shape)}, masks_max={mz:.2f} — shot/decode/detect/SeC skipped\n"
                      f"[H3-FaceCut] 缓存命中: {len(subtracks)} 条子轨, 裁剪 {tuple(crop_images.shape)}, "
                      f"mask最大值={mz:.2f} — 已跳过 分镜/解码/检测/SeC" + "\033[0m")
                return io.NodeOutput(crop_images.contiguous(), pack, mask_rows.contiguous(), shot_info)
            except Exception as e:
                h3ff.warn(f"[H3-FaceCut] cache load failed ({e}), recomputing\n[H3-FaceCut] 缓存载入失败 ({e})，重新计算")

        per_frame = [None] * F_expect
        opt = {"conf": float(yolo_threshold), "model": model_path, "batch_size": int(yolo_batch)}
        if use_images:
            frames = np.clip(imgs.detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
            per_frame = h3ff.detect_faces(frames, opt)
            n_hit = sum(1 for b in per_frame if b)
            h3ff.log(f"[H3-FaceCut] input source: external images ({F_expect} frames @ {W}x{H}), "
                     f"detected {n_hit}/{F_expect}\n"
                     f"[H3-FaceCut] 输入来源: 外部画面 ({F_expect} 帧 @ {W}x{H}), "
                     f"已跳过 latent 解码, 检出 {n_hit}/{F_expect} 帧")
        else:
            frames = np.empty((F_expect, H, W, 3), dtype=np.uint8)
            for bi, (dk0, dk1, kp0, kp1) in enumerate(blocks):
                comfy.model_management.throw_exception_if_processing_interrupted()
                blk = h3ff.decode_probe_frames(vae, v_lat, range(dk0, dk1))
                off = kp0 - h3ff._pixels_for_tokens(dk0)
                if int(blk.shape[0]) < off + (kp1 - kp0):
                    raise RuntimeError(f"[H3-FaceCut] block {bi+1}: decoded {int(blk.shape[0])} frames < expected {off + kp1 - kp0}\n"
                                       f"[H3-FaceCut] 块 {bi+1}: 解码帧数不足")
                seg_px = blk[off: off + (kp1 - kp0)]
                frames[kp0:kp1] = seg_px
                per_frame[kp0:kp1] = h3ff.detect_faces(seg_px, opt)
                del blk, seg_px
            n_hit = sum(1 for b in per_frame if b)
            h3ff.log(f"[H3-FaceCut] detected {n_hit}/{F_expect} frames\n[H3-FaceCut] 检出 {n_hit}/{F_expect} 帧")

        res = int(res)
        expand_f = max(float(expand) / 100.0, 0.10)
        fps_eff = _fps_eff(a_lat, F_expect, seg)

        shot_cuts, shots = [], [(0, int(F_expect))]
        if bool(shot_active):
            _rt = seg.get("h3_runtime") if isinstance(seg.get("h3_runtime"), dict) else {}
            fps_hint = float(_rt.get("fps") or seg.get("fps") or 24.0)
            shot_cuts = _detect_shot_cuts(frames, float(shot_threshold), fps_hint)
            shots = _shot_bounds(blocks, shot_cuts, F_expect)
            if shot_cuts:
                h3ff.log(f"[H3-FaceCut] shot map: {len(shots)} unit(s), cuts={shot_cuts}\n"
                         f"[H3-FaceCut] 镜头划分: {len(shots)} 个处理单元, 切点={shot_cuts}")
            else:
                h3ff.vlog(f"[H3-FaceCut] shot map: no cuts ({len(shots)} unit(s))\n"
                          f"[H3-FaceCut] 镜头划分: 无切点 ({len(shots)} 个处理单元)")
        elif shot_detect:
            h3ff.warn("[H3-FaceCut] shot_detect ON but scenedetect is not installed — "
                      "pip install scenedetect; falling back to upstream-segment isolation only\n"
                      "[H3-FaceCut] 分镜检测已开启但未安装 scenedetect — pip install scenedetect; "
                      "已降级为仅上游分段隔离")
        shot_id = np.zeros(F_expect, dtype=np.int64)
        for _si, (_s, _e) in enumerate(shots):
            shot_id[_s:_e] = _si

        tracks = None
        
        if unload_main_models:
            try:
                comfy.model_management.unload_all_models()
            except Exception:
                pass
            comfy.model_management.soft_empty_cache()
        
        if face_tracking == "multi_sec" and any(per_frame):
            sec_handle = h3ff.load_sec_model(model_file=str(sec_model), device="auto",
                                             use_flash_attn=use_flash_attn, allow_mask_overlap=True)
            try:
                tracks = []
                nid = 0
                _prev_unit_ids = []   
                for (s0, e0) in shots:
                    sub_det = per_frame[s0:e0]
                    if not any(sub_det):
                        _prev_unit_ids = []   
                        continue
                    st_list = h3ff.sec_track_identities(
                        sec_handle, frames[s0:e0], sub_det,
                        tracking_direction="bidirectional",
                        mllm_memory_size=int(mllm_memory_size),
                        offload_video_to_cpu=True,
                        iou_thr=float(sec_threshold),
                        max_ids=int(max_identities),
                        tag=(f" [shot {s0}-{e0}]" if len(shots) > 1 else ""))
                    got = 0
                    n_new = len(st_list or [])                                  
                    _continue_id = (len(_prev_unit_ids) == 1 and n_new == 1)    
                    for t in (st_list or []):
                        boxes_f = [None] * F_expect
                        masks_f = [None] * F_expect
                        for kk, b in enumerate(t["boxes"]):
                            if b is not None:
                                boxes_f[s0 + kk] = b
                        if t.get("masks") is not None:
                            for kk, m in enumerate(t["masks"]):
                                if kk < len(masks_f) and m is not None:
                                    masks_f[s0 + kk] = m
                        if _continue_id:                        
                            _id = _prev_unit_ids[0]             
                        else:
                            _id = nid
                            nid += 1
                        tracks.append({"id": _id, "anchor": s0 + int(t["anchor"]),
                                       "boxes": boxes_f, "masks": masks_f})
                        got += 1
                    _prev_unit_ids = [t["id"] for t in tracks[-got:]] if got > 0 else []   
                    if got:
                        h3ff.vlog(f"[H3-FaceCut] shot [{s0},{e0}): {got} identit(ies)"
                                  f"{', continued from previous unit (single-identity link)' if _continue_id else ''}\n"
                                  f"[H3-FaceCut] 镜头 [{s0},{e0}): {got} 个身份"
                                  f"{', 单人续接上一单元身份' if _continue_id else ''}")

                if tracks:
                    _n_distinct = len({t["id"] for t in tracks})
                    h3ff.log(f"[H3-FaceCut] identities tracked across {len(shots)} shot unit(s): "
                             f"{len(tracks)} track(s) -> {_n_distinct} distinct id(s) "
                             f"(flash_attn={'on' if use_flash_attn else 'off'})\n"
                             f"[H3-FaceCut] 跨 {len(shots)} 个镜头单元共追踪到 {len(tracks)} 条轨迹 "
                             f"-> {_n_distinct} 个独立身份 (flash_attn={'开' if use_flash_attn else '关'})")

                else:
                    h3ff.log("[H3-FaceCut] SeC returned no identities, falling back to single-face\n"
                             "[H3-FaceCut] SeC 未产出身份, 回退单脸路径")
                    tracks = None
            finally:
                h3ff.unload_sec_model()
        elif face_tracking == "multi_sec":
            h3ff.log("[H3-FaceCut] multi_sec requested but no detections, falling back to single-face\n"
                     "[H3-FaceCut] multi_sec 模式但无任何检测框, 回退单脸路径")

        sr_model, sr_name = None, str(upscale_model or "None")
        if sr_name != "None":
            try:
                sr_model = _load_sr_model(sr_name)
                h3ff.log(f"[H3-FaceCut] crop upscale: {sr_name} (SR chain -> {res}px, max 2 passes)\n"
                         f"[H3-FaceCut] 裁剪放大: {sr_name} (SR 链 → {res}px, 最多 2 次)")
            except Exception as e:
                h3ff.warn(f"[H3-FaceCut] upscale model load failed ({e}) — falling back to bicubic\n"
                          f"[H3-FaceCut] 放大模型加载失败 ({e}) — 回退 bicubic")
                sr_model = None

        if tracks:
            src_tracks = []
            for t in tracks:
                sz = _yolo_size_track(t["boxes"], per_frame) if per_frame is not None else None
                src_tracks.append((t["id"], t["boxes"], t.get("masks"), sz))
            
            _measured, _span = set(), set()
            for _i, _b, _m, _sz in src_tracks:
                _valid = [k for k, b in enumerate(_b) if b is not None]
                if not _valid:
                    continue
                _span.update(range(_valid[0], _valid[-1] + 1))
                if _sz is not None:
                    _measured.update(k for k, v in enumerate(_sz) if v is not None)
            n_meas, n_all = len(_measured), len(_span)
            h3ff.log(f"[H3-FaceCut] size signal: YOLO measured {n_meas}/{n_all} frames "
                     f"(SeC box fallback on the rest)\n"
                     f"[H3-FaceCut] 尺寸信号: {n_meas}/{n_all} 帧用 YOLO 实测 (其余回退 SeC 框)")

        else:
            filled_single = [list(max(per_frame[j], key=lambda x: x[4])[:4]) if per_frame[j] else None
                             for j in range(F_expect)]
            src_tracks = [(0, filled_single, None, None)]

        subtracks, crop_parts = [], []
        cursor = 0
        _total_work = sum(_track_work(t[1], int(gap_tol)) for t in src_tracks)
        crop_pbar = comfy.utils.ProgressBar(max(1, _total_work)) if h3ff._HAS_COMFY else None

        for tid, boxes_seq, masks_full, sz_override in src_tracks:
            sts, cps, n_rows = _build_subtracks_for_track(
                boxes_seq, frames, H, W, res, expand_f, float(scale_split), float(skip_ratio),
                int(gap_tol), tid, masks_full=masks_full,
                size_override=(sz_override if face_tracking == "multi_sec" else None),
                shot_id=shot_id, sr_model=sr_model, sr_batch=sr_batch, pbar=crop_pbar, pre_blur=float(pre_blur))

            for st in sts:
                if not st["skip"]:
                    st["crop_off"] = cursor + int(st["crop_off"])  
            subtracks.extend(sts)
            crop_parts.extend(cps)
            cursor += n_rows

        if sr_model is not None:
            try:
                sr_model.cpu()
            except Exception:
                pass
            del sr_model
            comfy.model_management.soft_empty_cache()


        if crop_parts:
            crop_images = torch.cat(crop_parts, dim=0).contiguous()
        else:
            crop_images = torch.zeros(1, res, res, 3)
        n_sampled = sum(1 for st in subtracks if not st["skip"])

        mask_rows = _fill_mask_rows(subtracks, cursor, res)
        multi_track_flag = tracks is not None
        src_tracks = None
        tracks = None

        meta = {"T": int(T), "LH": int(LH), "LW": int(LW),  
                "W": int(W), "H": int(H), "res": res, "n_hit": int(n_hit),
                "n_frames": F_expect, "n_subtracks": len(subtracks), "n_sampled": n_sampled,
                "n_identities": len({st.get("track_id", 0) for st in subtracks}),
                "face_tracking": face_tracking, "gap_tol": int(gap_tol),
                "scale_split": float(scale_split), "skip_ratio": float(skip_ratio),
                "yolo_threshold": float(yolo_threshold), "sec_threshold": float(sec_threshold),
                "shot_active": bool(shot_active), "shot_threshold": float(shot_threshold),
                "flash_attn": bool(use_flash_attn), "sec_memory": int(mllm_memory_size),
                "n_shots": len(shots), "shot_cuts": [int(c) for c in shot_cuts],
                "shots": [[int(s), int(e)] for (s, e) in shots],
                "source": "images" if use_images else "latent"}
        pack = _build_pack(a_lat, subtracks, cursor, multi_track_flag, meta, fps_eff)
        shot_info = {"shots": [[int(s), int(e)] for s, e in shots],
                     "shot_cuts": [int(c) for c in shot_cuts],
                     "n_shots": len(shots), "fps": float(fps_eff)}

       
        skip_str = [(st.get("track_id", 0), st["f0"], st["f1"]) for st in subtracks if st.get("skip")]
        if skip_str:
            h3ff.warn(f"[H3-FaceCut] skipped subtracks (face >= {res * float(skip_ratio):.0f}px): {skip_str}\n"
                      f"[H3-FaceCut] 跳过重采样的子轨 (脸 >= {res * float(skip_ratio):.0f}px): {skip_str}")
    
        covered = sorted((int(st["f0"]), int(st["f1"])) for st in subtracks if not st.get("skip"))
        gaps, cur = [], 0
        for s, e in covered:
            if s > cur:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < F_expect:
            gaps.append((cur, F_expect))
        if gaps:
            h3ff.warn(f"[H3-FaceCut] UNCOVERED frame ranges (passthrough, no resample): {gaps}\n"
                      f"[H3-FaceCut] 未覆盖帧区间 (原样透传, 不重采样): {gaps}")
    
        info_str = [(st.get("track_id", 0), st["f0"], st["f1"], st["S"]) for st in subtracks if not st["skip"]]
        
        mz = float(mask_rows.max()) if mask_rows.numel() else 0.0
        print("\033[35m" + f"[H3-FaceCut] {len(subtracks)} subtracks ({n_sampled} sampled, "
              f"masks_max={mz:.2f}): (tid,f0,f1,S)={info_str}, "
              f"crop_images {tuple(crop_images.shape)}, masks {tuple(mask_rows.shape)}, "
              f"shots={len(shots)}\n"
              f"[H3-FaceCut] 共 {len(subtracks)} 条子轨 ({n_sampled} 条待采样, mask最大值={mz:.2f}): "
              f"(身份,帧起,帧止,S)={info_str}, 裁剪输出 {tuple(crop_images.shape)}, "
              f"mask输出 {tuple(mask_rows.shape)}, 镜头 {len(shots)} 个\033[0m")
        if face_tracking == "multi_sec" and mz <= 0.0:
            h3ff.warn("[H3-FaceCut] WARNING: masks output all zero — SeC masks missing (check SeC logs above)\n"
                      "[H3-FaceCut] 警告: mask 输出全零 — SeC mask 缺失 (检查上方 SeC 日志)")

        crop_images = crop_images.detach().to(torch.float16).float().contiguous()

        if cache_dir:
            latent_cache.save_blob_async(
                cache_dir, blob,
                {"crop_images": crop_images.detach().to(torch.float16).cpu().contiguous(),
                 "subtracks": subtracks,
                 "n_crop_rows": int(cursor),
                 "multi_track": bool(multi_track_flag),
                 "meta": meta},
                fp)
            h3ff.vlog(f"[H3-FaceCut] cache save submitted (async)\n[H3-FaceCut] 缓存保存已提交 (异步)")

        return io.NodeOutput(crop_images.contiguous(), pack, mask_rows.contiguous(), shot_info)

