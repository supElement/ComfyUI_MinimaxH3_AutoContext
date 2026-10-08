"""h3_face_cut.py — 修脸第 1 步: 分镜优先的检测与稳定裁剪

当前版本要点:
- 分镜优先 (shot-aware): 官方 PySceneDetect 管线 (临时视频 + open_video +
  SceneManager), 依赖缺失自动降级为仅上游分段隔离。镜头单元只按真实切点划分,
  上游生成接缝不充当镜头边界 (接缝是生成账本, 内容跨接缝连续)。
- 身份追踪: sec_model=None → 单脸模式; 选权重 → 内置 SeC-4B 逐镜头单元独立
  追踪多身份, 相邻单元单人时自动续接同一 id。YOLO 提供逐帧实测尺寸信号。
- 逐帧平滑窗口 (S_t): 脸尺寸序列先经差分域 Hampel 去脉冲 (_smooth_size_gaps),
  再经零相位高斯 去密集抖动 (_gauss1d, 对称核 → 推拉趋势零滞后);
  S_t = 平滑后脸尺寸 ×(1+余量) 逐帧直通 → 钳制 [_MIN_WIN, min(W,H)]。
  画布内脸占比恒 ≈1/(1+余量), 推拉镜头下不胀缩; 窗口中心同样
  中值平滑 + 零相位高斯, 贴回后同一身份无直切感。
- 分段 = 仅镜头切点 + skip 边界 (脸 >= res×skip_ratio 跳过重采样),
  不产生几何接缝 (逐帧窗口跨段连续), 镜头未切换不切段。
- GPU 批量化: _sample_windows 整批上卡 grid_sample; 无 SR 路径 bicubic+
  antialias; SR 路径按尺寸桶分批, OOM 自动减半重试。输出 crop_images 统一
  fp16 量化 — 新鲜输出与缓存命中位精确一致, 下游 rows_hash 不漂移。
- pre_blur (默认 0=关): SR/bicubic 放大前对裁剪窗口施加高斯预模糊, 压制源噪声
  与插值锯齿, 提高小脸 SR 稳定性。已入缓存键。
- 全帧保留: 不按 17n+5 掐尾 — 网格约束由 Face_Resample 编码期补齐。
- 身份锚定: 逐子轨按 (身份可信分 + 清晰度) 打分, 逐身份取全局最优行;
  赢家子轨另构建"干净参考帧" (源帧按锚定窗口重采样, 未经 pre_blur/SR)。
  输出 identity_refs 供用户在 Resample 前预检 — 参考错 = 修脸身份错。
- 缓存: 指纹含 win_v / 模型文件实体 (size+mtime) / VAE / blocks 等; 升级后旧
  缓存自动失效一次, 无需手动 clear_cache。
- 日志收编: 常规运行只保留入口/出口/警告; 逐子轨细节走 h3ff.vlog
  (h3_facefix.py 顶部 _VERBOSE=True 打开)。
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

# ---- 窗口下限 (px) ----
# 裁剪窗口 S 的最小值 (自动 16 对齐)。远景小脸时 S 会被钳在此地板上 —
# 调小 → 远景占比更高 (脸在画布上更大)
_MIN_WIN = 48

_SZGAP_WIN = 2     # Hampel 邻域半径 (差分个数; ±2 → 邻域窗5)
_SZGAP_K = 6.0     # MAD 倍数 (阈值主项; 调小更激进, 调大更保守)
_SZGAP_R = 0.30    # 相对下限 (邻域中值差分的30%, 防 MAD≈0 时误杀同向真实波动)
_TRJ_SIGMA = 3.0   # 高斯 σ (帧)

# ---- SR 放大的尺寸桶宽 ----
_SR_BUCKET = 16

# ---- SR 最大遍数 (1 = 单次) ----
_SR_MAX_PASSES = 1

# ---- 裁剪窗口边长 >= 此值时跳过 SR, 直接 bicubic 到 res (px) ----
# 设为 0 = 关闭 (所有窗口都走 SR)
_SR_SKIP_WIN = 192

def _win_floor():
    """窗口下限 (16 对齐后的 _MIN_WIN) — 唯一来源, 供窗口规划与 DP 估计共用。"""
    return ((int(_MIN_WIN) + 15) // 16) * 16


# ================= 内置常量=================
_GAP_TOL = 24        # 检测缺失多少帧内视为同一次出现 (绝不跨镜头)
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
    """框中心中值平滑 (窗口5, 端点收缩) — 中心去脉冲; 密集抖动由调用方零相位高斯处理。"""
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

def _sr_pass(sr_model, x, batch_size=4, pbar=None):
    """单次放大: [N,H,W,3] float 0..1 → [N,H*s,W*s,3]。分批执行, OOM 自动减半批。
    pbar: 外层共享进度条, 每完成一批推进实际帧数 (不再自建任何条/打印)。"""
    dev = comfy.model_management.get_torch_device()
    sr_model.to(dev)
    out, n, bs, i = [], int(x.shape[0]), max(1, min(int(batch_size), int(x.shape[0]))), 0
    while i < n:
        try:
            b = x[i:i + bs].movedim(-1, 1).to(dev)
            with torch.no_grad():
                o = sr_model(b)
            out.append(o.movedim(1, -1).clamp(0.0, 1.0).cpu())
            step = int(b.shape[0])          
            i += step
            del b, o
            if pbar is not None:
                pbar.update(step)
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

def _smooth_size_gaps(sizes):
    """差分域 Hampel 去检测噪声 (v25): 相邻帧差分与邻域运动模式显著矛盾的
    (脉冲型检测噪声, 如 ...71→58 的尾部 -13) 替换为邻域中值差分后累加还原。
    真实运动的差分连续同号, 阈值内原样保留 → 推拉趋势零滞后零失真。
    thr = k×MAD + 30%×|邻域中值差分|; 端点用单侧邻域。返回平滑后序列 (等长)。"""
    n = len(sizes)
    out0 = [float(s) for s in sizes]
    if n < 3:
        return out0
    d = [out0[i + 1] - out0[i] for i in range(n - 1)]
    m = len(d)
    d2 = list(d)
    n_fix, max_dev = 0, 0.0
    for i in range(m):
        lo, hi = max(0, i - _SZGAP_WIN), min(m, i + _SZGAP_WIN + 1)
        nb = d[lo:i] + d[i + 1:hi]
        if not nb:
            continue
        med = float(np.median(nb))
        mad = float(np.median([abs(x - med) for x in nb]))
        thr = _SZGAP_K * mad + _SZGAP_R * abs(med)
        dev = abs(d[i] - med)
        if dev > thr:
            d2[i] = med
            n_fix += 1
            max_dev = max(max_dev, dev)
    if n_fix == 0:
        return out0
    h3ff.vlog(f"[H3-FaceCut] size gaps: {n_fix}/{m} outlier gap(s) fixed "
              f"(max dev {max_dev:.1f}px) — Hampel on frame diffs\n"
              f"[H3-FaceCut] 尺寸差分去噪: 修复 {n_fix}/{m} 个离群差分 "
              f"(最大偏离 {max_dev:.1f}px) — 相邻帧差分 Hampel")
    out = [out0[0]]
    for i in range(m):
        out.append(out[-1] + d2[i])
    return out


def _gauss1d(arr, sigma):
    """零相位高斯 (replicate 端点): 对称核 → 线性趋势零滞后零失真, 只压高频残噪。"""
    a = np.asarray(arr, dtype=np.float64)
    n = len(a)
    if n < 3 or sigma <= 0:
        return a
    r = max(1, int(round(3.0 * float(sigma))))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * float(sigma) ** 2))
    k = k / k.sum()
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    return torch.nn.functional.conv1d(
        torch.nn.functional.pad(t, (r, r), mode="replicate"), k[None, None])[0, 0].double().numpy()


def _smooth_window_sizes(face_sizes, expand_f, W, H):
    """逐帧窗口边长序列: S_t = (已平滑的)脸尺寸 × (1+余量) — 逐帧正比直通。
    face_sizes 由调用方先完成两级平滑: 差分域 Hampel 去脉冲 (_smooth_size_gaps)
    + 零相位高斯去密集抖动 (_gauss1d, 对称核 → 真实推拉趋势零滞后) —
    测量噪声不进入 S, 真实推拉完整保留。
    面部在画布中的呈现占比恒 ≈1/(1+expand), 与脸绝对尺寸无关, 推拉镜头下不胀缩。
    钳制 [_win_floor(), min(W,H)]: 近景脸大到窗口顶满画面时按画面上限收缩,
    与后期软件固定比例裁剪在画面极限处的行为一致。"""
    fl = _win_floor()
    cap = max(fl, min(int(W), int(H)))
    return [max(fl, min(cap, int(round(float(s) * (1.0 + float(expand_f)))))) for s in face_sizes]


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


def _build_subtracks_for_track(boxes_seq, frames, H, W, res, expand_f,
                               skip_ratio, gap_tol, tid, masks_full=None,
                               size_override=None, shot_id=None, sr_model=None, sr_batch=4, pre_blur=0.0, pbar=None):
    """ 分工: SeC-4B 身份/mask, YOLO 逐帧实测尺寸; 本函数只做应用层几何:
    分段 (镜头切点 + skip 边界) → 几何平滑 (中心: 中值窗5→零相位高斯;
    尺寸: 差分Hampel→零相位高斯) → S_t 直通 → 采样 → 账本。
    检测帧全部分配, 不丢弃。
    镜头单元约束: appearance 合并与框插值禁止跨镜头单元 (切镜头处强制断开, 防跳变)。
    无 DP 尺寸切段: 镜头未切换时整段一条连续 (S_t, center_t) 序列。
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
            # ---- 零相位高斯 (中心/尺寸轨迹去密集抖动, 对称核零滞后) ----
            _cxs = _gauss1d([(b[0] + b[2]) * 0.5 for b in raw_boxes], _TRJ_SIGMA)
            _cys = _gauss1d([(b[1] + b[3]) * 0.5 for b in raw_boxes], _TRJ_SIGMA)
            _hfs = [_size(b) * 0.5 for b in raw_boxes]
            raw_boxes = [[float(cx) - h, float(cy) - h, float(cx) + h, float(cy) + h]
                         for cx, cy, h in zip(_cxs, _cys, _hfs)]
            # 尺寸 (缩放): Hampel 去脉冲 零相位高斯去密集抖动
            raw_sz = _gauss1d(
                _smooth_size_gaps([_fs(i) for i in range(ra, rb)]), _TRJ_SIGMA)
            S_seq = _smooth_window_sizes(raw_sz, expand_f, W, H)

            if (rb - ra) < 5:
                pieces.append((ra, rb, S_seq, raw_boxes, raw_sz))
                continue
            _thr = res * float(skip_ratio)
            _cur, _i = ra, ra
            while _i < rb:
                if raw_sz[_i - ra] >= _thr:
                    _j = _i
                    while _j < rb and raw_sz[_j - ra] >= _thr:
                        _j += 1
                    if _j - _i >= 5:  # 连续 >=5 帧才整段跳过; 零碎大脸帧归入采样
                        if _cur < _i:
                            pieces.append((_cur, _i, S_seq[_cur - ra:_i - ra], raw_boxes[_cur - ra:_i - ra], raw_sz[_cur - ra:_i - ra]))
                        pieces.append((_i, _j, S_seq[_i - ra:_j - ra], raw_boxes[_i - ra:_j - ra], raw_sz[_i - ra:_j - ra]))
                        _cur = _j
                    _i = _j
                else:
                    _i += 1
            if _cur < rb:
                pieces.append((_cur, rb, S_seq[_cur - ra:rb - ra], raw_boxes[_cur - ra:rb - ra], raw_sz[_cur - ra:rb - ra]))

    if not pieces:
        return subtracks, crop_parts, 0

    for si, (pa, pb, S_seq, boxes, sz_seq) in enumerate(pieces):
        idxs = list(range(pa, pb))
        med = float(np.median(sz_seq))
        if med <= 0:
            med = float(np.median([_size(b) for b in boxes]))
        f0_all, f1_all = pa, pb

        if min(sz_seq) >= res * float(skip_ratio):

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

        if float(pre_blur) > 0.0:
            crops = _blur_batch(crops, float(pre_blur))
        have = [i for i, m in enumerate(win_masks) if m is not None]

        proc = [None] * len(crops)
        if sr_model is not None:
            TOL = max(8, int(_SR_BUCKET))
            sc = max(1, int(getattr(sr_model, "scale", 4) or 4))   # SR 模型放大倍数
            buckets = {}
            for _ci, _c in enumerate(crops):
                _K = -(-int(_c.shape[1]) // TOL) * TOL
                buckets.setdefault(_K, []).append(_ci)
            h3ff.vlog(f"[H3-FaceCut] id{tid} sub{si+1}: SR buckets={len(buckets)} "
                      f"(TOL={TOL}px, scale={sc}, skip_win={_SR_SKIP_WIN}, batch<={int(sr_batch)})")
            for _K in sorted(buckets):
                _idxs_g = buckets[_K]
                if _SR_SKIP_WIN > 0 and _K >= _SR_SKIP_WIN:
                    _resized = _gpu_resize_batch([crops[_i] for _i in _idxs_g], res)
                    for _j, _i in enumerate(_idxs_g):
                        proc[_i] = _resized[_j]
                    if pbar is not None:
                        pbar.update(len(_idxs_g))
                    continue
                frames_p = []
                for _i in _idxs_g:
                    c_ = crops[_i]
                    p = _K - int(c_.shape[1])
                    frames_p.append(
                        torch.nn.functional.pad(c_, (0, p, 0, p), mode="replicate") if p > 0 else c_)
                x = torch.stack(frames_p, dim=0).permute(0, 2, 3, 1) / 255.0
                n_pass = 0
                while int(x.shape[1]) < res and n_pass < _SR_MAX_PASSES:
                    x = _sr_pass(sr_model, x, batch_size=sr_batch, pbar=pbar)
                    n_pass += 1
                ratio = float(int(x.shape[1])) / float(_K)
                _dev = comfy.model_management.get_torch_device()
                for _j, _i in enumerate(_idxs_g):
                    S_i = int(crops[_i].shape[1])
                    S2 = int(round(S_i * ratio))
                    f_i = x[_j, :S2, :S2].contiguous()
                    if (int(f_i.shape[0]), int(f_i.shape[1])) != (res, res):
                        ft = f_i.permute(2, 0, 1).unsqueeze(0).to(_dev)
                        ft = torch.nn.functional.interpolate(
                            ft, size=(res, res), mode="bicubic", antialias=True)
                        f_i = ft.clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu()
                    proc[_i] = f_i

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

class _DualPbar:
    """同一份进度, 两处显示: 节点上的 ComfyUI 进度条 + 后台控制台 tqdm 条。
    接口只有 update(n) — _build_subtracks_for_track 无需感知。"""
    def __init__(self, total, desc="FaceCut crop"):
        self._tq = None
        self._cp = None
        try:
            from tqdm import tqdm
            self._tq = tqdm(total=int(total), desc=desc, unit="f", dynamic_ncols=True)
        except Exception:
            pass
        if h3ff._HAS_COMFY:
            try:
                self._cp = comfy.utils.ProgressBar(int(total))
            except Exception:
                pass

    def update(self, n):
        n = int(n)
        if self._tq is not None:
            self._tq.update(n)
        if self._cp is not None:
            self._cp.update(n)

    def close(self):
        if self._tq is not None:
            self._tq.close()

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

# ==================================================================
# ====== 身份锚定帧定案 (FaceCut 侧, 经 face_pack 传给 Resample) ======
# ==================================================================

_ANCHOR_TOPK = 48          # 候选抽样帧数上限 (等距覆盖整条子轨)
_ANCHOR_DET_IOU = 0.45     # SeC框 ↔ YOLO检出框 认定同一张脸的重叠率 (交集/较小面积)
_ANCHOR_XCL_IOU = 0.30     # 身份互斥阈值: 与其他身份框重叠低于此值 → 确为分开的两人
_ANCHOR_CROSS_IOU = 0.50   # 与其他身份框重叠高于此值 → 交叉换位嫌疑 (强惩罚)
_ANCHOR_COVER_LO = 0.04    # SeC mask 前景占比合理下限 (窗口里确实有脸)
_ANCHOR_COVER_HI = 0.90    # 上限 (mask 糊满整个窗口 = 追踪发散, 不可信)
_ANCHOR_CTR_TOL = 0.20     # mask 质心偏离窗口中心的比例容限

def _laplacian_sharpness(rows, work=256):
    """[N,H,W,3] float 0..1 → [N] 清晰度 (灰度拉普拉斯方差; 统一 area 降采样到 work² 保证成本有界)。"""
    g = rows.float().mean(dim=-1)
    if max(int(g.shape[-2]), int(g.shape[-1])) > int(work):
        g = torch.nn.functional.interpolate(
            g.unsqueeze(1), size=(int(work), int(work)), mode="area").squeeze(1)
    lap = (g[..., :-2, :-2] + g[..., :-2, 2:] + g[..., 2:, :-2] + g[..., 2:, 2:]
           - 4.0 * g[..., 1:-1, 1:-1])
    return lap.var(dim=(1, 2))

def _pick_identity_anchor(rows, f0, my_idx, all_tracks_boxes, per_frame, masks=None, ref_res=512):
    """为一条非 skip 子轨挑选"身份可信 + 清晰"的锚定行 → (子轨内局部行号, 旗标 dict)。

    rows:             [K,res,res,3] 本子轨裁剪行 (float 0..1)
    f0:               子轨起始全局帧号 (boxes/per_frame 按全局帧对齐)
    my_idx:           本身份在 all_tracks_boxes 中的下标
    all_tracks_boxes: 全部身份的每帧 SeC 框 ([F] bbox|None 的列表的列表)
    per_frame:        YOLO 每帧检出
    masks:            [K,res,res] SeC mask (float 0..1) 或 None
    """
    K = int(rows.shape[0])
    flags = {"score": 0.0, "sharp": 0.0, "n_cross": 0, "all_cross": False}
    if K <= 0:
        return None, flags
    my_boxes = (all_tracks_boxes[my_idx]
                if (all_tracks_boxes is not None and 0 <= int(my_idx) < len(all_tracks_boxes)) else None)
    n_s = min(K, int(_ANCHOR_TOPK))
    idxs = sorted(set(int(round(i * (K - 1) / max(1, n_s - 1))) for i in range(n_s)))
    sharp = _laplacian_sharpness(rows[idxs])
    sharp_n = (sharp - sharp.min()) / (sharp.max() - sharp.min() + 1e-8)
    score = torch.zeros(len(idxs))
    n_det_frames = 0
    for ii, lf in enumerate(idxs):
        f = int(f0) + lf
        b = my_boxes[f] if (my_boxes is not None and 0 <= f < len(my_boxes)) else None
        dets = per_frame[f] if (per_frame is not None and 0 <= f < len(per_frame)) else []
        if b is None:
            score[ii] -= 1.0          
            if not dets:
                score[ii] -= 1.0
            continue
        n_det_frames += 1
        score[ii] += 2.0              
        if any(h3ff._overlap_ratio(b[:4], d[:4]) >= _ANCHOR_DET_IOU for d in dets):
            score[ii] += 1.5          
        else:
            score[ii] -= 0.5
        _fsz = max(float(b[2]) - float(b[0]), float(b[3]) - float(b[1]))
        score[ii] += 0.5 * min(1.0, _fsz / max(1.0, 0.8 * float(ref_res)))
        if not dets:
            score[ii] -= 1.0

        others = []
        for ti, tb in enumerate(all_tracks_boxes or []):
            if ti == int(my_idx) or tb is None:
                continue
            ob = tb[f] if 0 <= f < len(tb) else None
            if ob is not None:
                others.append(h3ff._overlap_ratio(b[:4], ob[:4]))
        if others:
            if len(dets) >= 2 and all(r < _ANCHOR_XCL_IOU for r in others):
                score[ii] += 1.0      
            if any(r >= _ANCHOR_CROSS_IOU for r in others):
                score[ii] -= 2.0     
                flags["n_cross"] += 1
        if masks is not None and lf < int(masks.shape[0]):
            m = masks[lf].float()
            cov = float(m.mean())
            if _ANCHOR_COVER_LO <= cov <= _ANCHOR_COVER_HI:
                score[ii] += 0.5
                nz = torch.nonzero(m > 0.5)
                if int(nz.shape[0]):
                    cy_ = float(nz[:, 0].float().mean()) / max(1, int(m.shape[0])) - 0.5
                    cx_ = float(nz[:, 1].float().mean()) / max(1, int(m.shape[1])) - 0.5
                    if (cx_ * cx_ + cy_ * cy_) ** 0.5 < float(_ANCHOR_CTR_TOL):
                        score[ii] += 0.25   
    if n_det_frames and flags["n_cross"] >= n_det_frames:
        flags["all_cross"] = True   
    top = float(score.max())
    cand = [i for i in range(len(idxs)) if float(score[i]) >= top - 0.01]
    best = max(cand, key=lambda i: float(sharp_n[i]))   
    flags["score"] = top
    flags["sharp"] = float(sharp_n[best])     
    flags["sharp_raw"] = float(sharp[best])   
    return int(idxs[best]), flags

def _build_identity_refs(subtracks, crop_images, res):
    """逐身份汇总锚定参考 → 预览张量 [n_id, res, res, 3] (行序=身份id 升序)。
    优先 subtrack['ref_image'] (仅逐身份赢家子轨携带: 源帧重采样原画面, 未经 pre_blur/SR);
    无 ref_image 时回退 crop_images[ref_row] (加工后的行); 无 ref_row (旧缓存) 才回退纯清晰度。
    选择键 (ref_score, ref_sharp) 字典序取最大, 与 Resample 消费逻辑一致 — 预览即所得。
    返回 (tensor | None, fell_back): fell_back=True 表示至少一个身份最终用的是加工行/纯清晰度。
    """
    best = {}
    n_rows = int(crop_images.shape[0])
    for st in subtracks:
        if st.get("skip") or st.get("crop_off") is None:
            continue
        tid = int(st.get("track_id", 0))
        rr = st.get("ref_row")
        if rr is not None:
            rr = int(rr)
            if not (0 <= rr < n_rows):
                continue
            _sh = st.get("ref_sharp")
            if _sh is None:
                _sh = float(_laplacian_sharpness(crop_images[rr:rr + 1].float())[0])
            key = (float(st.get("ref_score", 0.0)), float(_sh))
            img = st.get("ref_image")
            clean = img is not None
            if not clean:
                img = crop_images[rr].float()
        else:
            off, K = int(st["crop_off"]), int(st["f1"]) - int(st["f0"])
            rows = crop_images[off:off + K].float()
            n_s = min(K, int(_ANCHOR_TOPK))
            idxs = sorted(set(int(round(i * (K - 1) / max(1, n_s - 1))) for i in range(n_s)))
            sharp = _laplacian_sharpness(rows[idxs])
            bi = int(torch.argmax(sharp).item())
            rr = off + int(idxs[bi])
            img = crop_images[rr].float()
            key = (-1.0, float(sharp[bi]))
            clean = False
        cur = best.get(tid)
        if cur is None or key > cur[0]:
            best[tid] = (key, rr, img, clean)
    if not best:
        return None, False
    ordered = [best[tid] for tid in sorted(best)]
    fell_back = any(not t[3] for t in ordered)
    return torch.stack([t[2] for t in ordered], dim=0).contiguous(), fell_back


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
                               
                io.Image.Output(display_name="identity_refs", tooltip="One identity-verified anchor crop per track (row order = track id). "
                     "PRE-CHECK this BEFORE running Face_Resample — wrong ref = wrong identity repair\n"
                     "每个身份一张已验证锚定参考 (行序=身份id)。请在运行 Face_Resample 之前用它预检 — 参考错 = 修脸身份错"),
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
                      "gap_tol": int(gap_tol),
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
                _ir, _ir_fb = _build_identity_refs(subtracks, crop_images, res)
                if _ir is None:
                    _ir = torch.zeros(1, res, res, 3)
                if _ir_fb:
                    h3ff.warn("[H3-FaceCut] some identity refs fell back to PROCESSED crop rows "
                              "(old cache without ref_row/ref_image, or clean-ref build failed) — "
                              "rerun Face_Cut once with clear_cache for original-frame refs\n"
                              "[H3-FaceCut] 部分身份参考回退为加工后的裁剪行 (旧缓存无 ref_row/ref_image, "
                              "或干净参考构建失败) — 勾选一次 clear_cache 重跑 Face_Cut 可获得原画面参考")
                
                print("\033[33m" + f"[H3-FaceCut] cache hit: {len(subtracks)} subtracks, "
                      f"crop {tuple(crop_images.shape)}, masks_max={mz:.2f} — shot/decode/detect/SeC skipped\n"
                      f"[H3-FaceCut] 缓存命中: {len(subtracks)} 条子轨, 裁剪 {tuple(crop_images.shape)}, "
                      f"mask最大值={mz:.2f} — 已跳过 分镜/解码/检测/SeC" + "\033[0m")
                return io.NodeOutput(crop_images.contiguous(), pack, mask_rows.contiguous(), shot_info, _ir.contiguous())

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
                _pt = "single pass" if _SR_MAX_PASSES == 1 else f"max {_SR_MAX_PASSES} pass(es)"
                _pt_cn = "单次 SR" if _SR_MAX_PASSES == 1 else f"最多 {_SR_MAX_PASSES} 次 SR"
                h3ff.log(f"[H3-FaceCut] crop upscale: {sr_name} (SR {_pt} -> {res}px)\n"
                         f"[H3-FaceCut] 裁剪放大: {sr_name} ({_pt_cn} → {res}px)")
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
        crop_pbar = _DualPbar(max(1, _total_work), "FaceCut crop")

        for tid, boxes_seq, masks_full, sz_override in src_tracks:
            sts, cps, n_rows = _build_subtracks_for_track(
                boxes_seq, frames, H, W, res, expand_f, float(skip_ratio),
                int(gap_tol), tid, masks_full=masks_full,
                size_override=(sz_override if face_tracking == "multi_sec" else None),
                shot_id=shot_id, sr_model=sr_model, sr_batch=sr_batch, pbar=crop_pbar, pre_blur=float(pre_blur))
                
            for st in sts:
                if not st["skip"]:
                    st["crop_off"] = cursor + int(st["crop_off"])  
            subtracks.extend(sts)
            crop_parts.extend(cps)
            cursor += n_rows
        crop_pbar.close()

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
        # ---- 身份锚定帧定案 (v21.1): 逐子轨评分走 vlog; 逐身份取 (score, 原始清晰度) 全局最优 ----
        _boxes_by_id = {int(t[0]): i for i, t in enumerate(src_tracks or [])}
        _track_boxes_list = [t[1] for t in (src_tracks or [])]
        for st in subtracks:
            if st.get("skip") or st.get("crop_off") is None:
                continue
            _off, _K = int(st["crop_off"]), int(st["f1"]) - int(st["f0"])
            _tid = int(st.get("track_id", 0))
            _masks = st.get("masks")
            if _masks is not None:
                _masks = _masks.to(torch.float32).mul(1.0 / 255.0)
            _lf, _fl = _pick_identity_anchor(
                crop_images[_off:_off + _K], int(st["f0"]),
                _boxes_by_id.get(_tid, -1), _track_boxes_list, per_frame,
                masks=_masks, ref_res=res)
            if _lf is None:
                continue
            st["ref_row"] = int(_off + _lf)
            st["ref_local"] = int(_lf)
            st["ref_score"] = float(_fl.get("score", 0.0))
            st["ref_sharp"] = float(_fl.get("sharp_raw", 0.0))
            st["ref_sharp_n"] = float(_fl.get("sharp", 0.0))   
            h3ff.vlog(f"[H3-FaceCut] anchor cand: id{_tid} sub[{st['f0']},{st['f1']}) → "
                      f"row {_off + _lf} (local {_lf}) score={_fl.get('score', 0.0):.1f} "
                      f"sharp={_fl.get('sharp_raw', 0.0):.4f}\n"
                      f"[H3-FaceCut] 锚定候选: 身份{_tid} 子轨[{st['f0']},{st['f1']}) → "
                      f"全局第 {_off + _lf} 行 得分={_fl.get('score', 0.0):.1f} "
                      f"清晰度={_fl.get('sharp_raw', 0.0):.4f}")

            if _fl.get("all_cross"):
                h3ff.warn(f"[H3-FaceCut] anchor id{_tid} [{st['f0']},{st['f1']}): EVERY sampled "
                          f"detection frame overlaps another identity — SeC tracking likely "
                          f"swapped identities; anchor is least-bad, check sec_threshold/YOLO\n"
                          f"[H3-FaceCut] 身份{_tid} [{st['f0']},{st['f1']}): 所有抽样检测帧均与其他"
                          f"身份高重叠 — SeC 追踪疑似身份互换, 锚定仅为最优可用帧, "
                          f"请检查 sec_threshold/YOLO 检测")
        # 逐身份汇总 (每身份仅 1 行日志): 同分不再"先到先得", 用原始清晰度跨子轨裁决
        _best_by_track = {}
        for st in subtracks:
            if st.get("skip") or st.get("ref_row") is None:
                continue
            _tid = int(st.get("track_id", 0))
            _key = (float(st.get("ref_score", 0.0)), float(st.get("ref_sharp", 0.0)))
            _cur = _best_by_track.get(_tid)
            if _cur is None or _key > _cur[0]:
                _best_by_track[_tid] = (_key, st)
        for _tid, (_k, _st) in sorted(_best_by_track.items()):
            h3ff.log(f"[H3-FaceCut] anchor id{_tid}: global row {_st['ref_row']} "
                     f"(sub[{_st['f0']},{_st['f1']}) local {_st['ref_local']}) "
                     f"score={_k[0]:.1f} sharp={_k[1]:.4f} (norm {_st.get('ref_sharp_n', 0.0):.2f})\n"
                     f"[H3-FaceCut] 身份锚: 身份{_tid} → 全局第 {_st['ref_row']} 行 "
                     f"(子轨[{_st['f0']},{_st['f1']}) 内第 {_st['ref_local']} 行) "
                     f"得分={_k[0]:.1f} 清晰度={_k[1]:.4f} (子轨内相对 {(_st.get('ref_sharp_n', 0.0)) * 100:.0f}%)")

        # ---- 干净参考帧 (仅逐身份赢家, 每身份 1 张): 源帧重采样原画面, 未经 pre_blur/SR ----
        for _tid, (_k, _st) in sorted(_best_by_track.items()):
            try:
                _lf = int(_st["ref_local"])
                _gf = int(_st["f0"]) + _lf
                _centers_l = _st.get("centers") or []
                _slist_l = _st.get("S_list") or []
                if _gf < int(frames.shape[0]) and _lf < len(_centers_l) \
                        and _lf < len(_slist_l) and int(_slist_l[_lf]) > 0:
                    _c_win, _ = _sample_windows(
                        frames, None, [_gf],
                        [[float(_centers_l[_lf][0]), float(_centers_l[_lf][1])]],
                        [int(_slist_l[_lf])])
                    _clean = _gpu_resize_batch(_c_win, res)[0]   # [res,res,3] 0..1, 无模糊无SR
                    _st["ref_image"] = _clean.to(torch.float16).contiguous()
                else:
                    h3ff.vlog(f"[H3-FaceCut] clean ref skipped: id{_tid} anchor geometry missing\n"
                              f"[H3-FaceCut] 干净参考跳过: 身份{_tid} 锚定几何缺失")
            except Exception as _e:
                h3ff.vlog(f"[H3-FaceCut] clean ref_image build failed id{_tid} ({_e}) — "
                          f"ref falls back to processed crop row\n"
                          f"[H3-FaceCut] 身份{_tid} 干净参考帧构建失败 ({_e}) — 参考回退为加工后的裁剪行")
        mask_rows = _fill_mask_rows(subtracks, cursor, res)

        multi_track_flag = tracks is not None
        src_tracks = None
        tracks = None

        meta = {"T": int(T), "LH": int(LH), "LW": int(LW),  
                "W": int(W), "H": int(H), "res": res, "n_hit": int(n_hit),
                "n_frames": F_expect, "n_subtracks": len(subtracks), "n_sampled": n_sampled,
                "n_identities": len({st.get("track_id", 0) for st in subtracks}),
                "face_tracking": face_tracking, "gap_tol": int(gap_tol),
                "skip_ratio": float(skip_ratio),
                "yolo_threshold": float(yolo_threshold), "sec_threshold": float(sec_threshold),
                "shot_active": bool(shot_active), "shot_threshold": float(shot_threshold),
                "flash_attn": bool(use_flash_attn), "sec_memory": int(mllm_memory_size),
                "n_shots": len(shots), "shot_cuts": [int(c) for c in shot_cuts],
                "shots": [[int(s), int(e)] for (s, e) in shots],
                "anchors": [[tid, st["f0"], st["f1"], st["ref_row"]]
                            for tid, (_k, st) in sorted(_best_by_track.items())],
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
        # ---- identity_refs 预览 (在量化后的 crop_images 上取行, 与 Resample 实际消费的行一致) ----
        identity_refs_out, _ = _build_identity_refs(subtracks, crop_images, res)
        if identity_refs_out is None:
            identity_refs_out = torch.zeros(1, res, res, 3)
        if cache_dir:
            latent_cache.save_blob_async(
                cache_dir, blob,
                {"crop_images": crop_images.detach().to(torch.float16).cpu().contiguous(),
                 "subtracks": subtracks, "n_crop_rows": int(cursor),
                 "multi_track": bool(multi_track_flag), "meta": meta}, fp)
        h3ff.vlog(f"[H3-FaceCut] cache save submitted (async)\n[H3-FaceCut] 缓存保存已提交 (异步)")
        return io.NodeOutput(crop_images.contiguous(), pack, mask_rows.contiguous(),
                             shot_info, identity_refs_out.contiguous())


