"""h3_face_blend.py — 修脸第 3 步: 缩放 + 像素贴回 (无 VAE)。

逐子轨把 Resample 画布行缩放回原帧窗口并像素级贴回。要点:
- 窗口几何: 支持 FaceCut 逐帧 S_list (窗口随脸缩放), 旧 pack 回退常数 S。
- motion_align (默认开): 中心模板 NCC 估计画布脸相对原帧的位置漂移并抵消,
  贴回运动完全继承原视频, 消除高 σ 块边界抖动; 软失败回退普通贴回。
- lowfreq_lock: 低频取原帧 / 高频取画布, 仅适用低 σ 细节锐化; 修崩坏脸保持关。
- use_sec_mask / masks 端口: 可选 SeC mask 贴回 (只贴人脸), 行与画布 1:1;
  alpha 来源优先级: 端口 > pack 内置 > 羽化框回退。
- 等尺寸 run 分块批处理, 自动 GPU (无 CUDA 回退 CPU), 纯 torch 无新增依赖。

"""
import numpy as np
import torch
import comfy.utils
from comfy_api.latest import io

try:
    from . import h3_facefix as h3ff
except ImportError:
    import h3_facefix as h3ff


def _mask_alpha_frame(m2d, S_fr, feather_px, device):
    """单帧 2D mask (uint8 0..255, 任意 [H,W]/[1,H,W]) → [S_fr,S_fr] float alpha (0..1)。
    feather_px=0 → 硬边; 否则腐蚀 feather/2 (过渡带内移, 防背景渗入) + blur σfeather/2。
    腐蚀掏空时回退原 mask (小脸保护)。"""
    t = m2d.detach().float().cpu()
    if t.dim() > 2:
        t = t.reshape(-1, t.shape[-2], t.shape[-1])[0]
    if int(t.shape[0]) != S_fr or int(t.shape[1]) != S_fr:
        t = torch.nn.functional.interpolate(
            t[None, None], size=(S_fr, S_fr), mode="bilinear", align_corners=False)[0, 0]
    a = t.numpy()
    f_px = max(0, int(feather_px))
    if f_px > 0:
        from scipy import ndimage
        er = max(1, f_px // 2)
        e = ndimage.binary_erosion(a > 0.5, iterations=er).astype(np.float32)
        a = e if e.max() > 0 else a
        a = ndimage.gaussian_filter(a, sigma=max(1.0, f_px / 2.0))
    else:
        a = (a > 0.5).astype(np.float32)
    return torch.from_numpy(np.clip(a, 0.0, 1.0)).to(device)


# ================= 金字塔低频锁定 (P0) =================
# mix = low(原帧) + high(画布): 低频来自原帧 (逐帧连续, 不抖), 高频来自重采样画布 (细节)。
# σ 随窗口大小缩放: 远景小窗 (S 小) σ 小, 近景大窗 σ 大 — 对所有脸型统一"锁定结构、
# 保留纹理"的语义, 无需用户调参。

_GAUSS_KERNEL_CACHE = {}


def _gauss_kernel_1d(sigma, device, dtype=torch.float32):
    key = (round(float(sigma), 3), str(device), dtype)
    k = _GAUSS_KERNEL_CACHE.get(key)
    if k is None:
        r = int(3.0 * float(sigma) + 0.5)
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
        k = torch.exp(-(x * x) / (2.0 * float(sigma) * float(sigma)))
        k = k / k.sum()
        if len(_GAUSS_KERNEL_CACHE) > 64:
            _GAUSS_KERNEL_CACHE.clear()
        _GAUSS_KERNEL_CACHE[key] = k
    return k.to(device=device, dtype=dtype)


def _blur_batch(x, sigma, device):
    """[N,3,S,S] 可分离高斯 (replicate 边界, 与像素域 blend 语义一致)。"""
    if sigma <= 0:
        return x
    k = _gauss_kernel_1d(sigma, device, x.dtype)
    r = k.numel() // 2
    C = int(x.shape[1])
    kw = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    xp = torch.nn.functional.pad(x, (r, r, 0, 0), mode="replicate")
    xp = torch.nn.functional.conv2d(xp, kw, groups=C)
    xp = torch.nn.functional.pad(xp, (0, 0, r, r), mode="replicate")
    return torch.nn.functional.conv2d(xp, kh, groups=C)


def _pyramid_mix(reg, can, sigma, device):
    """低频锁定混合: 返回 低频=reg / 高频=can 的合成 (钳回 0..1 防过冲)。"""
    low_r = _blur_batch(reg, sigma, device)
    low_c = _blur_batch(can, sigma, device)
    return (low_r + can - low_c).clamp(0.0, 1.0)


# ================= motion_align: 贴回位置修正 (v16) =================
# 目标: 把贴回的脸"粘"到原帧面部的真实位置上 — 贴回后的运动完全继承原视频,
# 消除重采样画布 (尤其高 σ / 块边界) 造成的贴回位置抖动。
# 方法: 每帧统一分辨率下做中心模板 NCC (FFT 批量互相关 + 亚像素抛物线),
# 峰值强度/显著性双门控 (借鉴 Smart_merge_images 的可靠性思想, 但为视频全序列
# 联合设计而非逐帧独立) + 时间中值平滑 + 失败帧线性插值, 修正量钳制 ±12.5% 窗口。

_MA_WORK = 192      # 估计分辨率 (所有窗口统一 area 降采样到此尺寸)
_MA_TMPL = 0.40     # 模板边长占窗口比例 (≈脸区域 — 刻意小于窗口: 画布背景基本
                    #   原地, 模板混入背景会把相关性"锚"在背景上, 测不出脸的漂移)
_MA_SEARCH = 0.15   # 搜索半径占窗口比例
_MA_PEAK = 0.22     # NCC 峰值强度下限 (低于 → 该帧不可靠)
_MA_PROM = 2.5      # 峰值显著性下限 (峰值 / 搜索面中值绝对值)
_MA_MED = 9         # Hampel 离点检测窗宽
_MA_MAXSH = 0.125   # 单帧修正量上限 (占窗口边长比例)
_MA_GSIGMA = 2.5    # 零相位高斯 σ (帧)
_MA_SEC = 1.15      # 次峰抑制比 (峰值 / ±3邻域外最大值)

def _ma_interp1d(idx, vi, vals):
    """1D 线性插值 (np.interp 语义: 越界取端值)。idx [K] 查询; vi [m] 升序; vals [m]。"""
    pos = torch.searchsorted(vi, idx).clamp(1, max(1, int(vi.numel()) - 1))
    v0, v1 = vi[pos - 1].float(), vi[pos].float()
    w = ((idx - v0) / (v1 - v0 + 1e-12)).clamp(0.0, 1.0)
    return vals[pos - 1] * (1.0 - w) + vals[pos] * w


def _grad2d(x):
    """[m,W,W] 中心差分梯度幅值 — 平坦区权重归零, 相关由边缘/纹理主导。"""
    gx = x[:, :, 2:] - x[:, :, :-2]
    gx = torch.nn.functional.pad(gx, (1, 1, 0, 0))
    gy = x[:, 2:, :] - x[:, :-2, :]
    gy = torch.nn.functional.pad(gy, (0, 0, 1, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-8)

def _gauss1d_f(arr, sigma):
    """零相位高斯平滑 (replicate 边界) — 不引入时间滞后。"""
    if float(sigma) <= 0.0:
        return np.asarray(arr, dtype=np.float32)
    r = max(1, int(round(3.0 * float(sigma))))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * float(sigma) ** 2))
    k = k / k.sum()
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
    return torch.nn.functional.conv1d(
        torch.nn.functional.pad(t, (r, r), mode="replicate"), k[None, None])[0, 0].numpy()

def _ma_clean(arr, ok, K):
    """失败帧线性插值 → 中值(9) (保阶跃、压尖刺) → 零相位高斯(σ=2.5)。
    替代旧版 Hampel + [1,2,1]/4: 旧版只把白噪声减半, 残噪逐帧直通进贴回网格
    = 贴回抖动主因; 新链噪声抑制强一个量级且同样零相位, 块边界真实阶跃保留。"""
    if not bool(ok.any()):
        return torch.zeros(K)
    vi = torch.nonzero(ok).flatten()
    if int(vi.numel()) == 1:
        a2 = arr[vi].repeat(K).numpy().astype(np.float32)
    else:
        a2 = _ma_interp1d(torch.arange(K, dtype=torch.float32),
                          vi.float(), arr[vi]).numpy().astype(np.float32)
    r = _MA_MED // 2
    med = np.empty(K, dtype=np.float32)
    for i in range(K):
        med[i] = np.median(a2[max(0, i - r): i + r + 1])
    return torch.from_numpy(_gauss1d_f(med, _MA_GSIGMA))


def _motion_align_track(img, can_rows, geo, f0, sizes, K, dev):
    """v18 = v16 中心模板 NCC 骨架 + 三处强化:
    ① 梯度幅值域 (对扩散重绘的非线性色调漂移更稳);
    ② 次峰抑制门控 (唯一尖峰才可信, 平台/周期纹理剔除);
    ③ 时域 中值(9)+零相位高斯(2.5) 替代 Hampel+[1,2,1]/4 (残噪直通是抖动主因)。
    接口/返回与 v16 一致: (dx, dy) 已含 fx/fy; 全帧失败回退 fx/fy。"""
    try:
        W = _MA_WORK
        t = max(24, int(W * _MA_TMPL)) & ~1
        p0 = (W - t) // 2
        R = max(4, int(W * _MA_SEARCH))
        S = torch.arange(p0 - R, p0 + R + 1, device=dev)
        n_s = int(S.numel())
        P2 = W + t
        meas_x = torch.full((K,), float("nan"))
        meas_y = torch.full((K,), float("nan"))
        ok = torch.zeros(K, dtype=torch.bool)
        CH = 32
        for c0 in range(0, K, CH):
            c1 = min(c0 + CH, K)
            A_l, B_l = [], []
            for i in range(c0, c1):
                Su = sizes[i]
                ix1, iy1 = geo[i][0], geo[i][1]
                a = img[f0 + i, iy1:iy1 + Su, ix1:ix1 + Su].to(torch.float32).mean(dim=-1)[None, None]
                A_l.append(torch.nn.functional.interpolate(a, size=(W, W), mode="area").to(dev)[0, 0])
                b = can_rows[i].to(torch.float32).mean(dim=-1)[None, None]
                B_l.append(torch.nn.functional.interpolate(b, size=(W, W), mode="area").to(dev)[0, 0])
            A = torch.stack(A_l)
            B = torch.stack(B_l)
            del A_l, B_l
            A = _grad2d(A)                       # ← v18①
            B = _grad2d(B)                       # ← v18①
            T = B[:, p0:p0 + t, p0:p0 + t]       # 模板 = 画布中心 (脸)
            FA = torch.fft.rfft2(A, s=(P2, P2))
            FT = torch.fft.rfft2(T, s=(P2, P2))
            G = torch.fft.irfft2(FA * torch.conj(FT), s=(P2, P2))
            Ap = torch.zeros_like(G)
            Ap[:, :W, :W] = A
            II1 = torch.cumsum(torch.cumsum(Ap, dim=1), dim=2)
            II2 = torch.cumsum(torch.cumsum(Ap * Ap, dim=1), dim=2)

            def _win(II):
                a1 = II.index_select(1, S + t).index_select(2, S + t)
                a2 = II.index_select(1, S).index_select(2, S + t)
                a3 = II.index_select(1, S + t).index_select(2, S)
                a4 = II.index_select(1, S).index_select(2, S)
                return a1 - a2 - a3 + a4

            S1 = _win(II1)
            S2 = _win(II2)
            meanT = T.mean(dim=(1, 2))
            ssT = ((T - meanT[:, None, None]) ** 2).sum(dim=(1, 2))
            Gs = G.index_select(1, S).index_select(2, S)
            num = Gs - S1 * meanT[:, None, None]
            varA = torch.clamp(S2 - S1 * S1 / float(t * t), min=1e-8)
            ncc = num / torch.sqrt(varA * torch.clamp(ssT, min=1e-8)[:, None, None])
            flat = ncc.reshape(ncc.shape[0], -1)
            pk, arg = flat.max(dim=1)
            # ---- v18② 次峰抑制 ----
            m_b = int(pk.shape[0])
            ar = torch.arange(m_b, device=ncc.device)
            py_, px_ = arg // n_s, arg % n_s
            oy_, ox_ = torch.meshgrid(torch.arange(-3, 4, device=ncc.device),
                                      torch.arange(-3, 4, device=ncc.device), indexing="ij")
            ny_ = (py_[:, None] + oy_.reshape(1, -1)).clamp(0, n_s - 1)
            nx_ = (px_[:, None] + ox_.reshape(1, -1)).clamp(0, n_s - 1)
            sup = ncc.clone()
            sup[ar[:, None], ny_, nx_] = -1e30
            sec = sup.reshape(m_b, -1).max(dim=1).values
            del sup
            med = flat.median(dim=1).values
            good = (pk >= _MA_PEAK) \
                 & (pk / torch.clamp(sec.abs(), min=1e-3) >= _MA_SEC) \
                 & (pk / torch.clamp(med.abs(), min=1e-3) >= _MA_PROM)

            def _sub3(v0, v1, v2):
                dd, ee = v1 - v0, v1 - v2
                s = 0.5 * (dd - ee) / (dd + ee) if (dd + ee) > 1e-12 else 0.0
                return float(min(0.5, max(-0.5, s)))

            for m_i in range(int(pk.shape[0])):
                if not bool(good[m_i]):
                    continue
                y0, x0 = int(arg[m_i]) // n_s, int(arg[m_i]) % n_s
                nm = ncc[m_i]
                sy = y0 + _sub3(nm[max(0, y0 - 1), x0], nm[y0, x0], nm[min(n_s - 1, y0 + 1), x0])
                sx = x0 + _sub3(nm[y0, max(0, x0 - 1)], nm[y0, x0], nm[y0, min(n_s - 1, x0 + 1)])
                gi = c0 + m_i
                meas_x[gi] = (sx - R)
                meas_y[gi] = (sy - R)
                ok[gi] = True
        if not bool(ok.any()):
            fx_all = torch.tensor([float(geo[i][2]) for i in range(K)])
            fy_all = torch.tensor([float(geo[i][3]) for i in range(K)])
            return fx_all, fy_all
        fx_w = torch.tensor([geo[i][2] * (W / float(sizes[i])) for i in range(K)])
        fy_w = torch.tensor([geo[i][3] * (W / float(sizes[i])) for i in range(K)])
        mov_x = _ma_clean(meas_x - fx_w, ok, K)   # ← v18③
        mov_y = _ma_clean(meas_y - fy_w, ok, K)
        _mm = mov_x.abs() + mov_y.abs()
        h3ff.vlog(f"[H3-FaceBlend] ma_track: ok {int(ok.sum())}/{K}, "
                  f"|mov| med={float(_mm.median()):.2f} max={float(_mm.max()):.2f} (W px)")
        sz = torch.tensor([float(s) for s in sizes])
        dx = mov_x * (sz / float(W)) + torch.tensor([float(geo[i][2]) for i in range(K)])
        dy = mov_y * (sz / float(W)) + torch.tensor([float(geo[i][3]) for i in range(K)])
        lim = sz * _MA_MAXSH
        dx = torch.minimum(torch.maximum(dx, -lim), lim)
        dy = torch.minimum(torch.maximum(dy, -lim), lim)
        dx = torch.where(dx.abs() < 0.05, torch.zeros_like(dx), dx)
        dy = torch.where(dy.abs() < 0.05, torch.zeros_like(dy), dy)
        return dx, dy
    except Exception as _e:
        h3ff.warn(f"[H3-FaceBlend] motion_align estimation failed ({_e}) — "
                  f"falling back to plain paste\n"
                  f"[H3-FaceBlend] 位置修正估计失败 ({_e}) — 回退普通贴回")
        try:
            return (torch.tensor([float(geo[i][2]) for i in range(K)]),
                    torch.tensor([float(geo[i][3]) for i in range(K)]))
        except Exception:
            return torch.zeros(K), torch.zeros(K)



class H3FaceBlend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3FaceBlend",
            display_name="Minimax_H3_Face_Blend",
            category="MinimaxH3_AutoContext/FaceFix",
            description="Face fix step 3: per-subtrack scale + pixel blend-back (no VAE). "
                        "v15: low-frequency lock paste-back — structure/color come from the "
                        "original frames (no jitter at higher sigmas), texture/detail from canvas.\n"
                        "修脸第 3 步: 逐子轨缩放 + 像素贴回 (无VAE)。v15: 金字塔低频锁定贴回 — "
                        "结构/色彩锁定原帧 (高 sigma 也不抖), 纹理细节取自画布。",
            inputs=[
                io.Image.Input("images", tooltip="Original video frames\n原视频画面"),
                io.Image.Input("canvas", tooltip="Canvas from Minimax_H3_Face_Resample\n来自 Face_Resample 的画布"),
                io.Dict.Input("bbox", tooltip="face_pack from Minimax_H3_Face_Resample\n来自 Face_Resample 的 face_pack"),
                io.Int.Input("feather_px", default=16, min=0, max=128,
                             tooltip="Blend-back edge feather (pixels); box mode: rectangle edge, "
                                     "mask mode: erode+blur around SeC mask edge (scaled per-frame in v14)\n"
                                     "贴回边缘羽化 (像素); 框模式羽化矩形边, mask 模式羽化 SeC mask 边 "
                                     "(v14 起随逐帧窗口等比缩放)"),
                io.Boolean.Input("use_sec_mask", default=False,
                                 tooltip="Blend back with feathered SeC mask (face only, background untouched). "
                                         "Source priority: masks port > face_pack > feathered box\n"
                                         "用羽化的 SeC mask 贴回 (只贴人脸)。来源优先级: masks端口 > face_pack > 羽化框回退"),
                io.Mask.Input("masks", optional=True,
                              tooltip="SeC masks from Minimax_H3_Face_Cut (rows 1:1 with crop_images/canvas rows, "
                                      "1=face). Connect directly from Cut, bypassing Resample\n"
                                      "来自 Face_Cut 的 SeC mask (行与裁剪/画布 1:1 对齐, 1=人脸)。"
                                      "从 Cut 直连, 不经过 Resample"),
                io.Boolean.Input("lowfreq_lock", default=False,
                                 tooltip="Pyramid low-freq lock: keep structure/color/geometry from the ORIGINAL "
                                         "frames, take only texture detail from the canvas. For DETAIL-sharpening "
                                         "workflows at LOW sigma (suppresses jitter). Keep OFF for structural repair "
                                         "of corrupted faces — at high sigma the canvas redraws correct geometry "
                                         "and this lock would keep the broken original geometry\n"
                                         "金字塔低频锁定: 结构/色彩/几何取原帧, 仅纹理细节取画布。适用于低 σ 细节"
                                         "锐化 (抑制抖动)。修'崩坏脸'请保持关闭 — 高 σ 下画布重绘了正确的五官几何, "
                                         "锁定反而会保住原帧的错误几何"),
                io.Boolean.Input("motion_align", default=True,
                                 tooltip="Local position correction at paste-back: per-frame NCC estimates how far "
                                         "the resampled canvas face drifted from the ORIGINAL frame's face and "
                                         "cancels it — the pasted face is glued to the original face position, "
                                         "its motion fully inherited from the source video. Kills paste-back "
                                         "jitter (esp. block boundaries at high sigma). Peak-gated + temporally "
                                         "median-smoothed; frames where the estimate is unreliable are "
                                         "interpolated. Soft-fails to plain paste\n"
                                         "贴回局部位置修正: 逐帧 NCC 估计重采样画布脸相对原帧脸的位置漂移并抵消 — "
                                         "贴回的脸被粘到原帧面部的真实位置, 运动完全继承原视频。消除贴回后抖动 "
                                         "(尤其高 σ 块边界)。峰值门控 + 时间中值平滑, 不可靠帧由邻帧插值, "
                                         "失败自动回退普通贴回"),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )


    @classmethod
    def execute(cls, images, canvas, bbox, feather_px=16, use_sec_mask=False, masks=None,
                lowfreq_lock=False, motion_align=True) -> io.NodeOutput:
        pack = bbox or {}
        if "subtracks" not in pack or int(pack.get("version") or 0) not in (5, 6, 7):
            raise ValueError(
                "[H3-FaceBlend] bbox port needs a face_pack (version 5/6/7 with 'subtracks') — "
                "connect Minimax_H3_Face_Resample's bbox output. shot_info is a SHOT MAP "
                "(shots/fps), not geometry — it must NOT go here.\n"
                "[H3-FaceBlend] bbox 端口需要 face_pack (version 5/6/7, 含 subtracks) — "
                "请接 Minimax_H3_Face_Resample 的 bbox 输出。shot_info 是镜头表 (shots/fps), "
                "不是几何包, 不能接这里 (接 FaceCut 的 face_pack 也可以, 本节点会自动用 crop_off 兜底)")
        # ---- 子轨收集: canvas_off (Resample 产出) > crop_off (FaceCut 原始 pack 兜底) ----
        entries = []
        _off_src_warned = False
        for st in (pack.get("subtracks") or []):
            if st.get("skip"):
                continue
            off = st.get("canvas_off")
            src = "canvas_off"
            if off is None:
                off = st.get("crop_off")
                src = "crop_off"
                if off is not None and not _off_src_warned:
                    _off_src_warned = True
                    h3ff.warn("[H3-FaceBlend] bbox came from FaceCut's face_pack (no canvas_off) — "
                              "using crop_off (identical 1:1 ledger). Recommended wiring is still "
                              "Resample's bbox\n"
                              "[H3-FaceBlend] bbox 来自 FaceCut 的 face_pack (无 canvas_off) — "
                              "已用 crop_off 兜底 (账本 1:1 等价)。推荐接线仍是 Resample 的 bbox")
            if off is None:
                continue
            entries.append((st, int(off)))
        if not entries:
            h3ff.log("[H3-FaceBlend] no blended subtrack, passed through\n"
                     "[H3-FaceBlend] 无需贴回的子轨, 原样透传")
            return io.NodeOutput(images)
        entries.sort(key=lambda t: (int(t[0]["f0"]), int(t[0].get("track_id", 0))))

        # ---- masks 端口 (行与画布对齐, 直连自 Cut) ----
        port = None
        if masks is not None:
            port = masks.detach().float()
            if port.dim() == 2:
                port = port.unsqueeze(0)
            need = max(int(st["canvas_off"] if st.get("canvas_off") is not None else st["crop_off"])
                       + int(st["f1"]) - int(st["f0"]) for st, _o in entries)
            if int(port.shape[0]) < need:
                h3ff.warn(f"[H3-FaceBlend] masks port rows {int(port.shape[0])} < expected {need}, "
                          f"ignoring port (rerun Face_Cut)\n"
                          f"[H3-FaceBlend] masks 端口行数 {int(port.shape[0])} < 期望 {need}, "
                          f"忽略端口 (请重跑 Face_Cut)")
                port = None
            elif int(port.shape[0]) > need:
                port = port[:need]

        img = images
        if img.dim() == 5:
            img = img[0]
        img = img.detach().float()
        if canvas.dim() == 5:
            canvas = canvas[0]
        can = canvas.detach().float()
        H, W = int(img.shape[1]), int(img.shape[2])
        Kc = sum(int(st["f1"]) - int(st["f0"]) for st, _o in entries)
        if int(can.shape[0]) != Kc:
            raise ValueError(f"[H3-FaceBlend] canvas rows {int(can.shape[0])} != expected {Kc}\n"
                             f"[H3-FaceBlend] 画布行数 {int(can.shape[0])} ≠ 期望 {Kc}")

        # ---- 计算设备: 有 CUDA 就批处理上 GPU, 否则同一代码路径走 CPU ----
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        out = img.clone()
        blended_rows = 0
        n_masked = 0
        n_locked = 0
        n_aligned = 0
        warned_fallback = False

        for st, off in entries:
            f0, f1, S = int(st["f0"]), int(st["f1"]), int(st["S"])
            K = f1 - f0
            S_list = st.get("S_list")
            if S_list is not None and len(S_list) != K:
                S_list = None
            sizes = [int(S_list[fr]) if (S_list is not None and int(S_list[fr]) > 0) else S
                     for fr in range(K)]

            # ---- alpha 来源: masks端口 > pack内置 > 羽化框 ----
            masks_t = None
            masks_src = None
            if use_sec_mask:
                if port is not None and int(port.shape[0]) >= off + K \
                        and float(port[off:off + K].max()) > 0.0:
                    masks_t = port[off:off + K].mul(255.0).byte()  # 统一到 0..255 约定
                    masks_src = "port"
                if masks_t is None:
                    pm = st.get("masks")
                    if pm is not None and int(pm.shape[0]) == K and int(pm.max()) > 0:
                        masks_t = pm
                        masks_src = "pack"
                if masks_t is not None:
                    n_masked += 1
                elif not warned_fallback:
                    warned_fallback = True
                    h3ff.warn("[H3-FaceBlend] use_sec_mask=True but no masks available "
                              "(single mode / old pack / SeC lost frames) -> feathered box\n"
                              "[H3-FaceBlend] use_sec_mask=True 但无可用 mask "
                              "(单脸模式/旧pack/SeC丢帧) → 回退羽化框")

            # ---- 逐帧窗口几何预计算 (与 v14 相同公式) ----
            geo = []
            for fr in range(K):
                S_fr = sizes[fr]
                cx, cy = st["centers"][fr]
                ex = float(cx) - S_fr * 0.5
                ey = float(cy) - S_fr * 0.5
                ix1 = int(np.floor(ex))
                iy1 = int(np.floor(ey))
                fx = ex - ix1
                fy = ey - iy1
                ix1 = max(0, min(ix1, W - S_fr))
                iy1 = max(0, min(iy1, H - S_fr))
                fx = float(min(max(fx, 0.0), 0.999))
                fy = float(min(max(fy, 0.0), 0.999))
                geo.append((ix1, iy1, fx, fy))

            can_rows = can[off:off + K]

            # ---- motion_align: 逐帧估计 画布脸→原帧脸 位置修正 (v16, 软失败) ----
            madx = mady = None
            if motion_align and K >= 2:
                madx, mady = _motion_align_track(img, can_rows, geo, f0, sizes, K, dev)
                _ad = (madx.abs() + mady.abs())
                h3ff.vlog(f"[H3-FaceBlend] id{st.get('track_id', 0)} motion_align: "
                          f"mean|d|={float(_ad.mean()):.2f}px max|d|={float(_ad.max()):.2f}px "
                          f"({int(K)} frames)\n"
                          f"[H3-FaceBlend] 身份{st.get('track_id', 0)} 位置修正: "
                          f"平均|d|={float(_ad.mean()):.2f}px 最大|d|={float(_ad.max()):.2f}px "
                          f"(共 {int(K)} 帧)")
                if float(_ad.max()) > 1e-6:
                    n_aligned += 1

            # ---- 等尺寸 run: 缩放 → (批量)亚像素 → (批量)混合 → 写回 ----
            kk = 0
            while kk < K:
                kk2 = kk
                while kk2 < K and sizes[kk2] == sizes[kk]:
                    kk2 += 1
                Su = sizes[kk]
                f_px = max(0, min(int(feather_px), (Su - 2) // 2))
                blk = can_rows[kk:kk2].movedim(-1, 1).contiguous().to(dev, torch.float32)
                if int(blk.shape[-2]) != Su or int(blk.shape[-1]) != Su:
                    blk = comfy.utils.common_upscale(blk, Su, Su, "lanczos", "disabled")
                lock_sigma = float(min(14.0, max(4.0, Su * 0.05))) if lowfreq_lock else 0.0
                chunk = max(1, int(48 * 1024 * 1024) // max(1, 3 * Su * Su * 4))
                for a in range(kk, kk2, chunk):
                    b = min(a + chunk, kk2)
                    n = b - a
                    reg = torch.stack([
                        out[f0 + i, geo[i][1]:geo[i][1] + Su, geo[i][0]:geo[i][0] + Su, :]
                        for i in range(a, b)], dim=0).permute(0, 3, 1, 2).to(dev, torch.float32)
                    can_c = blk[a - kk:b - kk]

                    if madx is not None:
                        tot_x = [float(madx[a + j]) for j in range(n)]
                        tot_y = [float(mady[a + j]) for j in range(n)]
                    else:
                        tot_x = [geo[a + j][2] for j in range(n)]
                        tot_y = [geo[a + j][3] for j in range(n)]
                    sh_idx = [j for j in range(n) if abs(tot_x[j]) > 1e-3 or abs(tot_y[j]) > 1e-3]
                    if sh_idx:
                        ax_ = torch.arange(Su, dtype=torch.float32, device=dev)
                        grids = []
                        for j in sh_idx:
                            gn = ((ax_ + 0.5 - tot_x[j]) / Su) * 2.0 - 1.0
                            gm = ((ax_ + 0.5 - tot_y[j]) / Su) * 2.0 - 1.0
                            grids.append(torch.stack(
                                [gn[None, :].expand(Su, Su), gm[:, None].expand(Su, Su)], dim=-1))
                        grid = torch.stack(grids, dim=0)  # [m,Su,Su,2]
                        sel = torch.tensor(sh_idx, device=dev, dtype=torch.long)
                        shifted = torch.nn.functional.grid_sample(
                            can_c.index_select(0, sel), grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
                        can_c = can_c.clone()
                        can_c[sel] = shifted

                    if lock_sigma > 0.0:
                        can_c = _pyramid_mix(reg, can_c, lock_sigma, dev)
                        n_locked += n

                    # alpha: rect (run 内常数, 一次生成) / mask )
                    if masks_t is not None:
                        alpha = torch.stack(
                            [_mask_alpha_frame(masks_t[a + j], Su, f_px, dev) for j in range(n)],
                            dim=0)[:, None]  # [n,1,Su,Su]
                    else:
                        alpha = h3ff._rect_weight(Su, Su, f_px, dev,
                                                  torch.float32)[None, None]  # [1,1,Su,Su]

                    res_blend = reg * (1.0 - alpha) + can_c * alpha
                    res_cpu = res_blend.permute(0, 2, 3, 1).cpu()
                    for j in range(n):
                        i = a + j
                        ix1, iy1 = geo[i][0], geo[i][1]
                        out[f0 + i, iy1:iy1 + Su, ix1:ix1 + Su, :] = res_cpu[j]
                kk = kk2
            blended_rows += K

            uniq = sorted(set(sizes))
            mode = (f"mask-blend[{masks_src}] per-frame window S∈[{uniq[0]},{uniq[-1]}] subpixel"
                    if masks_t is not None else
                    f"box-blend per-frame window S∈[{uniq[0]},{uniq[-1]}] feather={feather_px}px subpixel")
            if lowfreq_lock:
                mode += f" lowfreq-lock(σ≈{min(14.0, max(4.0, uniq[-1] * 0.05)):.1f}px)"
            h3ff.vlog(f"[H3-FaceBlend] id{st.get('track_id', 0)} sub [{f0},{f1}) {mode} blended\n"
                      f"[H3-FaceBlend] 身份{st.get('track_id', 0)} 子轨 [{f0},{f1}) {mode} 贴回完成")

        h3ff.log(f"[H3-FaceBlend] done: {len(entries)} subtracks "
                 f"({n_masked} mask / {len(entries) - n_masked} box, {n_aligned} motion-aligned), "
                 f"{blended_rows} rows ({n_locked} lowfreq-locked), "
                 f"{int(out.shape[0])} frames, no VAE, dev={dev.type}\n"
                 f"[H3-FaceBlend] 完成: {len(entries)} 条子轨 "
                 f"({n_masked} 条 mask / {len(entries) - n_masked} 条 box, {n_aligned} 条位置修正), "
                 f"{blended_rows} 行 ({n_locked} 帧低频锁定), {int(out.shape[0])} 帧, "
                 f"零VAE, 设备={dev.type}")
        return io.NodeOutput(out.contiguous())
