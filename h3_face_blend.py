"""h3_face_blend.py — 修脸第 3 步: 缩放 + 像素贴回 (v14, 逐帧窗口贴回 + 日志收编)

v14 变更:
- 支持 v20 FaceCut 的逐帧窗口 (subtrack 内新增 S_list): 每帧按各自的 S_fr 缩放画布行
  并贴回 — 窗口随脸平滑缩放, 同一身份贴回无直切感; 旧 pack (无 S_list) 自动回退
  每子轨常数 S, 行为与 v13 一致。
- mask alpha 改为逐帧生成 (腐蚀/羽化随该帧 S_fr 等比缩放)。
- 日志收编: 常规运行只保留 出口摘要 + 警告; 逐子轨细节走 h3ff.vlog。

v13 变更:
- 新增可选输入端口 masks (来自 Minimax_H3_Face_Cut 的 MASK 输出, 直连线, 不经过 Resample)。
  行结构与 crop_images/画布 1:1 对齐 (Cut 侧保证), 1=人脸。
- alpha 来源优先级: masks 端口 > face_pack 内置 masks (旧接线兼容) > 羽化框回退。
- use_sec_mask=False 时忽略一切 mask, 行为与 v11 box 羽化完全一致。
- feather_px 复用: box 模式羽化矩形边; mask 模式 = 腐蚀 feather/2 + blur σfeather/2。
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


class H3FaceBlend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3FaceBlend",
            display_name="Minimax_H3_Face_Blend",
            category="MinimaxH3_AutoContext/FaceFix",
            description="Face fix step 3: per-subtrack scale + pixel blend-back (no VAE). "
                        "v14: supports v20 FaceCut per-frame windows (S_list) for seamless "
                        "same-identity paste-back.\n"
                        "修脸第 3 步: 逐子轨缩放 + 像素贴回 (无VAE)。v14: 支持 v20 FaceCut 逐帧窗口 "
                        "(S_list), 同一身份贴回无直切。",
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
            ],
            outputs=[io.Image.Output(display_name="images")],
        )


    @classmethod
    def execute(cls, images, canvas, bbox, feather_px=16, use_sec_mask=False, masks=None) -> io.NodeOutput:
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

        out = img.clone()
        blended_rows = 0
        n_masked = 0
        warned_fallback = False

        for st, off in entries:
            f0, f1, S = int(st["f0"]), int(st["f1"]), int(st["S"])
            K = f1 - f0
            S_list = st.get("S_list")
            if S_list is not None and len(S_list) != K:
                S_list = None
            sizes = [int(S_list[fr]) if (S_list is not None and int(S_list[fr]) > 0) else S
                     for fr in range(K)]

            can_rows = can[off:off + K]                         
            scaled = [None] * K
            kk = 0
            while kk < K:
                kk2 = kk
                while kk2 < K and sizes[kk2] == sizes[kk]:
                    kk2 += 1
                Su = sizes[kk]
                blk = can_rows[kk:kk2]
                if (int(blk.shape[1]), int(blk.shape[2])) != (Su, Su):
                    blk = comfy.utils.common_upscale(
                        blk.movedim(-1, 1).contiguous(), Su, Su,
                        "lanczos", "disabled").movedim(1, -1)
                for i in range(blk.shape[0]):
                    scaled[kk + i] = blk[i]                     
                kk = kk2

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

            for fr in range(K):
                S_fr = sizes[fr]
                f_px_fr = max(0, min(int(feather_px), (S_fr - 2) // 2))
                cx, cy = st["centers"][fr]
                can_fr = scaled[fr]     

                # ---- 亚像素贴回: 消除整数贴回与亚像素裁剪之间的 ±0.5px 相对抖动 (脸缓慢移动时的微晃根源)。 ----
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
                if fx > 0.0 or fy > 0.0:
                    ax_ = torch.arange(S_fr, dtype=torch.float32, device=can_fr.device)
                    gn = ((ax_ + 0.5 - fx) / S_fr) * 2.0 - 1.0
                    gm = ((ax_ + 0.5 - fy) / S_fr) * 2.0 - 1.0
                    grid = torch.stack([gn[None, :].expand(S_fr, S_fr),
                                        gm[:, None].expand(S_fr, S_fr)], dim=-1).unsqueeze(0)
                    c_ = can_fr.permute(2, 0, 1).unsqueeze(0)  # [1,3,S,S]
                    can_fr = torch.nn.functional.grid_sample(
                        c_, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)[0].permute(1, 2, 0).contiguous()

                if masks_t is not None:
                    alpha = _mask_alpha_frame(masks_t[fr], S_fr, f_px_fr, img.device).unsqueeze(-1)
                else:
                    alpha = h3ff._rect_weight(S_fr, S_fr, f_px_fr, img.device,
                                              torch.float32).unsqueeze(-1)  # [S,S,1]
                reg = out[f0 + fr, iy1:iy1 + S_fr, ix1:ix1 + S_fr, :]
                out[f0 + fr, iy1:iy1 + S_fr, ix1:ix1 + S_fr, :] = \
                    reg * (1.0 - alpha) + can_fr * alpha
            blended_rows += K

            uniq = sorted(set(sizes))
            mode = (f"mask-blend[{masks_src}] per-frame window S∈[{uniq[0]},{uniq[-1]}] subpixel"
                    if masks_t is not None else
                    f"box-blend per-frame window S∈[{uniq[0]},{uniq[-1]}] feather={feather_px}px subpixel")
            h3ff.vlog(f"[H3-FaceBlend] id{st.get('track_id', 0)} sub [{f0},{f1}) {mode} blended\n"
                      f"[H3-FaceBlend] 身份{st.get('track_id', 0)} 子轨 [{f0},{f1}) {mode} 贴回完成")

        h3ff.log(f"[H3-FaceBlend] done: {len(entries)} subtracks "
                 f"({n_masked} mask / {len(entries) - n_masked} box), "
                 f"{blended_rows} rows, {int(out.shape[0])} frames, no VAE\n"
                 f"[H3-FaceBlend] 完成: {len(entries)} 条子轨 "
                 f"({n_masked} 条 mask / {len(entries) - n_masked} 条 box), "
                 f"{blended_rows} 行, {int(out.shape[0])} 帧, 零VAE")
        return io.NodeOutput(out.contiguous())
