"""h3_face_blend.py — 修脸第 3 步: 全部缩放 + 像素贴回 (v8, 零VAE)

② 输出原生 res² 画布, 本节点完成唯一一次缩放 (lanczos → S×S) 后按逐帧平滑中心
贴回原画面。canvas 接 ① 的 crop_images (已是 S×S) 时跳过缩放, 贴回原像素原坐标
→ 输出与输入逐位一致 (恒等验收, 零容忍)。
"""

import torch
import comfy.utils
from comfy_api.latest import io

try:
    from . import h3_facefix as h3ff
except ImportError:
    import h3_facefix as h3ff


class H3FaceBlend(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3FaceBlend",
            display_name="Minimax_H3_Face_Blend",
            category="MinimaxH3_AutoContext/FaceFix",
            description="Face fix step 3: scale + pixel blend-back (no VAE). canvas takes ② output; "
                        "connecting ① crop_images gives the identity check.\n"
                        "修脸第 3 步: 缩放+像素贴回 (无VAE)。canvas 接②输出; 接①crop_images 则为恒等验收。",
            inputs=[
                io.Image.Input("images", tooltip="Original video frames (main sampling latent after VAE Decode goes here)\n"
                                                 "原视频画面 (主采样 latent 经 VAE Decode 后接这里)"),
                io.Image.Input("canvas", tooltip="Content to blend back: ② images; connect ① crop_images for a dry run\n"
                                                 "待贴内容: ② 的 images; 干跑时接 ① 的 crop_images"),
                io.Dict.Input("bbox", tooltip="bbox from H3FaceResample (or face_pack straight from H3FaceCut)\n"
                                              "来自 H3FaceResample 的 bbox (或直连 H3FaceCut 的 face_pack)"),
                io.Int.Input("feather_px", default=16, min=0, max=128,
                             tooltip="Blend-back edge feather (pixels), 0=hard edge; increase it when seams show up\n"
                                     "贴回边缘羽化 (像素), 0=硬边; 接缝可见就加大"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, images, canvas, bbox, feather_px=16) -> io.NodeOutput:
        pack = bbox or {}
        if int(pack.get("version") or 0) != 3:
            raise ValueError("[H3-FaceBlend] bbox version mismatch — rerun H3FaceCut (v4)"
                             "\n[H3-FaceBlend] bbox 版本不符 — 请重跑 H3FaceCut (v4)")
        S = int(pack.get("crop_size") or 0)
        centers = pack.get("centers") or []
        if S <= 0 or not centers:
            print("[H3-FaceBlend] no face empty pack, passed through unchanged\n"
                  "[H3-FaceBlend] 无脸空包, 原样透传")
            return io.NodeOutput(images)

        img = images
        if img.dim() == 5:
            img = img[0]
        img = img.detach().float()
        if canvas.dim() == 5:
            canvas = canvas[0]
        can = canvas.detach().float()
        F_img, H, W = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
        F_can, h_can, w_can = int(can.shape[0]), int(can.shape[1]), int(can.shape[2])
        if F_img != F_can:
            raise ValueError(f"[H3-FaceBlend] source frames {F_img} ≠ canvas frames {F_can} — "
                             f"both images must come from the same sampling run"
                             f"\n[H3-FaceBlend] 原画面 {F_img} 帧 ≠ canvas {F_can} 帧 — "
                             f"两个 images 必须来自同一次采样")

        # ---- 本节点唯一职责的缩放: canvas → S×S (已是 S×S 则跳过, 干跑恒等) ----
        if (h_can, w_can) != (S, S):
            can = comfy.utils.common_upscale(can.movedim(-1, 1), S, S,
                                             "lanczos", "disabled").movedim(1, -1)
            print(f"[H3-FaceBlend] canvas {h_can}x{w_can} → {S}x{S} (lanczos, done in this node)\n"
                  f"[H3-FaceBlend] canvas {h_can}x{w_can} → {S}x{S} (lanczos, 本节点完成)")

        f_px = max(0, min(int(feather_px), (S - 2) // 2))
        alpha = h3ff._rect_weight(S, S, f_px, img.device, torch.float32).unsqueeze(-1)

        # ---- 逐帧贴回: 中心公式与 ① 完全一致 (保证干跑逐位恒等) ----
        half = S / 2.0
        out = img.clone()
        for fr in range(F_img):
            cx, cy = centers[fr]
            ix1 = max(0, min(int(round(cx - half)), W - S))
            iy1 = max(0, min(int(round(cy - half)), H - S))
            reg = out[fr, iy1:iy1 + S, ix1:ix1 + S, :]
            out[fr, iy1:iy1 + S, ix1:ix1 + S, :] = reg * (1.0 - alpha) + can[fr] * alpha

        print(f"[H3-FaceBlend] pixel blend-back {F_img} frames, window {S}x{S}, feather={f_px}px, no VAE\n"
              f"[H3-FaceBlend] 像素贴回 {F_img} 帧, 窗口 {S}x{S}, feather={f_px}px, 零VAE")
        return io.NodeOutput(out.contiguous())
