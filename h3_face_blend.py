"""h3_face_blend.py — 修脸第 3 步: 缩放 + 像素贴回 (v10, 逐子轨)

pack v4: 每条 skip=False 且 canvas_off 有效的子轨, 从画布切出自己的行,
lanczos 缩到 S_i, 按各自 centers 贴回对应帧区间; 未覆盖帧保留原像素。
接缝仅存在于子轨边界 (分镜切换处), 纯像素运算无 VAE。
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
            description="Face fix step 3: per-subtrack scale + pixel blend-back (no VAE).\n"
                        "修脸第 3 步: 逐子轨缩放 + 像素贴回 (无VAE)。",
            inputs=[
                io.Image.Input("images", tooltip="Original video frames (main latent after VAE Decode)\n"
                               "原视频画面 (主采样 latent 经 VAE Decode 后接这里)"),
                io.Image.Input("canvas", tooltip="Canvas from H3FaceResample\n"
                               "来自 H3FaceResample 的画布"),
                io.Dict.Input("bbox", tooltip="bbox from H3FaceResample\n来自 H3FaceResample 的 bbox"),
                io.Int.Input("feather_px", default=16, min=0, max=128,
                             tooltip="Blend-back edge feather (pixels)\n贴回边缘羽化 (像素)"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, images, canvas, bbox, feather_px=16) -> io.NodeOutput:
        pack = bbox or {}
        if int(pack.get("version") or 0) != 5:
            raise ValueError("[H3-FaceBlend] bbox version mismatch — rerun H3FaceCut (v10)"
                             "\n[H3-FaceBlend] bbox 版本不符 — 请重跑 H3FaceCut (v10)")
        subs = [st for st in (pack.get("subtracks") or [])
                if not st.get("skip") and st.get("canvas_off") is not None]
        if not subs:
            print("[H3-FaceBlend] no blended subtrack, passed through\n"
                  "[H3-FaceBlend] 无需贴回的子轨, 原样透传")
            return io.NodeOutput(images)

        img = images
        if img.dim() == 5:
            img = img[0]
        img = img.detach().float()
        if canvas.dim() == 5:
            canvas = canvas[0]
        can = canvas.detach().float()
        H, W = int(img.shape[1]), int(img.shape[2])
        Kc = sum(int(st["f1"]) - int(st["f0"]) for st in subs)
        if int(can.shape[0]) != Kc:
            raise ValueError(f"[H3-FaceBlend] canvas rows {int(can.shape[0])} != expected {Kc}"
                             f"\n[H3-FaceBlend] 画布行数 {int(can.shape[0])} ≠ 期望 {Kc}")

        parts = []
        cur = 0
        for st in subs:
            f0, f1, S = int(st["f0"]), int(st["f1"]), int(st["S"])
            off = int(st["canvas_off"])
            K = f1 - f0
            if f0 > cur:
                parts.append(img[cur:f0])               # 未覆盖区间保留原像素
            can_blk = can[off:off + K]
            if (int(can_blk.shape[1]), int(can_blk.shape[2])) != (S, S):
                can_blk = comfy.utils.common_upscale(can_blk.movedim(-1, 1), S, S,
                                                     "lanczos", "disabled").movedim(1, -1)
            f_px = max(0, min(int(feather_px), (S - 2) // 2))
            alpha = h3ff._rect_weight(S, S, f_px, img.device, torch.float32).unsqueeze(-1)
            half = S / 2.0
            out_blk = img[f0:f1].clone()
            for fr in range(K):
                cx, cy = st["centers"][fr]
                ix1 = max(0, min(int(round(cx - half)), W - S))
                iy1 = max(0, min(int(round(cy - half)), H - S))
                reg = out_blk[fr, iy1:iy1 + S, ix1:ix1 + S, :]
                out_blk[fr, iy1:iy1 + S, ix1:ix1 + S, :] = reg * (1.0 - alpha) + can_blk[fr] * alpha
            parts.append(out_blk)
            print(f"[H3-FaceBlend] subtrack [{f0},{f1}) S={S} feather={f_px}px blended\n"
                  f"[H3-FaceBlend] 子轨 [{f0},{f1}) S={S} feather={f_px}px 贴回完成")
            cur = f1
        if cur < int(img.shape[0]):
            parts.append(img[cur:])
        out = torch.cat(parts, dim=0)
        print(f"[H3-FaceBlend] done: {len(subs)} subtracks, {int(out.shape[0])} frames, no VAE\n"
              f"[H3-FaceBlend] 完成: {len(subs)} 条子轨, {int(out.shape[0])} 帧, 零VAE")
        return io.NodeOutput(out.contiguous())
