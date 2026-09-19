"""h3_face_cut.py — 修脸第 1 步: 检测与稳定裁剪 (v6, 原生下拉选模型)

全量解码 → 逐帧 YOLO → 缺失填补+中值平滑 → 固定边长 S → 逐帧居中裁剪。
v6: face_model 用 ComfyUI 原生 combo 下拉 (INPUT_TYPES 元组) — 不依赖
io.Choice (部分 comfy 版本没有, v5 因此退化成字符串输入, 即问题根源)。
固定扫描目录: ComfyUI/models/elementEasy (含子目录)。
本节点改用原生节点写法; ② ③ 的新 API 写法不受影响, 可混用同一张导出表。
"""

import os
import numpy as np
import torch
import folder_paths

try:
    from . import h3_conditioning
    from . import h3_facefix as h3ff
except ImportError:
    import h3_conditioning
    import h3_facefix as h3ff

# ---- 固定模型目录: ComfyUI/models/elementEasy ----
MODEL_DIR = os.path.join(folder_paths.models_dir, "elementEasy")
_MODEL_EXTS = {".pt", ".pth", ".onnx", ".engine", ".torchscript"}


def _list_models():
    """扫描 models/elementEasy (含子目录), 返回相对路径列表。"""
    out = []
    if os.path.isdir(MODEL_DIR):
        for root, _dirs, files in os.walk(MODEL_DIR):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in _MODEL_EXTS:
                    rel = os.path.relpath(os.path.join(root, fn), MODEL_DIR)
                    out.append(rel.replace("\\", "/"))
    return sorted(out)


class H3FaceCut:
    """原生节点写法: face_model 是 combo 下拉, 所有 comfy 版本必出下拉框。"""

    CATEGORY = "MinimaxH3_AutoContext/FaceFix"
    DESCRIPTION = ("Face fix step 1: per-frame detection -> smooth fill -> fixed-size centered crop (stabilizer). "
                   "Crops only, no sampling.\n"
                   "修脸第 1 步: 逐帧检测→平滑填补→固定尺寸居中裁剪 (稳定器)。只切不算。")
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "*")
    RETURN_NAMES = ("crop_images", "face_pack")
    OUTPUT_TOOLTIPS = ("Fixed-size crop sequence with the face held at the center\n"
                       "脸恒居中的固定尺寸裁剪序列",
                       "Geometry pack (centers/crop_size/meta + a_lat) for steps 2 and 3\n"
                       "几何信息包 (centers/crop_size/meta + a_lat), 供 ② ③ 使用")

    @classmethod
    def INPUT_TYPES(cls):
        choices = _list_models() or [""]
        return {
            "required": {
                "face_model": (choices, {
                    "tooltip": ("Face detection model (dropdown = files inside ComfyUI/models/elementEasy, "
                                "subfolders included); refresh the browser after dropping in new files\n"
                                "人脸检测模型 (下拉 = ComfyUI/models/elementEasy 内的文件, "
                                "含子目录)。新放入文件后刷新浏览器即可出现"),
                }),
                "latent": ("LATENT", {"tooltip": "Full latent output of the sampler node (first latent socket)\n"
                                                 "采样节点输出的完整 latent (第一个 latent 口)"}),
                "vae": ("VAE", {"tooltip": "Video VAE (used to decode frames)\n视频 VAE (抽帧解码用)"}),
                "conf": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.9, "step": 0.05,
                                   "tooltip": "Detection confidence threshold (raise it when dark scenes give false hits)\n"
                                              "检测置信度阈值 (暗场误检多就调高)"}),
                "res": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32,
                                "tooltip": "Canvas side length: the crop sequence is upscaled to this size for H3 redraw\n"
                                           "画布边长: 裁剪序列放大到该尺寸交给 H3 重绘"}),
                "expand": ("INT", {"default": 20, "min": 0, "max": 100,
                                   "tooltip": "Crop margin % (background kept around the face box, needed for the feather blend-back)\n"
                                              "裁剪余量% (脸框四周保留的背景, 贴回羽化需要它)"}),
            },
        }

    def execute(self, latent, vae, face_model="", conf=0.3, res=512, expand=20):
        v_lat, a_lat = h3_conditioning.unpack_nested_latent(latent)
        if v_lat is None or v_lat.dim() != 5 or a_lat is None:
            raise ValueError("[H3-FaceCut] latent 必须同时含 video+audio")
        B, C, T, LH, LW = v_lat.shape
        if T < 4:
            raise ValueError(f"[H3-FaceCut] 视频 token 数 {T} 过短")
        H, W = LH * 16, LW * 16

        # ---- 模型路径解析 (固定目录) ----
        model_rel = (face_model or "").strip()
        if not model_rel:
            raise ValueError(f"[H3-FaceCut] 未选择检测模型 — 请将 YOLO 权重放入: {MODEL_DIR}")
        model_path = os.path.normpath(os.path.join(MODEL_DIR, model_rel))
        if not os.path.isfile(model_path):
            raise ValueError(f"[H3-FaceCut] 模型不存在: {model_path} — "
                             f"新放入的文件需刷新浏览器后才会出现在下拉列表")

        # ---- 1) 全量解码 + 逐帧检测 ----
        frames = h3ff.decode_probe_frames(vae, v_lat, list(range(int(T))))
        per_frame = h3ff.detect_faces(frames, {"conf": float(conf),
                                               "model": model_path})
        n_hit = sum(1 for b in per_frame if b)

        # ---- 2) 稳定轨迹 ----
        expand_f = max(float(expand) / 100.0, 0.10)
        track = h3ff.build_stable_track(per_frame, expand_f)
        fps_eff = max(1.0, h3ff._pixels_for_tokens(int(T)) * h3ff.AUDIO_LATENTS_PER_SEC
                      / max(int(a_lat.shape[-1]), 1))
        pack = {"version": 3, "a_lat": a_lat, "crop_size": 0, "centers": [],
                "fps_eff": float(fps_eff),
                "meta": {"T": int(T), "LH": int(LH), "LW": int(LW),
                         "W": int(W), "H": int(H), "res": int(res),
                         "n_hit": int(n_hit), "n_frames": len(per_frame)}}
        if track is None:
            print("[H3-FaceCut] no face detected -> empty pack (passed through unchanged downstream)\n"
                  "[H3-FaceCut] 未检测到人脸 → 空包 (下游原样透传)")
            return (torch.zeros(1, 64, 64, 3), pack)
        sm, S = track
        miss = 1.0 - n_hit / float(len(per_frame))
        print(f"\033[35m[H3-FaceCut] detected {n_hit}/{len(per_frame)} frames (missing {miss * 100:.0f}%, "
              f"{'high, raise conf in dark scenes' if miss > 0.4 else 'ok'}) | "
              f"fixed crop S={S}px -> canvas {int(res)}px (upscale {float(res) / S:.2f}x)\n"
              f"[H3-FaceCut] 检出 {n_hit}/{len(per_frame)} 帧 (缺失{miss * 100:.0f}%, "
              f"{'偏高, 暗场可升conf' if miss > 0.4 else 'ok'}) | "
              f"固定裁剪 S={S}px → 画布 {int(res)}px (放大 {float(res) / S:.2f}x)\033[0m")

        # ---- 3) 逐帧居中裁剪 ----
        half = S / 2.0
        centers, crops = [], []
        for i in range(len(per_frame)):
            x1, y1, x2, y2 = sm[i]
            cx = min(max((x1 + x2) * 0.5, half), W - half)
            cy = min(max((y1 + y2) * 0.5, half), H - half)
            ix1 = int(round(cx - half)); iy1 = int(round(cy - half))
            ix1 = max(0, min(ix1, W - S)); iy1 = max(0, min(iy1, H - S))
            centers.append([ix1 + half, iy1 + half])
            crops.append(frames[i][iy1:iy1 + S, ix1:ix1 + S])
        crop_t = torch.from_numpy(np.stack(crops)).to(torch.float32) / 255.0  # [F,S,S,C]

        pack["crop_size"] = int(S)
        pack["centers"] = centers
        print(f"[H3-FaceCut] crop sequence {tuple(crop_t.shape)} "
              f"(fc={crop_t.shape[0]}, expected {h3ff._pixels_for_tokens(int(T))})\n"
              f"[H3-FaceCut] 裁剪序列 {tuple(crop_t.shape)} "
              f"(fc={crop_t.shape[0]}, 期望 {h3ff._pixels_for_tokens(int(T))})")
        return (crop_t.contiguous(), pack)
