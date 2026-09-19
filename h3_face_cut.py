"""h3_face_cut.py — 修脸第 1 步: 检测与稳定裁剪 (v10, 分镜感知子轨 + 统一 res² 输出)

v10: 窗口与输出解耦 —
- 窗口 (内部): 轨迹按 出现区间 × 尺度突变 切子轨, 每条独立平滑、独立 S_i = 中位脸×(1+expand),
  小脸子轨 S_i 小 → 归一化后脸占画布比例最大 (最该修的得到最多像素);
- 输出 (外部): 每条子轨裁剪后立即 lanczos 归一化到统一 res², 拼成单一 crop_images 张量输出
  (行序 = 子轨序, 与 ② 的画布行结构 1:1 对应)。
- 无脸帧不进子轨; 中位脸 ≥ res×skip_ratio 的子轨跳过 (无行、无采样)。
- pack v5: 子轨几何 (f0/f1/S/centers/crop_off) 进 face_pack, crops 不再进 pack。
解码分块仍按主采样 H3 token 账目 (face_split_blocks / token_blocks)。
"""

import os
import numpy as np
import torch
import folder_paths
import comfy.utils

try:
    from . import h3_conditioning
    from . import h3_facefix as h3ff
except ImportError:
    import h3_conditioning
    import h3_facefix as h3ff

MODEL_DIR = os.path.join(folder_paths.models_dir, "elementEasy")
_MODEL_EXTS = {".pt", ".pth", ".onnx", ".engine", ".torchscript"}


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


def _median_smooth(boxes):
    sm = []
    for i in range(len(boxes)):
        win = boxes[max(0, i - 2): i + 3]
        sm.append([float(np.median([w[k] for w in win])) for k in range(4)])
    return sm


class H3FaceCut:
    CATEGORY = "MinimaxH3_AutoContext/FaceFix"
    DESCRIPTION = ("Face fix step 1: shot-aware subtracks, adaptive crop window, "
                   "uniform res^2 output.\n"
                   "修脸第 1 步: 分镜感知子轨, 裁剪窗口自适应, 输出统一 res²。只切不算。")
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "*")
    RETURN_NAMES = ("crop_images", "face_pack")
    OUTPUT_TOOLTIPS = ("Uniform res^2 crops of all sampled subtracks (row order = subtrack order)\n"
                       "所有待采样子轨的统一 res² 裁剪 (行序=子轨序)",
                       "Subtrack geometry pack (v5): f0/f1/S/centers/crop_off, consumed by ② ③\n"
                       "子轨几何包 (v5): f0/f1/S/centers/crop_off, 供 ② ③ 使用")

    @classmethod
    def INPUT_TYPES(cls):
        choices = _list_models() or [""]
        return {
            "required": {
                "face_model": (choices, {
                    "tooltip": ("Face detection model (dropdown = files inside ComfyUI/models/elementEasy)\n"
                                "人脸检测模型 (下拉 = ComfyUI/models/elementEasy 内的文件)")}),
                "latent": ("LATENT", {"tooltip": "Full latent output of the sampler node\n"
                                      "采样节点输出的完整 latent (第一个 latent 口)"}),
                "vae": ("VAE", {"tooltip": "Video VAE\n视频 VAE (抽帧解码用)"}),
                "conf": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.9, "step": 0.05,
                                   "tooltip": "Detection confidence threshold\n检测置信度阈值"}),
                "res": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 32,
                                "tooltip": "Canvas side length; crops are normalized to res^2 here; "
                                           "subtracks with face >= res*skip_ratio are skipped\n"
                                           "画布边长; 裁剪在此归一化到 res²; 脸 ≥ res×skip_ratio 的子轨跳过"}),
                "expand": ("INT", {"default": 20, "min": 0, "max": 100,
                                   "tooltip": "Crop window margin %\n裁剪窗口余量%"}),
                "gap_tol": ("INT", {"default": 12, "min": 0, "max": 60,
                                    "tooltip": "Max detection gap (frames) treated as one continuous appearance; "
                                               "longer absence splits subtracks. 12 ≈ 0.5s @24fps\n"
                                               "检测缺失多少帧内视为同一次出现; 更长的缺席切开子轨。12 ≈ 0.5秒@24fps"}),
                "scale_split": ("FLOAT", {"default": 1.4, "min": 1.05, "max": 3.0, "step": 0.05,
                                          "tooltip": "Split a new subtrack when face size exceeds scale_split× the "
                                                     "running median (shot/scale change)\n"
                                                     "脸尺寸超过运行中位数的该倍数时切新子轨 (分镜/景别切换)"}),
                "skip_ratio": ("FLOAT", {"default": 0.8, "min": 0.3, "max": 1.0, "step": 0.05,
                                         "tooltip": "Skip resampling when a subtrack's median face size >= "
                                                    "res×this (already large enough)\n"
                                                    "子轨中位脸尺寸 ≥ res×该值时跳过重采样 (已够大)"}),
            },
            "optional": {
                "info": ("*", {"tooltip": "Main sampler info: decode blocks follow the same H3 token accounting\n"
                               "主采样 info: 解码分块按同一套 H3 token 账目"}),
            },
        }

    def execute(self, latent, vae, face_model="", conf=0.3, res=512, expand=20,
                gap_tol=12, scale_split=1.4, skip_ratio=0.8, info=None):
        v_lat, a_lat = h3_conditioning.unpack_nested_latent(latent)
        if v_lat is None or v_lat.dim() != 5 or a_lat is None:
            raise ValueError("[H3-FaceCut] latent must contain both video+audio"
                             "\n[H3-FaceCut] latent 必须同时含 video+audio")
        B, C, T, LH, LW = v_lat.shape
        if T < 4:
            raise ValueError(f"[H3-FaceCut] video token count {T} is too small"
                             f"\n[H3-FaceCut] 视频 token 数 {T} 过短")
        H, W = LH * 16, LW * 16
        F_expect = h3ff._pixels_for_tokens(int(T))

        model_rel = (face_model or "").strip()
        if not model_rel:
            raise ValueError(f"[H3-FaceCut] no detection model selected — put the YOLO weights into: {MODEL_DIR}"
                             f"\n[H3-FaceCut] 未选择检测模型 — 请将 YOLO 权重放入: {MODEL_DIR}")
        model_path = os.path.normpath(os.path.join(MODEL_DIR, model_rel))
        if not os.path.isfile(model_path):
            raise ValueError(f"[H3-FaceCut] model not found: {model_path}"
                             f"\n[H3-FaceCut] 模型不存在: {model_path}")

        # ---- 分块账目: H3 token 域 (与主采样同递推 + 三重校验) ----
        seg = info if isinstance(info, dict) else {}
        blocks = h3ff.face_split_blocks(
            seg.get("seg_sizes"), seg.get("effective_context"),
            expect_tokens=int(T), boundaries=seg.get("boundaries"),
            decoded_frames=F_expect)
        if blocks is None:
            blocks = h3ff.token_blocks(int(T), 22)

        # ---- 1) 分块解码 + 逐块检测 ----
        frames = np.empty((F_expect, H, W, 3), dtype=np.uint8)
        per_frame = [None] * F_expect
        opt = {"conf": float(conf), "model": model_path}
        for bi, (dk0, dk1, kp0, kp1) in enumerate(blocks):
            blk = h3ff.decode_probe_frames(vae, v_lat, range(dk0, dk1))
            off = kp0 - h3ff._pixels_for_tokens(dk0)
            if int(blk.shape[0]) < off + (kp1 - kp0):
                raise RuntimeError(f"[H3-FaceCut] block {bi + 1}: decoded {int(blk.shape[0])} frames "
                                   f"< expected {off + kp1 - kp0}\n"
                                   f"[H3-FaceCut] 块 {bi + 1}: 解码帧数不足")
            seg_px = blk[off: off + (kp1 - kp0)]
            frames[kp0:kp1] = seg_px
            per_frame[kp0:kp1] = h3ff.detect_faces(seg_px, opt)
            del blk, seg_px
        n_hit = sum(1 for b in per_frame if b)
        print(f"[H3-FaceCut] detected {n_hit}/{F_expect} frames\n"
              f"[H3-FaceCut] 检出 {n_hit}/{F_expect} 帧")

        # ---- 2) 出现区间 (gap≤gap_tol 连续) + 区间内线性填补 ----
        det_idx = [i for i, b in enumerate(per_frame) if b]
        intervals = []
        for i in det_idx:
            if intervals and i - intervals[-1][1] - 1 <= int(gap_tol):
                intervals[-1][1] = i
                intervals[-1][2].append(i)
            else:
                intervals.append([i, i, [i]])
        filled = [None] * F_expect
        for a, b, dets in intervals:
            for j in dets:
                filled[j] = list(max(per_frame[j], key=lambda x: x[4])[:4])
            for j in range(a, b + 1):
                if filled[j] is None:
                    p = max(d for d in dets if d < j)
                    n = min(d for d in dets if d > j)
                    r = (j - p) / float(n - p)
                    filled[j] = [x + (y - x) * r for x, y in zip(filled[p], filled[n])]

        # ---- 3) 尺度突变切分 → 子轨 (<5 帧碎片能并则并, 否则丢弃) ----
        ratio = float(scale_split)
        pieces = []
        for a, b, _dets in intervals:
            cur, sizes = [a], [_size(filled[a])]
            for i in range(a + 1, b + 1):
                s_i = _size(filled[i])
                med = float(np.median(sizes))
                if med > 0 and (s_i > med * ratio or s_i < med / ratio):
                    pieces.append(cur)
                    cur, sizes = [i], [s_i]
                else:
                    cur.append(i)
                    sizes.append(s_i)
            pieces.append(cur)
        merged = []
        for p in pieces:
            if len(p) >= 5:
                merged.append(p)
                continue
            if merged:
                med_prev = float(np.median([_size(filled[j]) for j in merged[-1]]))
                if med_prev > 0 and _size(filled[p[0]]) <= med_prev * ratio:
                    merged[-1].extend(p)
                    continue
            print(f"[H3-FaceCut] drop tiny face run ({len(p)} frames) at [{p[0]},{p[-1]}]\n"
                  f"[H3-FaceCut] 丢弃过短人脸片段 ({len(p)} 帧) @ [{p[0]},{p[-1]})")

        # ---- 4) 每条子轨: 独立平滑 / S_i / skip / 17n+5 吸附 / 裁剪 / 归一化 res² ----
        expand_f = max(float(expand) / 100.0, 0.10)
        fps_eff = max(1.0, F_expect * h3ff.AUDIO_LATENTS_PER_SEC
                      / max(int(a_lat.shape[-1]), 1))
        res = int(res)
        subtracks, crop_parts, cursor = [], [], 0
        for si, idxs in enumerate(merged):
            boxes = _median_smooth([filled[j] for j in idxs])
            med = float(np.median([_size(b) for b in boxes]))
            S = max(64, (min(int(round(med * (1.0 + expand_f))), 1024) // 16) * 16)
            f0_all, f1_all = int(idxs[0]), int(idxs[-1]) + 1
            if med >= res * float(skip_ratio):
                print(f"[H3-FaceCut] subtrack {si + 1} [{f0_all},{f1_all}) skipped: face {med:.0f}px >= "
                      f"{float(skip_ratio):.2f}×res({res}) — already large enough\n"
                      f"[H3-FaceCut] 子轨 {si + 1} [{f0_all},{f1_all}) 跳过: 脸 {med:.0f}px ≥ "
                      f"{float(skip_ratio):.2f}×res({res}) — 已够大, 无需重采样")
                subtracks.append({"f0": f0_all, "f1": f1_all, "S": int(S),
                                  "face_med": round(med, 1), "skip": True,
                                  "centers": [], "crop_off": None})
                continue
            K = len(idxs)
            K_g = ((K - 5) // 17) * 17 + 5
            if K_g < 5:
                print(f"[H3-FaceCut] subtrack {si + 1} too short ({K} frames), dropped\n"
                      f"[H3-FaceCut] 子轨 {si + 1} 过短 ({K} 帧), 丢弃")
                continue
            if K != K_g:
                print(f"[H3-FaceCut] subtrack {si + 1}: {K} -> {K_g} frames (17n+5 grid, tail trimmed)\n"
                      f"[H3-FaceCut] 子轨 {si + 1}: {K} → {K_g} 帧 (17n+5 网格, 尾部裁剪)")
            use = idxs[:K_g]
            half = S / 2.0
            centers, crops = [], []
            for j, (x1, y1, x2, y2) in zip(use, boxes):
                cx = min(max((x1 + x2) * 0.5, half), W - half)
                cy = min(max((y1 + y2) * 0.5, half), H - half)
                ix1 = max(0, min(int(round(cx - half)), W - S))
                iy1 = max(0, min(int(round(cy - half)), H - S))
                centers.append([ix1 + half, iy1 + half])
                crops.append(frames[j][iy1:iy1 + S, ix1:ix1 + S])
            crop_t = torch.from_numpy(np.stack(crops)).to(torch.float32) / 255.0  # [K,S,S,C]
            if (S, S) != (res, res):
                crop_t = comfy.utils.common_upscale(
                    crop_t.movedim(-1, 1), res, res, "lanczos", "disabled").movedim(1, -1)
            subtracks.append({"f0": int(use[0]), "f1": int(use[-1]) + 1, "S": int(S),
                              "face_med": round(med, 1), "skip": False,
                              "centers": centers, "crop_off": cursor})
            crop_parts.append(crop_t.contiguous())
            cursor += K_g
            print(f"[H3-FaceCut] subtrack {si + 1} [{f0_all},{f1_all}) S={S}px -> {res}² "
                  f"({float(res) / S:.2f}x) rows [{cursor - K_g},{cursor})\n"
                  f"[H3-FaceCut] 子轨 {si + 1} [{f0_all},{f1_all}) S={S}px → {res}² "
                  f"(放大 {float(res) / S:.2f}x) 行 [{cursor - K_g},{cursor})")

        if crop_parts:
            crop_images = torch.cat(crop_parts, dim=0).contiguous()
        else:
            crop_images = torch.zeros(1, res, res, 3)
        n_sampled = sum(1 for st in subtracks if not st["skip"])
        pack = {"version": 5, "a_lat": a_lat, "subtracks": subtracks,
                "fps_eff": float(fps_eff), "n_crop_rows": int(cursor),
                "meta": {"T": int(T), "LH": int(LH), "LW": int(LW), "W": int(W), "H": int(H),
                         "res": res, "n_hit": int(n_hit), "n_frames": F_expect,
                         "n_subtracks": len(subtracks), "n_sampled": n_sampled,
                         "gap_tol": int(gap_tol), "scale_split": float(scale_split),
                         "skip_ratio": float(skip_ratio)}}
        info_str = [(st["f0"], st["f1"], st["S"]) for st in subtracks if not st["skip"]]
        print(f"\033[35m[H3-FaceCut] {len(subtracks)} subtracks ({n_sampled} sampled): "
              f"(f0,f1,S)={info_str}, crop_images {tuple(crop_images.shape)}\n"
              f"[H3-FaceCut] 共 {len(subtracks)} 条子轨 ({n_sampled} 条待采样): "
              f"(f0,f1,S)={info_str}, 裁剪输出 {tuple(crop_images.shape)}\033[0m")
        return (crop_images, pack)
