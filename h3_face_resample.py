"""h3_face_resample.py — 修脸第 2 步: 画布重采样

当前版本要点:
- 两种接线模式:
    集成 (info 接主采样): 本地 model/vae/clip/提示词/fps/parameter 忽略
      (唯一例外: parameter.shot_prompts); h3_runtime / seg_sizes /
      boundaries / segment_prompts / decoded_frames 全部取自 info。
    独立 (info 留空, 修普通视频): model/vae/clip/采样器取本节点端口,
      提示词/fps/分段取 parameter。
- 网格契约: 合法编码长度 = 17n+5 (5,22,39,56,73,90,...)。任何编码长度都一次
  补齐到最近合法长度 (补帧=重复末帧, 解码后裁掉; 音频账本同步补齐, 时间跨度
  一致); 补帧上限 _PAD_CAP=21 (覆盖最保守网格下最大间隙 16), 绝不因网格约束
  新增采样段/丢帧。解码后裁掉 ctx 头与补帧尾, 每块恰好贡献 (b1-b0) 行画布。
- 分块: 集成模式按主采样实际段边界切块 (与 _pick_prompt 的提示词映射同源);
  独立模式按 parameter.chunk_frames/context_frames 切块。同身份连续子轨合并
  为一条采样序列; 跨镜头单元强制断开。
- 块间续接: ctx 头锚定上一块尾部 (与主采样 _copy_overlap_tail + noise_mask=0
  同逻辑); noise_mask 同时处理 ctx 头与 lock_audio/audio_drive (音频区不重
  采样)。单脸模式下窗口跳变过大时该边界不锚定 (防换人)。
- 身份锚: 优先 Face_Cut 已验证的 ref_row/ref_image (干净参考: 源帧按锚定窗口
  重采样, 未经 pre_blur/SR); 旧 pack 无 ref_row 时回退纯清晰度选锚并警告。
  注入为 <Picture N> 参考; 与 face_prompt、身份固定噪声种子共同构成抗抖三件套。
- 缓存键 (rmeta): rows_hash / a_lat_hash / 块种子 / σ阶梯 / 提示词哈希 /
  参考素材 conditions_hash / sampler_tag / plan_hash / chunk/ctx 预算 /
  model_fp / vae_fp / clip_fp / noise_v / prompt_v / identity_ref_fp;
  独立模式额外入自身图指纹 graph_fp (覆盖本地权重链上注意力后端等挂载节点
  的控件变化)。任一敏感键变更 → 旧缓存自动失效。
- 音频账本: pack.a_lat (latent 模式) > audio 端口 (images 模式) > 静音占位。
- 文本条件缓存: 同 prompt + 同参考素材 + 同 CLIP 权重跳过 Qwen3-VL 编码;
  键含像素内容而非仅数量 (换参考图不会假命中)。
- 输入行/参考行统一 fp16 量化 (.to(fp16).float()): 与 Face_Cut v20.4 输出
  位精确一致 → rows_hash 不因缓存命中与否漂移。
- 输出画布行 fp16 驻留内存 (RAM 减半); 逐子轨色彩匹配 (Reinhard) 在缓存加载
  之后执行 — 切换 color_match 不会使缓存失效。
- VAE 解码 OOM 回退: 清缓存重试 → tiled 解码 (慢但优于崩溃)。
- 日志收编: 常规运行只保留入口摘要/出口摘要/警告/缓存命中; 逐身份/逐块/逐
  子轨细节走 h3ff.vlog (h3_facefix.py 顶部 _VERBOSE=True 打开)。
"""
import torch
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.nested_tensor
import latent_preview
import os
import re
import hashlib
import folder_paths

try:
    from . import latent_cache
except ImportError:
    import latent_cache
from comfy_api.latest import io
try:
    from . import h3_conditioning
    from . import h3_facefix as h3ff
    from . import h3_sampler
    from . import h3_patches
    from . import h3_utils
except ImportError:
    import h3_conditioning
    import h3_facefix as h3ff
    import h3_sampler
    import h3_patches
    import h3_utils

CFG = 1.0

# 合法编码长度 = 17n+5 (5,22,39,56,73,90,...) — 流水线既有路径一致使用并验证过的唯一契约。
_LEGAL_MOD17 = (5,)
_PAD_CAP = 21


def _legal_enc_len(L):
    """≥L 的最小合法编码长度 (17n+5 契约, 重复末帧补齐, 解码后裁掉)。最大补 16 帧。
    闭式: L=5 → 5; 否则 5 + ceil((L-5)/17)*17。"""
    L = max(5, int(L))
    q, r = divmod(L - 5, 17)
    legal = 5 + (q + (1 if r > 0 else 0)) * 17
    if legal - L > _PAD_CAP:  
        raise RuntimeError(f"[H3-FaceResample] grid pad {legal - L} > cap {_PAD_CAP} (L={L})\n"
                           f"[H3-FaceResample] 网格补帧超过上限 (L={L}) — 请检查 _LEGAL_MOD17/_PAD_CAP")
    return legal


def _plan_blocks(K, chunk_frames, context_frames):
    """子轨内分块 [(b0, b1, ctx)] — 与主采样分段账本同源 (h3_utils.compute_chunks)。
    chunk_frames/context_frames 必传, 本函数不携带任何帧数常量:
    集成模式 ← info.seg_sizes 的最大段长与 effective_context (主采样实际分段);
    独立模式 ← parameter 节点的 chunk_frames / context_frames。
    每块编码长 = (b1-b0)+ctx, 执行期补齐 17n+5 (pad≤16), 解码后裁掉 ctx 头与补帧尾;
    每块恰好贡献 (b1-b0) 行画布。"""
    K = int(K)
    if K < 1:
        return []
    chunks, _ = h3_utils.compute_chunks(K, int(chunk_frames), int(context_frames))
    if not chunks:
        return []
    ctx = max(0, int(context_frames))
    blocks, keep = [], 0
    for j, (s, e) in enumerate(chunks):
        new = (int(e) - int(s)) if j == 0 else (int(e) - int(s)) - ctx
        if new <= 0:
            continue
        new = min(new, K - keep)
        if new <= 0:
            break
        blocks.append((keep, keep + new, 0 if j == 0 else min(ctx, keep)))
        keep += new
        if keep >= K:
            break
    return blocks

def _plan_blocks_segments(K, f0, boundaries, total, ctx_budget):
    """集成模式: 按主采样实际分段结构切块 (块边界 = 主采样段边界)。
    boundaries: 主采样 seam_info.boundaries (各段新增内容首帧号, 像素帧, 与本节点全局帧同源);
    total: 主采样 decoded_frames (调用方用检测帧数兜底)。
    身份行序列 [f0, f0+K) 与切点区间求交集 → 每块 (b0, b1, ctx);
    块长即主采样段新增长 (任意值), 网格合法性由编码期补帧兜底 (补 ≤ 21 帧)。
    块边界与 _pick_prompt 的 boundaries 映射同源 → 每块提示词精确对应主采样同一段。
    """
    cuts = sorted({0} | {int(b) for b in (boundaries or []) if 0 < int(b) < int(total)})
    cuts.append(max(int(total), f0 + K))   
    blocks, keep = [], 0
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        b0 = max(s, f0) - f0
        b1 = min(e, f0 + K) - f0
        if b1 - b0 <= 0:
            continue
        ctx = 0 if not blocks else min(int(ctx_budget), b0)
        blocks.append((b0, b1, ctx))
        keep += b1 - b0
        if keep >= K:
            break
    if keep < K and blocks:              
        b0, b1, ctx = blocks[-1]
        blocks[-1] = (b0, K, ctx)
    elif not blocks:
        blocks = [(0, K, 0)]
    return blocks

def _pick_prompt(fc, boundaries, seg_prompts, long_prompt):
    if not seg_prompts:
        return long_prompt, "global prompt / 全局提示词"
    idx = min(sum(1 for b in (boundaries or []) if b <= fc), len(seg_prompts) - 1)
    return seg_prompts[idx], f"segment {idx + 1} prompt / 段{idx + 1}提示词"

def _normalize_sigmas(sigmas, device):
    if sigmas is None:
        raise ValueError("[H3-FaceResample] sigmas port not connected\n"
                         "[H3-FaceResample] sigmas 端口未连接 — 请接调度器输出")
    s = sigmas.detach().clone().flatten().to(device=device, dtype=torch.float32)
    if s.numel() < 2:
        raise ValueError(f"[H3-FaceResample] sigmas needs >= 2 values, got {s.numel()}"
                         f"\n[H3-FaceResample] sigmas 至少 2 个值, 收到 {s.numel()} 个")
    if not bool(torch.all(s[1:] <= s[:-1])):
        raise ValueError(f"[H3-FaceResample] sigmas must be non-increasing: "
                         f"{[round(float(x), 4) for x in s]}"
                         f"\n[H3-FaceResample] sigmas 必须单调不增: "
                         f"{[round(float(x), 4) for x in s]}")
    if float(s[0]) <= 0.0:
        raise ValueError("[H3-FaceResample] first sigma must be > 0"
                         "\n[H3-FaceResample] 首 σ 必须 > 0")
    if float(s[-1]) != 0.0:
        s = torch.cat([s, torch.zeros(1, device=device)])
        h3ff.log(f"[H3-FaceResample] last sigma != 0, appended 0: {[round(float(x), 4) for x in s]}\n"
                 f"[H3-FaceResample] sigmas 末值非 0, 自动补 0: {[round(float(x), 4) for x in s]}")
    return s

def _color_match_rows(rows, ref_rows, eps=1e-4):
    """逐子轨色彩匹配 (Reinhard 统计): 把 rows 的逐通道均值/方差迁移到 ref_rows 水平。
    rows/ref_rows: [K,H,W,3] float 0..1。整条子轨一组统计 → 帧间变换恒定, 不引入时间闪烁。"""
    mx = rows.mean(dim=(0, 1, 2), keepdim=True)
    sx = rows.std(dim=(0, 1, 2), keepdim=True, correction=0)
    mr = ref_rows.mean(dim=(0, 1, 2), keepdim=True)
    sr = ref_rows.std(dim=(0, 1, 2), keepdim=True, correction=0)
    out = (rows - mx) * (sr + eps) / (sx + eps) + mr
    return out.clamp(0.0, 1.0)

def _row_sharpness(row):
    """单行 [H,W,3] → 原始拉普拉斯方差 (256² area 降采样, 与 FaceCut 锚定同度量)。"""
    g = row.detach().float().mean(dim=-1)
    if max(int(g.shape[-2]), int(g.shape[-1])) > 256:
        g = torch.nn.functional.interpolate(g[None, None], size=(256, 256), mode="area")[0, 0]
    lap = (g[:-2, :-2] + g[:-2, 2:] + g[2:, :-2] + g[2:, 2:] - 4.0 * g[1:-1, 1:-1])
    return float(lap.var())


def _text_cond_fingerprint(prompt, ri, rv, ra):
    """块文本条件的内容指纹 — text-cond 缓存的键。
    Qwen3-VL 会把图像/视频的像素内容嵌入文本条件，键必须含像素内容而非仅数量，
    否则同 prompt 换参考图会假命中、复用旧图的嵌入；音频不经过文本编码器，只记数量。"""
    h = hashlib.md5()
    h.update(prompt.encode("utf-8"))
    for img in (ri or []):
        p = img.get("pixel")
        if torch.is_tensor(p):
            h.update(str(tuple(p.shape)).encode())
            h.update(p.detach().reshape(-1)[:: max(1, p.numel() // 2048)][:2048].float().cpu().numpy().tobytes())
    for vid in (rv or []):
        h.update(("vid_a" if vid.get("audio_latent") is not None else "vid").encode())
        p = vid.get("pixel")
        if torch.is_tensor(p):
            h.update(str(tuple(p.shape)).encode())
            h.update(p.detach().reshape(-1)[:: max(1, p.numel() // 2048)][:2048].float().cpu().numpy().tobytes())
    h.update(f"a:{len(ra or [])}".encode())
    return h.hexdigest()


# ---- 修脸提示词组合 + 自动身份锚 ----
_REF_TAG_RE = re.compile(r"<(?:Picture|Video|Audio)\s+\d+>", re.IGNORECASE)


def _pick_sharpest_row(rows, max_n=32):
    """从裁剪行中挑最清晰一帧 (拉普拉斯方差最大, 抽样上限 max_n) → (index, frame) 或 None。"""
    try:
        K = int(rows.shape[0])
        if K <= 0:
            return None
        n_s = min(K, max_n)
        idxs = sorted(set(int(round(i * (K - 1) / max(1, n_s - 1))) for i in range(n_s)))
        g = rows[idxs].float().mean(dim=-1)
        if max(int(g.shape[-2]), int(g.shape[-1])) > 256:
            g = torch.nn.functional.interpolate(g.unsqueeze(1), size=(256, 256), mode="area").squeeze(1)
        lap = (g[..., :-2, :-2] + g[..., :-2, 2:] + g[..., 2:, :-2] + g[..., 2:, 2:] - 4.0 * g[..., 1:-1, 1:-1])
        lv = lap.var(dim=(1, 2))
        best = int(torch.argmax(lv).item())
        gi = idxs[best]
        return gi, rows[gi]
    except Exception:
        return None


def _compose_face_prompt(orig_prompt, face_prompt, mode, id_tag=None):
    """修脸提示词组合器。
    - prepend: face_prompt + 原提示词 (环境上下文与参考声明全保留)
    - replace_text: face_prompt + 原提示词中的参考声明 (丢弃场景文本)
    - replace_all: 仅 face_prompt
    身份锚标签 (<Picture N>) 始终附加 (若原文本未含), 保证 identity_ref 注入的参考被
    声明-过滤管线识别并传递。face_prompt 与 id_tag 均空 → 原样返回 (旧行为)。"""
    fp = (face_prompt or "").strip()
    orig = orig_prompt or ""
    if not fp and id_tag is None:
        return orig
    if mode == "replace_all":
        text = fp if fp else orig
    elif mode == "replace_text":
        tags = list(dict.fromkeys(_REF_TAG_RE.findall(orig)))
        text = fp if fp else ""
        if tags:
            text = (text + "\n" if text else "") + " ".join(tags)
    else:  # prepend
        text = (fp + "\n" + orig) if fp else orig
    if id_tag and id_tag not in text:
        text = text + f"\n{id_tag}: identity reference of the person in this video"
    return text


# ================= 独立模式支持 (修复普通视频, 不经过主采样节点) =================
_SAMPLERS = ["euler", "euler_ancestral", "euler_cfg_pp", "res_multistep", "res_multistep_cfg_pp",
             "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_3m_sde",
             "uni_pc", "uni_pc_bh2", "ddpm", "lms", "heun", "dpm_2", "dpm_2_ancestral"]
_SCHEDULERS = ["simple", "normal", "karras", "exponential", "sgm_uniform", "beta", "linear_quadratic"]

def _autogrow_to_list(ag_dict, prefix, max_count):
    if not ag_dict:
        return []
    return [ag_dict[f"{prefix}{i}"] for i in range(max_count) if ag_dict.get(f"{prefix}{i}") is not None]


def _resolve_audio_latent(raw):
    """从 audio 端口提取音频 latent (时间维在最后一维, 与 pack.a_lat 同约定)。
    兼容: 裸张量 [.., T] / {'samples': ...} / NestedTensor(取 dim<5 的非视频分量)。"""
    if raw is None:
        return None
    if hasattr(raw, "is_nested") and getattr(raw, "is_nested"):
        for t in raw.unbind():
            if torch.is_tensor(t) and t.dim() < 5:
                return t
        return None
    if isinstance(raw, dict):
        return _resolve_audio_latent(raw.get("samples"))
    if isinstance(raw, (tuple, list)):
        for v in raw:
            r = _resolve_audio_latent(v)
            if r is not None:
                return r
        return None
    if torch.is_tensor(raw):
        return raw if raw.dim() < 5 else None
    return None

def _merge_runtime(rt_info, parameter, model, vae, audio_vae, clip, sampler_name,
                   scheduler, sampler_obj, seed, ref_images):
    """构建 h3_runtime (prompt/fps 本地端口已删除, 一律取自 parameter 或 info)。
    集成模式 (rt_info 非空): info 的非 None 键覆盖一切 
    (例外: parameter.shot_prompts; ref_images/ref_videos 由 execute 层的本地端口覆盖逻辑接管)。 
    独立模式: parameter 必选。"""
    rt_local = {
        "model": model, "vae": vae, "audio_vae": audio_vae, "clip": clip,
        "first_frame": None, "last_frame": None,
        "ref_images": _autogrow_to_list(ref_images, "ref_image_", 10),
        "ref_videos": [], "ref_audios": [], "drive_audio": None,
        "long_prompt": "", "prompt_format": "official", "clip_mode": "global",
        "clip_tag": "段1", "crop_mode": "stretch", "fps": 0,
        "lock_audio": True, "audio_drive": False, "ref_sync_mode": "global",
        "chunk_frames": 0, "context_frames": 0, "steps": 0, "cfg": 1.0, "denoise": 1.0,
        "sigmas": None, "sampler_name": str(sampler_name), "scheduler": str(scheduler),
        "seed": int(seed), "sampler_obj": sampler_obj,
    }
    if rt_info:
        return {**rt_local, **{k: v for k, v in rt_info.items() if v is not None}}
    p = parameter if isinstance(parameter, dict) else {}
    if not p:
        raise ValueError(
            "[H3-FaceResample] standalone mode requires the 'parameter' port "
            "(Minimax_H3_AutoContext_parameter) — connect it, or connect 'info' for "
            "integrated mode\n"
            "[H3-FaceResample] 独立模式必须连接 parameter 端口 "
            "(Minimax_H3_AutoContext_parameter); 或接入 info 走集成模式")
    _ints = ("fps", "chunk_frames", "context_frames")
    _bools = ("lock_audio", "audio_drive")
    for k in ("long_prompt", "prompt_format", "clip_mode", "clip_tag", "crop_mode",
              "ref_sync_mode", *_ints, *_bools):
        v = p.get(k)
        if v in (None, ""):
            continue
        if k in _ints:
            rt_local[k] = int(v)
        elif k in _bools:
            rt_local[k] = bool(v)
        else:
            rt_local[k] = str(v)
    return rt_local

def _make_standalone_prompt_fn(long_prompt, clip_mode, clip_tag, fmt, fps, n_frames):
    """独立模式: 用 h3_utils 的调度器把提示词映射到块。"""
    lp = (long_prompt or "").strip()
    if not lp:
        return lambda g0, g1: ("", "empty prompt")
    fps = max(1, int(fps))
    total_sec = max(0.1, float(max(1, int(n_frames))) / fps)
    if str(clip_mode) == "Clip_Tag":
        ts = h3_utils.build_tag_schedule(lp, clip_tag, default_seconds=total_sec)
        cum, t0 = [], 0.0
        for txt, dur in ts["segments"]:
            cum.append((t0, t0 + float(dur), txt, float(dur)))
            t0 += float(dur)
        span = max(t0, 1e-3)
        scale = span / total_sec  

        def _tag_fn(g0, g1):
            t = ((g0 + g1) * 0.5 / fps) * scale
            hit = cum[-1] if cum else (0.0, span, lp, span)
            for c in cum:
                if c[0] <= t < c[1]:
                    hit = c
                    break
            idx = cum.index(hit) + 1 if hit in cum else 1
            return (h3_utils.render_tag_segment(hit[2], hit[3], fmt, ts.get("prefix", "")),
                    f"clip_tag seg{idx}")
        return _tag_fn
    mode = str(clip_mode) if str(clip_mode) in ("timeline", "sequential", "global") else "global"
    sched = h3_utils.build_prompt_schedule(lp, total_sec, mode=mode)
    used = str(sched.get("mode") or mode)
    if used == "global":
        return lambda g0, g1: (lp, "global prompt")

    def _win_fn(g0, g1):
        return (h3_utils.compose_window_prompt(sched, g0 / fps, g1 / fps, fmt=fmt),
                f"{used} window [{g0},{g1})")
    return _win_fn

class H3FaceResample(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3FaceResample",
            display_name="Minimax_H3_Face_Resample",
            category="MinimaxH3_AutoContext/FaceFix",
            description=("Face fix step 2,Two wiring modes: "
                         "INTEGRATED (info from main sampler; local model/vae/clip/parameter "
                         "ignored, except parameter.shot_prompts) or STANDALONE (fix ordinary videos: "
                         "leave info empty, connect model/vae/clip + audio + parameter).\n"
                         "修脸第 2 步:"
                         "两种接线: 集成 (info 接主采样, 本地 model/vae/clip/提示词/fps/parameter 忽略, "
                         "parameter.shot_prompts 除外) 或独立 (修普通视频: info 留空, 接 model/vae/clip "
                         "+ audio + parameter 或提示词)。"),
            inputs=[
                io.Image.Input("crop_images", tooltip="crop_images from H3FaceCut (uniform res^2, row order = subtrack order, "
                               "ALL frames kept)\n来自 H3FaceCut 的 crop_images (统一 res², 行序=子轨序, 全帧保留)"),
                io.Dict.Input("face_pack", tooltip="face_pack from H3FaceCut\n来自 H3FaceCut 的 face_pack"),

                io.Dict.Input("parameter", tooltip="Output of Minimax_H3_AutoContext_parameter (REQUIRED in both wiring modes). "
                              "Ignored keys: width/height/total_frames/video_guide. \n"
                              "Minimax_H3_AutoContext_parameter 节点的输出 (两种接线模式均必选)。"
                              "忽略键: width/height/total_frames/video_guide。" ),
                io.Sigmas.Input("sigmas", tooltip="Refinement sigma schedule (decreasing, last=0)\n精修 σ 阶梯 (单调递减, 末值0)"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                             tooltip="Block sampling seed. In standalone mode also the conditioning seed\n"
                                     "块采样种子。独立模式下同时作为 conditioning 种子"),
                io.Dict.Input("info", optional=True,
                    tooltip="INTEGRATED mode: connect the main sampler's info (h3_runtime + "
                            "seg_sizes/boundaries/segment_prompts). STANDALONE mode: leave EMPTY. "
                            "Do NOT wire Face_Cut's shot_info here — it is not consumed; "
                            "shot_info belongs to the parameter node.\n"
                            "集成模式: 接主采样节点的 info。独立模式: 留空。"
                            "不要把 Face_Cut 的 shot_info 接到这里 — 本节点不消费它, "
                            "shot_info 应接 parameter 节点的 shot_info 端口"),

                io.Audio.Input("audio", optional=True, tooltip="Audio ledger for images mode (pack.a_lat is empty there): connect the source video's "
                               "audio; encoded internally with audio_vae (lip sync). pack.a_lat (latent mode) wins when "
                               "present; silent placeholder + warning if neither exists\n"
                               "images 模式的音频账本 (此时 pack 内无 a_lat): 接源视频音频, 内部用 audio_vae 编码 "
                               "(口型同步)。latent 模式的 pack.a_lat 优先; 两者皆无时用静音占位并警告"),
                io.Model.Input("model", optional=True, tooltip="STANDALONE mode only: diffusion model for resampling. Ignored when info carries h3_runtime\n"
                               "仅独立模式: 重采样用的扩散模型。info 携带 h3_runtime 时被忽略"),
                io.Vae.Input("vae", optional=True, tooltip="STANDALONE mode only: video VAE for canvas encode/decode\n"
                             "仅独立模式: 画布编码/解码的视频 VAE"),
                io.Vae.Input("audio_vae", optional=True, tooltip="STANDALONE mode only: audio VAE, encodes the audio port (lip sync)\n"
                             "仅独立模式: 音频 VAE, 编码 audio 端口 (口型同步)"),
                io.Clip.Input("clip", optional=True, tooltip="STANDALONE mode only: text encoder for block prompts\n"
                              "仅独立模式: 块提示词的文本编码器"),
                io.Combo.Input("sampler_name", options=_SAMPLERS, default="euler", tooltip="STANDALONE mode only\n仅独立模式使用"),
                io.Combo.Input("scheduler", options=_SCHEDULERS, default="simple", tooltip="STANDALONE mode only\n仅独立模式使用"),
                io.Sampler.Input("sampler", optional=True, tooltip="STANDALONE mode only: external SAMPLER, overrides sampler_name/scheduler\n"
                                 "仅独立模式: 外部采样器, 接入后覆盖 sampler_name/scheduler"),
                io.Autogrow.Input("ref_images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_image", tooltip="Reference image (referenced in prompt as <Picture N> — declared-"
                                         "only: refs are passed to a block only if its prompt mentions them)\n"
                                         "参考图 (提示词中用 <Picture N> 引用 — 声明才引用: 仅当块提示词提到时才传递)"),
                    prefix="ref_image_", min=0, max=9)),
                
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_video", tooltip="Reference video frames (24fps, 2-15s). Consumed only when the "
                    "block prompt declares <Video N>. Local port OVERRIDES info-passed reference videos when "
                    "non-empty; in segmented mode each block takes only its matching time slice (lip sync via "
                    "the paired audio)\n"
                    "参考视频帧 (24fps, 2-15s)。仅当块提示词声明 <Video N> 时传入。本地端口非空时优先于 "
                    "info 传入的参考视频；segmented 模式下每个采样块只取对应时间切片 (配对音轨驱动口型)"),
                    prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_video_audio", tooltip="Paired audio track of the reference video with the same "
                    "index (lip sync). Leave empty for a silent reference video\n"
                    "同编号参考视频的配对音轨 (口型同步)。不接则该参考视频作为无声参考传入"),
                    prefix="ref_video_audio_", min=0, max=3)),
                
                io.String.Input(
                    "face_prompt",
                    multiline=True,
                    default=(
                        "Enhance this video with sharp, crisp details while preserving a natural photorealistic appearance. "
                    ),
                    tooltip="Face-repair prompt, composed with the original per-segment prompt according to "
                            "prompt_mode. Scene prompts from the parameter node are written for whole-shot "
                            "generation and often say nothing useful about faces — at high sigma they actively "
                            "mislead the redraw. Ref declarations (<Picture N> etc.) are always preserved per "
                            "prompt_mode\n"
                            "修脸专用提示词, 按 prompt_mode 与原分段提示词组合。parameter 的场景提示词是为整镜头"
                            "生成写的, 往往没有面部描述 — 高 σ 重绘时甚至会误导结果。参考声明 "
                            "(<Picture N> 等) 按 prompt_mode 规则保留"
                ),
                io.Combo.Input("prompt_mode", options=["prepend", "replace_text", "replace_all"], default="prepend",
                               tooltip="How face_prompt combines with the original block prompt:\n"
                                       "- prepend: face_prompt + original (keeps environment context & refs)\n"
                                       "- replace_text: face_prompt + ONLY the ref tags of the original (drops scene "
                                       "text; best for high-sigma structural repair)\n"
                                       "- replace_all: face_prompt alone (refs dropped unless you declare them)\n"
                                       "face_prompt 与原分段提示词的组合方式:\n"
                                       "- prepend: 修脸词 + 原提示词 (保留环境上下文与参考声明)\n"
                                       "- replace_text: 修脸词 + 仅保留原提示词中的参考声明 (丢弃场景文本, "
                                       "高 σ 结构修复首选)\n"
                                       "- replace_all: 仅修脸词 (参考素材不再传递, 除非自行声明)"),
                io.Boolean.Input("identity_ref", default=True,
                                 tooltip="Auto identity anchor: picks the sharpest crop row of each identity and passes "
                                 "it as a <Picture N> reference for that identity's redraw — keeps identity and "
                                 "appearance consistent across frames/blocks at high sigma (the main anti-jitter "
                                 "lever, together with face_prompt and the fixed per-identity noise seed)\n"
                                 "自动身份锚: 为每条身份挑选最清晰的一帧裁剪, 以 <Picture N> 参考图形式注入该身份的"
                                 "重采样 — 高 σ 下保持跨帧/跨块的身份与外观一致 (与修脸提示词、身份固定噪声种子"
                                 "共同构成抗抖三件套)"),
                io.Boolean.Input("color_match", default=True, tooltip="Reinhard-style color match of each resampled subtrack to its source crop rows "
                                 "so the blended face matches its surroundings. Applied AFTER cache load — "
                                 "toggling does NOT invalidate cache\n"
                                 "将每条重采样子轨与对应裁剪行做 Reinhard 式色彩匹配 (逐通道均值/方差, 按子轨统计)。"
                                 "在缓存加载之后执行 — 切换开关不会使缓存失效"),
                io.Boolean.Input("enable_cache", default=True, tooltip="Cache per-subtrack canvas rows (keyed by crop-rows hash + block seeds + "
                                 "sigmas + prompts + refs + sampler + fps + lock_audio + model/VAE/CLIP "
                                 "fingerprints). Identical reruns skip sampling\n"
                                 "按子轨缓存画布行 (键: 裁剪行指纹+块种子+σ+提示词+参考素材+采样器+fps+lock_audio"
                                 "+模型/VAE/CLIP 指纹)。一致的重复运行将跳过采样"),
                io.Boolean.Input("clear_cache", default=False, tooltip="Delete this node's cache directory before running\n运行前删除本节点的缓存目录"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Dict.Output(display_name="bbox"),
                io.Image.Output(display_name="identity_refs"),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
        )

    @classmethod
    def execute(cls, crop_images, face_pack, sigmas, seed=0, info=None, enable_cache=True,
                clear_cache=False, color_match=True, audio=None, parameter=None,
                model=None, vae=None, audio_vae=None, clip=None, sampler_name="euler",
                scheduler="simple", sampler=None, ref_images=None,
                ref_videos=None, ref_video_audios=None,
                face_prompt="", prompt_mode="prepend",
                identity_ref=True) -> io.NodeOutput:
        h3_patches.apply_patches()  
        pack = face_pack or {}
        if int(pack.get("version") or 0) not in (5, 6, 7):
            raise ValueError("[H3-FaceResample] pack version mismatch — rerun H3FaceCut\n"
                             "[H3-FaceResample] bbox 版本不符 — 请重跑 Face_Cut")
        a_lat = pack.get("a_lat")
        # ---- 本地 ref_video/ref_video_audio Autogrow 端口组装 ----
        ref_video_audios = ref_video_audios or {}
        ref_video_list = []
        for i in range(4):
            vval = (ref_videos or {}).get(f"ref_video_{i}")
            if vval is None:
                continue
            ref_video_list.append({"video": vval,
                                   "audio": ref_video_audios.get(f"ref_video_audio_{i}")})
        # ---- 运行时: info (集成) > parameter > 本地端口 (独立兜底) ----
        rt_info = {}
        if isinstance(info, dict) and isinstance(info.get("h3_runtime"), dict):
            rt_info = info["h3_runtime"]
        rt = _merge_runtime(rt_info, parameter, model, vae, audio_vae, clip, sampler_name, scheduler, sampler, seed, ref_images)
        # if not rt.get("ref_images") and ref_images:
        if ref_images:
            rt["ref_images"] = _autogrow_to_list(ref_images, "ref_image_", 10)
        # ---- 本地 ref_video 端口优先级高于 info 传入: 端口非空即覆盖, 为空时沿用 info ----
        if ref_video_list:
            rt["ref_videos"] = ref_video_list
        if rt_info:
            h3ff.log("\033[33m[H3-FaceResample] integrated mode: local model/vae/clip/prompt/fps/"
                     "parameter ignored (exceptions: parameter.shot_prompts + ref_image/ref_video ports), "
                     "h3_runtime from info\n"
                     "[H3-FaceResample] 集成模式: 本地 model/vae/clip/提示词/fps/parameter 已忽略 "
                     "(例外: parameter.shot_prompts 与 ref_image/ref_video 本地端口), "
                     "以 info 的 h3_runtime 为准\033[0m")
        else:
            h3ff.log("[H3-FaceResample] standalone mode: model/vae/clip/sampler from local ports; "
                     "prompt/fps/segmentation from parameter\n"
                     "[H3-FaceResample] 独立模式: model/vae/clip/采样器取本节点端口; "
                     "提示词/fps/分段取 parameter")
            h3ff.vlog("[H3-FaceResample] parameter keys ignored (no consumer in this node): "
                      "width/height/total_frames/video_guide\n"
                      "[H3-FaceResample] parameter 中被忽略的键 (本节点无消费点): "
                      "width/height/total_frames/video_guide")

        subs_in = pack.get("subtracks") or []
        subs = [dict(st) for st in subs_in]
        todo = [st for st in subs if not st.get("skip") and st.get("crop_off") is not None]
        if not todo:
            h3ff.log("[H3-FaceResample] no samplable subtrack, empty canvas passed\n"
                     "[H3-FaceResample] 无可采样子轨, 输出空画布")
            return io.NodeOutput(torch.zeros(1, 64, 64, 3),
                                 {"version": int(pack.get("version") or 5), "subtracks": subs, "a_lat": a_lat, "meta": pack.get("meta", {})},
                                 torch.zeros(1, 64, 64, 3))   

        missing = [k for k in ("model", "vae", "clip") if rt.get(k) is None]
        if missing:
            raise ValueError(f"[H3-FaceResample] missing required inputs: {missing} — connect the "
                             f"'info' port (integrated) OR fill local model/vae/clip (standalone)\n"
                             f"[H3-FaceResample] 缺少必需输入: {missing} — 请接 info 端口 (集成模式), "
                             f"或填写本节点 model/vae/clip (独立模式)")
        model = rt["model"]
        clip = rt["clip"]
        vae = rt["vae"]
        audio_vae = rt.get("audio_vae")
        device = comfy.model_management.get_torch_device()
        res = int((pack.get("meta") or {}).get("res", 512))
        if crop_images is None or crop_images.dim() != 4 \
                or int(crop_images.shape[1]) != res or int(crop_images.shape[2]) != res:
            raise ValueError(f"[H3-FaceResample] crop_images must be [N,{res},{res},3] from H3FaceCut\n"
                             f"[H3-FaceResample] crop_images 必须是 [N,{res},{res},3] (来自 H3FaceCut)")
        if crop_images.shape[0] != int(pack.get("n_crop_rows") or 0):
            raise ValueError("[H3-FaceResample] crop_images rows do not match pack accounting\n"
                             "[H3-FaceResample] crop_images 行数与 pack 账目不符 — 请重新运行 H3FaceCut")

        boundaries = info.get("boundaries") or [] if isinstance(info, dict) else []
        seg_prompts = info.get("segment_prompts") or [] if isinstance(info, dict) else []
        long_prompt = rt.get("long_prompt") or ""
        crop_mode = rt.get("crop_mode", "stretch")
        ref_fps = int(rt.get("fps") or 0)
        if ref_fps <= 0:
            raise ValueError("[H3-FaceResample] fps missing: connect parameter (standalone) "
                             "or info (integrated) — audio slicing and prompt time mapping need it\n"
                             "[H3-FaceResample] 缺少 fps: 独立模式接 parameter, 集成模式接 info "
                             "— 音频切片与提示词时间映射依赖它")
        lock_a = bool(rt.get("lock_audio")) or bool(rt.get("audio_drive"))
        ref_sync = str(rt.get("ref_sync_mode") or "global")
        _meta = pack.get("meta") or {}
        n_frames_total = int(_meta.get("n_frames") or 0)

        # ---- 块预算三级来源 (与 Minimax_H3_AutoContext_parameter 分段逻辑同源) ----
        seg_src = info.get("seg_sizes") if isinstance(info, dict) else None
        main_boundaries = [int(b) for b in (info.get("boundaries") or [])] if isinstance(info, dict) else []
        main_decoded = int(info.get("decoded_frames") or 0) if isinstance(info, dict) else 0
        if seg_src:
            chunk_budget = max(int(s) for s in seg_src)
            ctx_budget = int(info.get("effective_context") or info.get("context_frames") or rt.get("context_frames") or 0)
            _src = (f"info real segmentation: seg_sizes={[int(s) for s in seg_src]}, "
                    f"boundaries={main_boundaries}, decoded={main_decoded}, ctx={ctx_budget}")
        else:
            chunk_budget = int((isinstance(info, dict) and info.get("chunk_frames")) or rt.get("chunk_frames") or 0)
            ctx_budget = int((isinstance(info, dict) and info.get("context_frames")) or rt.get("context_frames") or 0)
            _src = "parameter.chunk_frames/context_frames"
    
        if chunk_budget < 5 or ctx_budget < 0:
            raise ValueError("[H3-FaceResample] block budget missing — connect parameter "
                             "(standalone) or info with seg_sizes (integrated)\n"
                             "[H3-FaceResample] 缺少块预算 — 独立模式接 parameter, "
                             "集成模式接含 seg_sizes 的 info")
        h3ff.log(f"[H3-FaceResample] block budget: chunk={chunk_budget} ctx={ctx_budget} from {_src}\n"
                 f"[H3-FaceResample] 块预算: chunk={chunk_budget} ctx={ctx_budget} 来源: {_src}")

        _src_area = int(_meta.get("W") or 0) * int(_meta.get("H") or 0)
        if _src_area and res * res > _src_area:
            h3ff.warn(f"[H3-FaceResample] WARNING: canvas {res}² > main canvas {_src_area}px — "
                      f"main sampler's validated envelope no longer backs chunk={chunk_budget}; "
                      f"lower res or chunk_frames if OOM\n"
                      f"[H3-FaceResample] 警告: 画布 {res}² 大于主采样画布 — 主采样已验证的 "
                      f"chunk={chunk_budget} 包络不再背书显存, 爆显存请降 res 或 chunk_frames")

        fix_sig = _normalize_sigmas(sigmas, device)
        k = int(fix_sig.numel()) - 1
        h3ff.log(f"[H3-FaceResample] fix_sig ({k} steps), canvas {res}x{res}, "
                 f"{len(todo)}/{len(subs)} subtracks sampled (grid 17n+5, pad cap {_PAD_CAP}, "
                 f"lock_audio={lock_a}, ref_sync={ref_sync})\n"
                 f"[H3-FaceResample] fix_sig ({k} 步), 画布 {res}x{res}, "
                 f"{len(todo)}/{len(subs)} 条子轨待采样 (网格 17n+5, 补帧上限 {_PAD_CAP}, "
                 f"音频锁定={lock_a}, 参考切片={ref_sync})")

        # segmented 模式跳过全量预编码: 参考视频/音频只在各块切片时编码
        segmented_active = (ref_sync == "segmented") and bool(rt.get("ref_videos") or rt.get("ref_audios")) and n_frames_total > 0
        ri = (h3_sampler._prepare_ref_images(rt.get("ref_images"), vae, device, res, res, crop_mode) if rt.get("ref_images") else [])
        rv = (h3_sampler._prepare_ref_videos(rt.get("ref_videos"), vae, audio_vae, device, res, res, ref_fps, crop_mode, pre_encode=(not segmented_active)) if rt.get("ref_videos") else [])
        ra = (h3_sampler._prepare_ref_audios(rt.get("ref_audios"), audio_vae, device, pre_encode=(not segmented_active)) if rt.get("ref_audios") else [])


        # ---- 节点缓存 (与主采样器同款) ----
        try:
            unique_id = cls.hidden.unique_id
        except AttributeError:
            unique_id = None
        # C1-v3: 独立模式 (无 info.h3_runtime) 时, 本地 model/vae/audio_vae/clip 端口
        # 的权重链配置以自身图指纹入键 — 否则注意力后端等挂载类节点的控件变化在
        # 独立模式下不可检出 (model_fingerprint 不再指纹 object_patches/model_options)。
        _own_graph_fp = None
        if not rt_info:
            try:
                _own_graph_fp = latent_cache.graph_weight_fingerprint(
                    getattr(cls.hidden, "prompt", None), unique_id,
                    ports=("model", "vae", "audio_vae", "clip"))
            except Exception:
                _own_graph_fp = None
        if clear_cache and unique_id is not None:
            try:
                target = os.path.join(folder_paths.get_output_directory(), "cache", f"node_{unique_id}")
                if latent_cache.clear_cache_dir(target):
                    h3ff.log(f"[H3-FaceResample] 🗑️ Cache cleared: {target}\n[H3-FaceResample] 🗑️ 缓存已清除: {target}")
                else:
                    h3ff.vlog("[H3-FaceResample] ℹ️ Cache dir does not exist, nothing to clear\n[H3-FaceResample] ℹ️ 缓存目录不存在，无需清除")
            except Exception as e:
                h3ff.warn(f"[H3-FaceResample] ⚠️ Failed to clear cache: {e}\n[H3-FaceResample] ⚠️ 清除缓存失败: {e}")
        cache_dir = ""
        if enable_cache and unique_id is not None:
            try:
                cache_dir = os.path.join(folder_paths.get_output_directory(), "cache", f"node_{unique_id}")
            except Exception:
                cache_dir = ""

        def _md5_bytes(b):
            return hashlib.md5(b).hexdigest()

        # ---- 条件指纹 (参考素材 + 采样器形态), 供缓存校验 ----
        _cm = hashlib.md5()
        _cm.update((h3_sampler._compute_conditions_hash(ri, rv) if (ri or rv) else "no_ref").encode())
        for _a in ra:
            _lat = _a.get("latent")
            if _lat is not None:
                _cm.update(_lat.detach().float().cpu().numpy().tobytes())
            else:
                _w = (_a.get("audio_dict") or {}).get("waveform")
                if torch.is_tensor(_w):
                    _cm.update(latent_cache.tensor_sig(_w, 4096).encode())
        conditions_hash = _cm.hexdigest()

        _so = rt.get("sampler_obj")
        if _so is not None:
            _sfn = getattr(_so, "sampler_function", None)
            _sdesc = f"custom:{type(_so).__name__}:{getattr(_sfn, '__name__', 'anon')}"
        else:
            _sdesc = "builtin"
        sampler_tag = (f"{_sdesc}|{rt.get('sampler_name', 'euler')}|{rt.get('scheduler', 'simple')}|cfg={CFG}")
        model_fp = latent_cache.model_fingerprint(model)
        vae_fp = latent_cache.vae_fingerprint(vae)
        clip_fp = latent_cache.clip_fingerprint(clip)
        # segmented_active = (ref_sync == "segmented") and bool(rv or ra) and n_frames_total > 0

        if (ref_sync == "segmented") and (rv or ra) and not segmented_active:
            h3ff.vlog("[H3-FaceResample] ref_sync=segmented but total frame count unknown, "
                      "refs passed in full\n"
                      "[H3-FaceResample] ref_sync=segmented 但总帧数未知, 参考素材全量传递")

        # ---- 块提示词解析: shot_prompts (最高优先) > 集成 seg_prompts / 独立调度 > 全局 ----
        shot_prompts = parameter.get("shot_prompts") if isinstance(parameter, dict) else None
        if shot_prompts:
            h3ff.log(f"[H3-FaceResample] prompt source: shot_prompts x{len(shot_prompts)} "
                     f"(parameter + FaceCut shot_info) — highest priority\n"
                     f"[H3-FaceResample] 提示词来源: shot_prompts x{len(shot_prompts)} "
                     f"(parameter + FaceCut 镜头表) — 最高优先级")
        standalone_sched = None
        if not rt_info and not shot_prompts:
            standalone_sched = _make_standalone_prompt_fn(
                long_prompt, str(rt.get("clip_mode") or "global"), str(rt.get("clip_tag") or "段1"),
                str(rt.get("prompt_format") or "official"), ref_fps, n_frames_total)
            if long_prompt.strip():
                h3ff.vlog(f"[H3-FaceResample] standalone prompt: clip_mode={rt.get('clip_mode')}, "
                          f"format={rt.get('prompt_format')}\n"
                          f"[H3-FaceResample] 独立模式提示词: clip_mode={rt.get('clip_mode')}, "
                          f"格式={rt.get('prompt_format')}")

        def _prompt_lookup(g0, g1):
            if shot_prompts:
                fc = (g0 + g1) // 2
                for sp in shot_prompts:
                    if int(sp["start"]) <= fc < int(sp["end"]):
                        return sp["text"], f"shot [{sp['start']},{sp['end']})"
                last = shot_prompts[-1]
                return last["text"], f"shot fallback [{last['start']},{last['end']})"
            if standalone_sched is not None:
                return standalone_sched(g0, g1)
            return _pick_prompt((g0 + g1) // 2, boundaries, seg_prompts, long_prompt)

        # ---- 音频账本: pack 内置 (latent 模式) > audio 端口 (images 模式) > 静音占位 ----
        if a_lat is None and audio is not None:
            if isinstance(audio, dict) and audio.get("waveform") is not None:
                if audio_vae is None:
                    raise ValueError("[H3-FaceResample] audio port got a waveform but audio_vae is missing — "
                                     "connect audio_vae (standalone) or info with h3_runtime.audio_vae (integrated)\n"
                                     "[H3-FaceResample] audio 端口收到波形但缺少 audio_vae — "
                                     "独立模式请接 audio_vae, 集成模式确认 info 携带 audio_vae")
                a_lat = h3_sampler._encode_audio(audio_vae, audio, device)
            else:
                a_lat = _resolve_audio_latent(audio)
            if a_lat is not None:
                h3ff.log(f"[H3-FaceResample] audio ledger from 'audio' port, T_a={int(a_lat.shape[-1])}\n"
                         f"[H3-FaceResample] 音频账本来自 audio 端口, T_a={int(a_lat.shape[-1])}")
        if a_lat is None:
            a_t = max(1, int(round(max(1, n_frames_total) / max(1, ref_fps) * h3ff.AUDIO_LATENTS_PER_SEC)))
            a_lat = torch.zeros(1, int(getattr(h3_sampler, "AUDIO_CHANNELS", 32)),
                                int(getattr(h3_sampler, "AUDIO_STEREO", 2)), a_t)
            h3ff.warn("[H3-FaceResample] WARNING: no audio input (pack.a_lat empty, audio port "
                      "not connected) — using a SILENT placeholder; lip sync is NOT driven\n"
                      "[H3-FaceResample] 警告: 无音频输入 (pack 内无 a_lat 且 audio 端口未接) — "
                      "使用静音占位, 口型不受音频驱动")
        Ta = int(a_lat.shape[-1])

        # ---- 预计算分块计划 (同身份连续子轨合并为一条采样序列; 跨镜头单元强制断开) ----

        _shots_meta = [(int(s), int(e)) for s, e in ((pack.get("meta") or {}).get("shots") or [])]

        def _window_at(group, gframe):
            """全局帧 gframe 所在子轨的窗口 (cx, cy, S); 找不到返回 None (几何护栏用)。"""
            for st in group:
                f0, f1 = int(st["f0"]), int(st["f1"])
                if f0 <= int(gframe) < f1:
                    idx = int(gframe) - f0
                    centers = st.get("centers") or []
                    if idx < len(centers):
                        s_list = st.get("S_list")
                        s_val = 0
                        if s_list is not None and idx < len(s_list) and int(s_list[idx]) > 0:
                            s_val = int(s_list[idx])
                        elif int(st.get("S") or 0) > 0:
                            s_val = int(st["S"])
                        return float(centers[idx][0]), float(centers[idx][1]), float(s_val)
            return None

        plans = []
        gi_total = 0
        for st in subs:
            if st.get("skip") or st.get("crop_off") is None:
                st["canvas_off"] = None
        _sampled = [st for st in subs if not st.get("skip") and st.get("crop_off") is not None]
        _i = 0
        while _i < len(_sampled):
            st0 = _sampled[_i]
            tid = st0.get("track_id", 0)
            _j = _i + 1
            while _j < len(_sampled) \
                    and _sampled[_j].get("track_id", 0) == tid \
                    and int(_sampled[_j]["f0"]) == int(_sampled[_j - 1]["f1"]):
                _j += 1
            group = _sampled[_i:_j]
            _i = _j
            off0 = int(group[0]["crop_off"])
            K = sum(int(s["f1"]) - int(s["f0"]) for s in group)
            _chk = off0
            for s in group:
                if int(s["crop_off"]) != _chk:
                    raise RuntimeError("[H3-FaceResample] row ledger broken inside merged group — rerun Face_Cut\n"
                                       "[H3-FaceResample] 合并组内行账本断裂 — 请重跑 Face_Cut")
                _chk += int(s["f1"]) - int(s["f0"])

            if seg_src:
                # 集成模式: 按主采样实际段边界切块 (与 _pick_prompt 的提示词映射同源)
                blocks = _plan_blocks_segments(K, int(group[0]["f0"]), main_boundaries,
                                               main_decoded or n_frames_total, ctx_budget)
            else:
                blocks = _plan_blocks(K, chunk_budget, ctx_budget)
            if not blocks:
                raise RuntimeError(f"[H3-FaceResample] identity {tid}: empty block plan — check info seg account\n"
                                   f"[H3-FaceResample] 身份 {tid}: 块计划为空 — 请检查 info 分段账本")

            plans.append({"group": group, "tid": tid, "off": off0, "K": K, "blocks": blocks, "gi_base": gi_total})
            gi_total += len(blocks)
            h3ff.vlog(f"[H3-FaceResample] identity {tid}: "
                      f"{len(group)} subtrack(s) merged, K={K} rows [{off0},{off0 + K}), {len(blocks)} block(s)\n"
                      f"[H3-FaceResample] 身份 {tid} : 合并 {len(group)} 条子轨, "
                      f"K={K} 行 [{off0},{off0 + K}), {len(blocks)} 个采样块")

        _face_tracking = str((pack.get("meta") or {}).get("face_tracking") or "single")

        text_cond_cache = {}
        text_cond_hits = 0

        canvas_rows = []
        identity_ref_frames = []
        overall_pbar = comfy.utils.ProgressBar(gi_total)
        for pi, p in enumerate(plans):
            group, tid, off, K = p["group"], p["tid"], p["off"], p["K"]
            blocks, gi_base = p["blocks"], p["gi_base"]
            f0_first = int(group[0]["f0"])
            for gst in group:
                gst["canvas_off"] = int(gst["crop_off"]) 
            rows_in = crop_images[off:off + K].to(torch.float16).float()
            block_seeds = [int(seed) + 1 + gi_base for _ in range(len(blocks))]

            # ---- 自动身份锚 (v21): 优先 Face_Cut 定案的身份锚定行, 回退纯清晰度 ----
            ri_eff = list(ri)
            id_tag = None
            if identity_ref:
                _gi, _anchor_src = None, None
                _best_rr, _best_key, _best_img = None, None, None
                for _gst in group:
                    _rr = _gst.get("ref_row")
                    if _rr is None:
                        continue
                    _rel = int(_rr) - off
                    if not (0 <= _rel < int(K)):
                        continue
                    _sh = _gst.get("ref_sharp")
                    if _sh is None:                      # 旧 pack 无 ref_sharp → 现算补齐 (与 FaceCut 同度量)
                        _sh = _row_sharpness(rows_in[_rel])
                    _key = (float(_gst.get("ref_score", 0.0)), float(_sh))
                    if _best_key is None or _key > _best_key:
                        _best_rr, _best_key = _rel, _key
                        _best_img = _gst.get("ref_image")
                if _best_rr is not None:
                    _gi = _best_rr
                    if _best_img is not None:
                        # 干净参考: FaceCut 从源帧按锚定窗口重采样的原画面 (未经 pre_blur/SR)
                        _frame = _best_img.detach().to(torch.float32).contiguous()
                        _anchor_src = (f"face_cut clean ref_image (original, no pre_blur/SR, "
                                       f"score={_best_key[0]:.1f} sharp={_best_key[1]:.4f})")
                    else:
                        _frame = rows_in[_gi]
                        _anchor_src = (f"face_cut ref_row PROCESSED (old pack w/o ref_image — "
                                       f"rerun Face_Cut with clear_cache for original ref; "
                                       f"score={_best_key[0]:.1f} sharp={_best_key[1]:.4f})")
                else:
                    _pick = _pick_sharpest_row(rows_in)
                    if _pick is not None:
                        _gi, _frame = _pick
                        _anchor_src = "sharpest (fallback: pack has no ref_row)"
                        h3ff.warn(f"[H3-FaceResample] identity_ref id{tid}: pack has no verified "
                                  f"ref_row — plain sharpness anchor in use; rerun H3FaceCut for "
                                  f"identity-verified anchors\n"
                                  f"[H3-FaceResample] 身份{tid}: face_pack 无已验证锚定行 — "
                                  f"回退纯清晰度选锚; 请重跑 Face_Cut 以获得身份校验锚定")
                if _gi is not None:
                    # _frame = rows_in[_gi]
                    try:

                        _entry = (h3_sampler._prepare_ref_images(
                            [_frame.unsqueeze(0).float()], vae, device, res, res, crop_mode) or [None])[0]
                        if _entry and _entry.get("pixel") is not None:
                            ri_eff.append(_entry)
                            id_tag = f"<Picture {len(ri_eff)}>"
                            _px = _entry["pixel"].detach().float().cpu()
                            if _px.dim() == 3:
                                _px = _px.unsqueeze(0)
                            identity_ref_frames.append(_px.clamp(0.0, 1.0).contiguous())
                            h3ff.log(f"[H3-FaceResample] identity_ref: id{tid} anchor = row {_gi} "
                                     f"(global {off + _gi}) [source: {_anchor_src}] → {id_tag}\n"
                                     f"[H3-FaceResample] 身份锚: 身份{tid} 锚定帧 = 组内第 {_gi} 行 "
                                     f"(全局第 {off + _gi} 行) [来源: {_anchor_src}] → {id_tag}")
                    except Exception as _e:
                        h3ff.warn(f"[H3-FaceResample] identity_ref build failed ({_e}) — "
                                  f"continuing without anchor\n"
                                  f"[H3-FaceResample] 身份锚构建失败 ({_e}) — 不使用锚定继续")
    

            fix_sig_id = fix_sig

            block_meta = []
            hash_lines = []
            for bi, (b0, b1, _ctx) in enumerate(blocks):
                g0, g1 = f0_first + b0, f0_first + b1
                prompt_b, src_name = _prompt_lookup(g0, g1)
                prompt_b = _compose_face_prompt(prompt_b, face_prompt, prompt_mode, id_tag)
                s_r = (g0 / n_frames_total) if segmented_active else 0.0
                e_r = (g1 / n_frames_total) if segmented_active else 0.0
                hash_lines.append(prompt_b + "\x00" + f"{s_r:.6f}->{e_r:.6f}" + f"|{len(ri_eff)},{len(rv)},{len(ra)}")
                block_meta.append({"prompt": prompt_b, "src": src_name, "s_r": s_r, "e_r": e_r})

            rmeta = {"rows_hash": latent_cache.tensor_sig(rows_in.contiguous(), 65536),
                     "seg_frames": int(K), "latent_w": int(res), "latent_h": int(res),
                     "a_lat_hash": _md5_bytes(a_lat.detach().float().cpu().numpy().tobytes()),
                     "ref_fps": int(ref_fps), "lock_audio": bool(lock_a),
                     "seed": _md5_bytes(str(block_seeds).encode()),
                     "sigmas_hash": _md5_bytes(fix_sig_id.detach().cpu().numpy().tobytes()),
                     "sigma_scale": "1.0000",
                     "noise_v": 2,
                     "prompt_v": 2,
                     "identity_ref_fp": (latent_cache.tensor_sig(ri_eff[-1]["pixel"], 4096)
                                         if id_tag is not None else None),
                     "window_prompt_hash": _md5_bytes("\n".join(hash_lines).encode("utf-8")),
                     "conditions_hash": conditions_hash, "sampler_tag": sampler_tag,
                     "plan_hash": _md5_bytes(str(blocks).encode()),
                     "chunk_frames": int(chunk_budget), "context_frames": int(ctx_budget),
                     "model_fp": model_fp, "vae_fp": vae_fp, "clip_fp": clip_fp}
            if _own_graph_fp is not None:
                # 仅独立模式存在 集成模式 rmeta 不变
                rmeta["graph_fp"] = _own_graph_fp
            cname = f"resample_grp{pi:03d}.pt"
            rows = None
            if cache_dir:
                _hit = latent_cache.load_blob(cache_dir, cname, rmeta, sensitive_keys=list(rmeta.keys()))
                if _hit is not None:
                    try:
                        _t = _hit.float()
                        _ok = (_t.dim() == 4 and int(_t.shape[0]) == K
                               and int(_t.shape[1]) == res and int(_t.shape[2]) == res)
                    except Exception:
                        _ok, _t = False, None
                    if _ok:
                        rows = _t.contiguous()
                        print("\033[33m" + f"[H3-FaceResample] identity {tid} rows [{off},{off + K}) loaded from cache, "
                              f"skipping sampling\n[H3-FaceResample] 身份 {tid} 行 [{off},{off + K}) 加载缓存，跳过采样"
                              + "\033[0m")

            if rows is None:
                blk_parts = []
                _no_ref_warned = False
                prev_v = None           
                _prev_enc_real = 0      
                for bi, (b0, b1, ctx) in enumerate(blocks):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    _gblk = gi_base + bi + 1
                    _g0, _g1 = f0_first + b0, f0_first + b1
                    h3ff.log(f"[H3-FaceResample] ▶ block {_gblk}/{gi_total} — identity {tid}\n"
                             f"[H3-FaceResample] ▶ 采样块 {_gblk}/{gi_total} — 身份 {tid}")

                    # ---- 编码长度补齐到 17n+5 合法长度 (重复末帧, 上限 _PAD_CAP=21), 解码后裁掉 ----
                    enc_real = (b1 - b0) + ctx
                    enc_legal = _legal_enc_len(enc_real)
                    pad = enc_legal - enc_real
                    T_blk = h3ff.video_latent_frames(enc_legal)
                    enc_px = rows_in[b0 - ctx:b1].contiguous()
                    if pad > 0:
                        enc_px = torch.cat([enc_px, enc_px[-1:].repeat(pad, 1, 1, 1)], dim=0)
                    enc0 = f0_first + b0 - ctx
                    prompt_b = block_meta[bi]["prompt"]
                    src_name = block_meta[bi]["src"]

                    blk_ri, blk_rv, blk_ra = ri_eff, rv, ra
                    # ---- 提前解析本块提示词的引用声明: 未声明 <Video N>/<Audio N> 时跳过切片,
                    _fmt_b = str(rt.get("prompt_format") or "official")
                    _mentions = h3_sampler._parse_ref_mentions(prompt_b)
                    _declares_va = (_fmt_b == "raw") or any(m[0] in ("video", "audio") for m in _mentions)
                    if segmented_active:
                        if _declares_va:
                            try:
                                blk_rv = h3_sampler._slice_ref_videos_for_segment(
                                    rv, block_meta[bi]["s_r"], block_meta[bi]["e_r"], vae, audio_vae, device, ref_fps)
                                blk_ra = h3_sampler._slice_ref_audios_for_segment(
                                    ra, block_meta[bi]["s_r"], block_meta[bi]["e_r"], audio_vae, device)
                            except Exception as _e:
                                h3ff.warn(f"[H3-FaceResample] ref slicing failed ({_e}), re-encoding full refs for this block\n"
                                          f"[H3-FaceResample] 参考切片失败 ({_e})，本块回退为全量参考 (即时编码)")
                                try:
                                    blk_rv = h3_sampler._prepare_ref_videos(rt.get("ref_videos"), vae, audio_vae, device, res, res, ref_fps, crop_mode, pre_encode=True)
                                    blk_ra = h3_sampler._prepare_ref_audios(rt.get("ref_audios"), audio_vae, device, pre_encode=True)
                                except Exception as _e2:
                                    h3ff.warn(f"[H3-FaceResample] full-ref fallback failed ({_e2}), this block gets no refs\n"
                                              f"[H3-FaceResample] 全量参考回退失败 ({_e2})，本块不传参考")
                                    blk_rv, blk_ra = [], []
                        else:
                            # 本块未声明 <Video N>/<Audio N>: 跳过切片与 VAE 编码 (省一次)
                            blk_rv, blk_ra = [], []
                            h3ff.vlog(f"[H3-FaceResample] block frames [{f0_first + b0},{f0_first + b1}): prompt declares "
                                      f"no <Video N>/<Audio N> — reference slicing/encoding skipped\n"
                                      f"[H3-FaceResample] 块帧 [{f0_first + b0},{f0_first + b1}): 提示词未声明 "
                                      f"<Video N>/<Audio N> — 跳过参考切片与编码")
                
                    # ---- 声明才引用: 提示词提到 <Picture N> 等才传递参考素材 ----
                    try:
                        if _mentions:
                            blk_ri, blk_rv, blk_ra, prompt_b = h3_sampler._filter_refs_for_prompt(
                                prompt_b, blk_ri, blk_rv, blk_ra, fmt=_fmt_b)
                            if not blk_ri and any(m[0] == "picture" for m in _mentions):
                                h3ff.warn(f"[H3-FaceResample] block frames [{f0_first + b0},{f0_first + b1}): picture refs "
                                          f"declared but none resolved — check ref count\n"
                                          f"[H3-FaceResample] 块帧 [{f0_first + b0},{f0_first + b1}): 声明了参考图但无一解析成功, "
                                          f"请检查 ref_image 数量与编号")
                        else:
                            blk_ri = blk_rv = blk_ra = []
                            if (ri or rv or ra) and not _no_ref_warned:
                                _no_ref_warned = True
                                h3ff.vlog(f"[H3-FaceResample] identity {tid} rows [{off},{off + K}): prompt declares no refs — "
                                          f"reference media NOT passed (declare <Picture N> to use)\n"
                                          f"[H3-FaceResample] 身份 {tid} 行 [{off},{off + K}): 提示词未声明引用 — "
                                          f"参考素材未传递 (如需使用请在提示词中写 <Picture N>)")
                    except Exception as _e:
                        h3ff.warn(f"[H3-FaceResample] ref filter failed ({_e}), passing all refs\n"
                                  f"[H3-FaceResample] 参考过滤失败 ({_e})，回退全量传递")
                
                    base_lat = h3ff.encode_frames_adaptive(vae, enc_px.contiguous(), want_t=T_blk, tag=f"[身份{tid} 块{bi + 1}]")
                    if base_lat is None:
                        raise RuntimeError("[H3-FaceResample] canvas encoding failed\n[H3-FaceResample] 画布编码失败")
                    base_lat = base_lat.to(device=device, dtype=torch.float32)

                    # ---- 块间续接锚定 (与主采样 _copy_overlap_tail + noise_mask=0 同逻辑) ----
                    vh = min(h3ff.video_latent_frames(ctx), T_blk - 1) if ctx > 0 else 0   
                    if bi > 0 and prev_v is not None and vh > 0:
                        _anchor_ok = True
                        _w_prev = _window_at(group, f0_first + b0 - 1)
                        _w_cur = _window_at(group, f0_first + b0)
                        if _w_prev is not None and _w_cur is not None:
                            _s_ref = max(_w_prev[2], _w_cur[2])
                            if _s_ref > 0:
                                _dist = ((_w_prev[0] - _w_cur[0]) ** 2 + (_w_prev[1] - _w_cur[1]) ** 2) ** 0.5
                                if _dist > 1.0 * _s_ref:
                                    if _face_tracking == "multi_sec":
                                        h3ff.warn(f"[H3-FaceResample] identity {tid} block {bi + 1}: window jump "
                                                  f"{_dist:.0f}px at boundary — likely shot cut, same identity "
                                                  f"guaranteed by SeC, anchoring continues\n"
                                                  f"[H3-FaceResample] 身份 {tid} 块 {bi + 1}: 边界窗口跳变 "
                                                  f"{_dist:.0f}px — 疑似镜头切换, SeC 保证同一身份, 继续锚定")
                                    else:
                                        _anchor_ok = False
                                        h3ff.warn(f"[H3-FaceResample] identity {tid} block {bi + 1}: window jump "
                                                  f"{_dist:.0f}px > {_s_ref:.0f}px — possible identity switch in "
                                                  f"single-face mode, anchoring disabled for this boundary\n"
                                                  f"[H3-FaceResample] 身份 {tid} 块 {bi + 1}: 窗口跳变 "
                                                  f"{_dist:.0f}px > {_s_ref:.0f}px — 单脸模式疑似换人, 本边界不锚定")
                                
                        if _anchor_ok:
                            T_real_prev = h3ff.video_latent_frames(_prev_enc_real)
                            n_copy = min(vh, T_real_prev)
                            if n_copy > 0:
                                base_lat = base_lat.clone()
                                base_lat[:, :, :n_copy] = prev_v[:, :, T_real_prev - n_copy:T_real_prev].to(
                                    device=base_lat.device, dtype=base_lat.dtype)
                                h3ff.vlog(f"[H3-FaceResample] identity {tid} block {bi + 1}: head anchored on prev tail "
                                          f"({n_copy} tokens ≈ {ctx} frames)\n"
                                          f"[H3-FaceResample] 身份 {tid} 块 {bi + 1}: 头部锚定上一块尾部 "
                                          f"({n_copy} token ≈ {ctx} 帧)")

                    a0 = max(0, min(Ta - 1, int(round(enc0 / ref_fps * h3ff.AUDIO_LATENTS_PER_SEC))))
                    a1 = max(a0 + 1, min(Ta, int(round((f0_first + b1) / ref_fps * h3ff.AUDIO_LATENTS_PER_SEC))))
                    audio_blk = a_lat[..., a0:a1].to(device=device, dtype=torch.float32)
                    if pad > 0 and audio_blk.numel():
                        pad_a = max(1, int(round(pad / ref_fps * h3ff.AUDIO_LATENTS_PER_SEC)))
                        tail = audio_blk[..., -1:]
                        rep = [1] * tail.dim()
                        rep[-1] = pad_a
                        audio_blk = torch.cat([audio_blk, tail.repeat(rep)], dim=-1)
                    try:
                        latent_i = comfy.nested_tensor.NestedTensor((base_lat, audio_blk))
                    except Exception:
                        latent_i = (base_lat, audio_blk)
                    # ---- noise_mask: ctx 头 + lock_audio/audio_drive (音频区不重采样) ----
                    mask = None
                    if ctx > 0 or lock_a:
                        vm = torch.ones_like(base_lat)
                        if ctx > 0:
                            vm[:, :, :vh] = 0.0
                        am = torch.zeros_like(audio_blk) if lock_a else torch.ones_like(audio_blk)
                        if (not lock_a) and ctx > 0:
                            ah = min(max(1, int(round(ctx / ref_fps * h3ff.AUDIO_LATENTS_PER_SEC))),
                                     max(1, int(audio_blk.shape[-1]) - 1))
                            am[..., :ah] = 0.0
                        try:
                            mask = comfy.nested_tensor.NestedTensor((vm, am))
                        except Exception:
                            mask = (vm, am)


                    payload = h3_conditioning.build_conditioning_payload(
                        seed=int(rt.get("seed", 0)) + 1 + gi_base,
                        frame_count=b1 - b0,
                        ref_img_data=blk_ri,
                        ref_vid_data=blk_rv,
                        ref_aud_latents=[a["latent"] for a in blk_ra if a.get("latent") is not None],
                        fps=ref_fps)

                    # ---- 文本条件缓存: 同 prompt + 同参考素材 + 同 CLIP 权重跳过 Qwen3-VL 编码 ----
                    _cond_key = _text_cond_fingerprint(prompt_b, blk_ri, blk_rv, blk_ra) + "|" + str(clip_fp)
                    _cond_hit = text_cond_cache.get(_cond_key)
                    if _cond_hit is not None:
                        positive = _cond_hit
                        text_cond_hits += 1
                        h3ff.vlog(f"[H3-FaceResample] identity {tid} block {bi + 1}: text-cond cache hit, "
                                  f"Qwen3-VL encode skipped\n"
                                  f"[H3-FaceResample] 身份 {tid} 块 {bi + 1}: 文本条件缓存命中, 跳过 Qwen3-VL 编码")
                    else:
                        positive = h3_conditioning.encode_text_with_references(
                            clip, prompt_b, payload["ref_items_for_clip"], device,
                            images_for_clip=payload.get("images_for_clip"))
                        text_cond_cache[_cond_key] = positive
                    positive = h3_conditioning.inject_conditioning_data(positive, payload)


                    s = block_seeds[bi]
                    noise = comfy.sample.prepare_noise(latent_i, s)
                    callback = latent_preview.prepare_callback(model, k)
                    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                    try:
                        if rt.get("sampler_obj") is not None:
                            out_s = comfy.sample.sample_custom(
                                model, noise, CFG, rt["sampler_obj"], fix_sig_id, positive, [], latent_i,
                                noise_mask=mask, callback=callback, disable_pbar=disable_pbar, seed=s)
                        else:
                            ks = comfy.samplers.KSampler(
                                model, steps=k, device=model.load_device,
                                sampler=rt.get("sampler_name", "euler"),
                                scheduler=rt.get("scheduler", "simple"),
                                denoise=1.0, model_options=model.model_options)
                            out_s = ks.sample(noise, positive, [], cfg=CFG, latent_image=latent_i,
                                              denoise_mask=mask, sigmas=fix_sig_id,
                                              callback=callback, disable_pbar=disable_pbar, seed=s,
                                              force_full_denoise=True)
                        v_i, _ = h3_conditioning.unpack_nested_latent({"samples": out_s})

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(f"[H3-FaceResample] block failed: {e}"
                                           f"\n[H3-FaceResample] 块重采样失败: {e}")
                    if v_i is None or v_i.dim() != 5 or int(v_i.shape[2]) != T_blk:
                        raise RuntimeError(f"[H3-FaceResample] bad output shape: "
                                           f"{None if v_i is None else tuple(v_i.shape)}"
                                           f"\n[H3-FaceResample] 输出形状异常: "
                                           f"{None if v_i is None else tuple(v_i.shape)}")
                    prev_v = v_i.detach()
                    _prev_enc_real = enc_real
                    try:
                        px_i = vae.decode(v_i)
                    except Exception as _oom:
                        if not isinstance(_oom, comfy.model_management.OOM_EXCEPTION):
                            raise
                        h3ff.warn(f"[H3-FaceResample] VAE decode OOM (identity {tid} block {bi + 1}) — "
                                  f"flushing cache and retrying\n"
                                  f"[H3-FaceResample] VAE 解码 OOM (身份 {tid} 块 {bi + 1}) — "
                                  f"清空缓存后重试")
                        comfy.model_management.soft_empty_cache()
                        try:
                            px_i = vae.decode(v_i)
                        except Exception as _oom2:
                            if not isinstance(_oom2, comfy.model_management.OOM_EXCEPTION):
                                raise
                            if hasattr(vae, "decode_tiled"):
                                h3ff.warn("[H3-FaceResample] retrying with tiled decode (slower, tile seams possible)\n"
                                          "[H3-FaceResample] 改用分块解码重试 (较慢, 可能有拼接缝)")
                                px_i = vae.decode_tiled(v_i)
                            else:
                                raise
                    if px_i.dim() == 4:
                        px_i = px_i.unsqueeze(1)

                    px_i = px_i[0].clamp(0.0, 1.0).float()[ctx: ctx + (b1 - b0)].cpu()
                    blk_parts.append(px_i)
                    h3ff.vlog(f"[H3-FaceResample] identity {tid} block {bi + 1}: frames [{f0_first + b0},{f0_first + b1}) "
                              f"+{pad}dup->{enc_legal} 17n+5 (ctx {ctx}) keep {int(px_i.shape[0])}, "
                              f"refs({len(blk_ri)}i/{len(blk_rv)}v/{len(blk_ra)}a), prompt: {src_name}\n"
                              f"[H3-FaceResample] 身份 {tid} 块 {bi + 1}: 帧 [{f0_first + b0},{f0_first + b1}) "
                              f"补{pad}帧→{enc_legal} (17n+5, 上下文 {ctx}) 保留 {int(px_i.shape[0])}, "
                              f"参考({len(blk_ri)}图/{len(blk_rv)}视频/{len(blk_ra)}音频), 提示词: {src_name}")
                    del base_lat, audio_blk, latent_i, noise, out_s, v_i, px_i
                    overall_pbar.update(1)
                rows = torch.cat(blk_parts, dim=0).contiguous()
                if int(rows.shape[0]) != K:
                    raise RuntimeError(f"[H3-FaceResample] identity {tid} row accounting mismatch: "
                                       f"{int(rows.shape[0])} != {K}\n"
                                       f"[H3-FaceResample] 身份 {tid} 行数账目不符: {int(rows.shape[0])} ≠ {K}")
                prev_v = None  
                if cache_dir:
                    latent_cache.save_blob_async(cache_dir, cname, rows.detach().to(torch.float16).cpu().contiguous(), rmeta)
                    h3ff.vlog(f"[H3-FaceResample] identity {tid} cache save submitted (async)\n"
                              f"[H3-FaceResample] 身份 {tid} 缓存保存已提交 (异步)")


            # ---- 拆回原子轨 + 逐子轨色彩匹配 ----
            _cur = 0
            for gst in group:
                gk = int(gst["f1"]) - int(gst["f0"])
                sub_rows = rows[_cur:_cur + gk]
                if color_match:
                    goff = int(gst["crop_off"])
                    ref_rows = crop_images[goff:goff + gk].to(torch.float16).float()
                    shift = float((sub_rows.mean(dim=(0, 1, 2)) - ref_rows.mean(dim=(0, 1, 2))).abs().max())
                    sub_rows = _color_match_rows(sub_rows, ref_rows)
                    h3ff.vlog(f"[H3-FaceResample] identity {gst.get('track_id', 0)} sub [{gst['f0']},{gst['f1']}) color match: "
                              f"max mean drift {shift * 100:.2f}% corrected\n"
                              f"[H3-FaceResample] 身份 {gst.get('track_id', 0)} 子轨 [{gst['f0']},{gst['f1']}) 色彩匹配: "
                              f"已修正最大通道均值漂移 {shift * 100:.2f}%")
                canvas_rows.append(sub_rows.contiguous().to(torch.float16))
                _cur += gk
            comfy.model_management.soft_empty_cache()

        px = torch.cat(canvas_rows, dim=0)
        if int(px.shape[0]) != int(pack.get("n_crop_rows") or 0):
            raise RuntimeError("[H3-FaceResample] canvas row accounting mismatch\n[H3-FaceResample] 画布行数账目不符")

        if identity_ref_frames:
            identity_ref_out = torch.cat(identity_ref_frames, dim=0).contiguous()
        else:
            identity_ref_out = torch.zeros(1, 64, 64, 3)

        h3ff.log(f"[H3-FaceResample] done: canvas {tuple(px.shape)} ({int(px.shape[0])} rows, "
                 f"same row structure as crop_images, fp16 resident)\n"
                 f"[H3-FaceResample] 完成: 画布 {tuple(px.shape)} ({int(px.shape[0])} 行, "
                 f"与 crop_images 行结构一致, fp16 驻留)")

        if text_cond_hits:
            h3ff.log(f"[H3-FaceResample] text-cond cache: {len(text_cond_cache)} unique combo(s), "
                     f"{text_cond_hits} block(s) reused, Qwen3-VL encoded once\n"
                     f"[H3-FaceResample] 文本条件缓存: {len(text_cond_cache)} 个唯一组合, "
                     f"{text_cond_hits} 个块复用, Qwen3-VL 仅编码一次")
        
        return io.NodeOutput(px.contiguous(),
                             {"version": int(pack.get("version") or 5), "subtracks": subs, "a_lat": a_lat, "meta": pack.get("meta", {})},
                             identity_ref_out)


