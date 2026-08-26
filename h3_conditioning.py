"""
h3_conditioning.py
构建 MiniMax H3 的 conditioning 结构，核心改进：

1. 多帧运动锚定 (Multi-frame Motion Keyframes)
   - 从上一段尾部提取 N 个 latent 帧作为独立 keyframe
   - 每个 keyframe 锚定到当前段对应的时间坐标
   - 模型看到真实运动序列而非单张静帧，正确延续运动方向和速度

2. 上下文音频作为续接 (Context Audio as Continuation)
   - 上下文音频通过 ref_blocks 传入，但不加入 ref_items_for_clip
   - Qwen3-VL 不插入 <Audio j> 标签，模型视为"之前的内容"而非"参考素材"

3. 用户参考与续接上下文严格分离
   - 续接 keyframes 走 cond 通道（不被去噪，模型延续）
   - 用户 refs 走 ref 通道（模型模仿）
   - 两者的 cond_video_latents 在 extra_conds 中拼接（由 h3_patches.py 保证）

4. encode_from_tokens_scheduled
   - 与官方实现对齐，正确传递 minimax_token_tags
"""

import torch

MAX_REF_SLOTS = 9

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _pixel_index_for_latent_frame(k):
    """latent frame k 对应的第一个像素帧索引。

    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)，每 5 个 latent token 覆盖 17 个像素帧。
    latent frame 0 -> pixel 0
    latent frame 1 -> pixel 1
    latent frame 2 -> pixel 5
    latent frame 3 -> pixel 9
    latent frame 4 -> pixel 13
    latent frame 5 -> pixel 17
    latent frame 6 -> pixel 18
    """
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(k))


def _latent_frames_for_pixel_frames(pixel_frames):
    """像素帧数 -> latent 帧数 (官方公式)。"""
    if pixel_frames <= 5:
        return 2
    return ((pixel_frames - 5) // 17) * 5 + 2


def conditioning_set_values(conditioning, values={}, append=False):
    """将 values 中的键值对写入 conditioning 列表的每一项。"""
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k, v in values.items():
            if append and k in n[1]:
                n[1][k] = n[1][k] + v
            else:
                n[1][k] = v
        c.append(n)
    return c


def unpack_nested_latent(latent_dict):
    """提取 NestedTensor 中的 5D 视频 Latent 和 4D 音频 Latent。"""
    if latent_dict is None or "samples" not in latent_dict:
        return None, None
    samples = latent_dict["samples"]
    v = a = None
    if hasattr(samples, "is_nested") and samples.is_nested:
        tensors = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        tensors = list(samples)
    else:
        tensors = [samples]
    for t in tensors:
        if t.dim() == 5:
            v = t
        elif t.dim() == 4:
            a = t
    return v, a


def build_conditioning_payload(seed, frame_count,
                               first_latent=None, last_latent=None,
                               prev_segment=None, context_frames=22, fps=24,
                               ref_img_data=None, ref_vid_data=None, ref_aud_latents=None,
                               first_frame_pixel=None, last_frame_pixel=None,
                               video_latent_frames_fn=None, audio_latent_frames_fn=None):
    """
    统一构建 H3 Conditioning 载荷。

    返回 dict 包含:
      - keyframes: list of {"resolved_frame_index": int, "latent": tensor}
      - refs: list of ref block dicts
      - ref_items_for_clip: list of {"type": ..., "data": ...} for Qwen3-VL (ref 通道)
      - images_for_clip: list of [1, H, W, C] tensors for Qwen3-VL (keyframe 通道, 官方 images 参数)
      - seed, frame_count
    """
    keyframes = []
    refs = []
    ref_items_for_clip = []
    images_for_clip = []


    if first_latent is not None:
        keyframes.append({"resolved_frame_index": 0, "latent": first_latent})
        if first_frame_pixel is not None:
            images_for_clip.append(first_frame_pixel[:1])
        print(f"[H3-Auto] 首帧 keyframe: pixel_index=0")

    if prev_segment is not None:
        v_lat, a_lat = unpack_nested_latent(prev_segment)

        if v_lat is not None:
            n_latent = _latent_frames_for_pixel_frames(context_frames)
            n_latent = min(n_latent, v_lat.shape[2])
            n_latent = max(n_latent, 2)

            while n_latent > 2:
                max_pixel = _pixel_index_for_latent_frame(n_latent - 1)
                if max_pixel < frame_count:
                    break
                n_latent -= 1

            tail_latents = v_lat[:, :, -n_latent:, :, :]

            kf_pixel_indices = []
            for i in range(n_latent):
                kf_latent = tail_latents[:, :, i:i + 1, :, :]
                pixel_idx = _pixel_index_for_latent_frame(i)
                keyframes.append({
                    "resolved_frame_index": pixel_idx,
                    "latent": kf_latent,
                })
                kf_pixel_indices.append(pixel_idx)

            # print(f"[H3-Auto] 段间续接: {n_latent} 个 latent 帧作为 keyframe "
                  # f"(pixel indices: {kf_pixel_indices})")

        if a_lat is not None:
            ctx_a = int(context_frames / fps * 40)
            ctx_a = min(ctx_a, a_lat.shape[-1])
            tail_a = a_lat[..., -ctx_a:]
            refs.append({
                "kind": "audio",
                "audio_latent": tail_a,
                "ref_audio_t": tail_a.shape[-1],
            })

    if last_latent is not None:
        keyframes.append({
            "resolved_frame_index": frame_count - 1,
            "latent": last_latent,
        })
        if last_frame_pixel is not None:
            images_for_clip.append(last_frame_pixel[:1])
        print(f"[H3-Auto] 尾帧 keyframe: pixel_index={frame_count - 1} (段帧数={frame_count})")

    used_slots = len(refs)
    max_user_slots = max(0, MAX_REF_SLOTS - used_slots)
    user_refs_added = 0

    for img in (ref_img_data or []):
        if user_refs_added >= max_user_slots:
            break
        img_lat = img["latent"]
        refs.append({
            "kind": "image",
            "latent": img_lat,
            "latent_h": img_lat.shape[3],
            "latent_w": img_lat.shape[4],
            "ref_audio_t": 0,
        })
        ref_items_for_clip.append({"type": "image", "data": img["pixel"][:1]})
        user_refs_added += 1

    for vid in (ref_vid_data or []):
        if user_refs_added >= max_user_slots:
            break
        v_lat = vid["video_latent"]
        a_lat = vid.get("audio_latent")
        if a_lat is not None:
            ref_items_for_clip.append({"type": "audio"})
        ref_entry = {
            "kind": "video_audio" if a_lat is not None else "video",
            "latent": v_lat,
            "latent_t": v_lat.shape[2],
            "latent_h": v_lat.shape[3],
            "latent_w": v_lat.shape[4],
            "ref_audio_t": a_lat.shape[-1] if a_lat is not None else 0,
        }
        if a_lat is not None:
            ref_entry["audio_latent"] = a_lat
        refs.append(ref_entry)

        pixel = vid["pixel"]
        fps_val = vid.get("fps", 24)
        sample_idx = list(range(0, pixel.shape[0], max(1, fps_val // 2)))
        qwen_frames = pixel[sample_idx]
        ref_items_for_clip.append({
            "type": "video",
            "data": qwen_frames,
            "timestamps": [i / 2.0 for i in range(len(sample_idx))],
        })
        user_refs_added += 1

    for aud_lat in (ref_aud_latents or []):
        if user_refs_added >= max_user_slots:
            break
        refs.append({
            "kind": "audio",
            "audio_latent": aud_lat,
            "ref_audio_t": aud_lat.shape[-1],
        })
        ref_items_for_clip.append({"type": "audio"})
        user_refs_added += 1

    # print(f"[H3-Auto] Conditioning: keyframes={len(keyframes)} refs={len(refs)} "
          # f"images_for_clip={len(images_for_clip)} ref_items={len(ref_items_for_clip)}")
    for i, r in enumerate(refs):
        shapes = {k: tuple(v.shape) for k, v in r.items()
                  if hasattr(v, "shape")}
        meta = {k: v for k, v in r.items() if not hasattr(v, "shape") and k != "kind"}
        # print(f"[H3-Auto]   ref[{i}] kind={r['kind']} shapes={shapes} meta={meta}")

    return {
        "seed": int(seed),
        "frame_count": int(frame_count),
        "keyframes": keyframes,
        "refs": refs,
        "ref_items_for_clip": ref_items_for_clip,
        "images_for_clip": images_for_clip,
    }


def encode_text_with_references(clip, text, ref_items_for_clip, device, images_for_clip=None):
    """
    通过 Qwen3-VL 对 Prompt + Reference Items + Keyframe Images 进行联合编码。

    关键改进：
    - 使用 encode_from_tokens_scheduled (与官方实现一致)
    - 首/尾帧走 images= 通道 (FL2VA keyframe, 与官方 MiniMaxH3ImageToVideo 一致)
    - 用户参考走 minimax_ref_items= 通道 (ref2va, 与官方 MiniMaxH3ReferenceToVideo 一致)
    - 正确传递 minimax_token_tags 给 DiT 的 adaLN 模态标签系统
    """
    tokenize_kwargs = {}

    if images_for_clip:
        tokenize_kwargs["images"] = images_for_clip

    if ref_items_for_clip:
        tokenize_kwargs["minimax_ref_items"] = ref_items_for_clip

    try:
        tokens = clip.tokenize(text, **tokenize_kwargs)
    except Exception as e:
        print(f"[H3-Auto] Tokenize 警告 ({e})，降级为纯文本编码。")
        tokens = clip.tokenize(text)

    cond = clip.encode_from_tokens_scheduled(tokens)
    return cond


def inject_conditioning_data(conditioning, payload):
    """
    将构建好的 keyframes, refs 等封装注入 conditioning 字典。

    注意：不再手动设置 cond_video_latents / cond_audio_latents。
    这些由 MiniMaxH3.extra_conds 从 minimax_keyframes 和 minimax_refs 自动构建，
    h3_patches.py 确保两者拼接而非覆写。
    """
    values = {
        "seed": payload["seed"],
        "minimax_frame_count": payload["frame_count"],
    }
    if payload["keyframes"]:
        values["minimax_keyframes"] = payload["keyframes"]
    if payload["refs"]:
        values["minimax_refs"] = payload["refs"]

    return conditioning_set_values(conditioning, values)
