"""
h3_patches.py
Monkey-patch two ComfyUI internals that block multi-frame motion context:

1. PackedLayout.__init__ — stock code raises ValueError for intermediate keyframe
   positions (only pixel_index 0 and frame_count-1 are allowed). We patch it to
   accept arbitrary frame indices via the general formula:
       cond_t = video_t0 + FRAME_RESCALE * pixel_index
   (video_t0 = video segment's start t; equals text_len when no refs present)
   This is mathematically equivalent to the stock formula for the two supported
   edge cases, and generalizes to any frame index.

2. MiniMaxH3.extra_conds — stock code OVERWRITES cond_video_latents when refs are
   present, discarding keyframe latents. We patch it to CONCATENATE:
       cond_video_latents = [kf latents] + [ref latents]
   This ensures both keyframe cond rows and ref rows get their latents filled.

Keyframe t-origin fix: ref blocks (e.g. context audio) advance PackedLayout's
cursor, so the video timeline starts at text_len + sum(ref spans), NOT text_len.
We anchor cond_t to the video segment's actual start t:
    cond_t = video_t0 + FRAME_RESCALE * pixel_index
Equivalent to stock when no refs exist (video_t0 == text_len).

Usage: call apply_patches() once at module import time.

新版 ComfyUI 兼容性：升级后 ComfyUI 已原生内置上述两项能力 ——
PackedLayout.__init__ 去掉了 frame_count 参数，并直接按
`cursor + FRAME_RESCALE * resolved_frame_index` 支持任意中间关键帧位置；
MiniMaxH3.extra_conds 也已原生拼接 keyframe + ref 的 latents。因此
apply_patches() 会检测到新版（PackedLayout 无 frame_count）并自动跳过，
退化为空操作，避免向已删除的参数传 frame_count 而报 TypeError。
"""

import inspect
import torch


def _is_new_comfyui():
    """Return True when the installed ComfyUI already handles arbitrary
    keyframe positions natively (PackedLayout.__init__ has no frame_count)."""
    try:
        from comfy.ldm.minimax.model import PackedLayout
        return "frame_count" not in inspect.signature(PackedLayout.__init__).parameters
    except Exception:
        return False


def _patch_packed_layout():
    """Patch PackedLayout.__init__ to accept arbitrary resolved_frame_index.

    新版 ComfyUI 已原生支持任意 resolved_frame_index，跳过 patch。
    """
    from comfy.ldm.minimax.model import PackedLayout, FRAME_RESCALE

    if _is_new_comfyui():
        return None

    original_init = PackedLayout.__init__

    def patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None):
        real_indices = []
        if keyframes:
            for kf in keyframes:
                idx = kf.get("resolved_frame_index", 0)
                real_indices.append(idx)
                kf["resolved_frame_index"] = 0

        original_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=keyframes, refs=refs, frame_count=frame_count)

        if keyframes:
            for i, kf in enumerate(keyframes):
                kf["resolved_frame_index"] = real_indices[i]

        video_t0 = float(text_len)
        for start, stop, kind in self.segments:
            if kind == "video":
                video_t0 = float(self.position_ids[start, 0])
                break

        cond_idx = 0
        for start, stop, kind in self.segments:
            if kind == "cond" and cond_idx < len(real_indices):
                pixel_index = real_indices[cond_idx]
                cond_t = video_t0 + float(FRAME_RESCALE) * float(pixel_index)
                self.position_ids[start:stop, 0] = cond_t
                cond_idx += 1

        if real_indices:
            ref_shift = video_t0 - float(text_len)
            print(f"[H3-Auto] Layout: video_t0={video_t0:.1f} (text_len={text_len}, "
                  f"ref偏移={ref_shift:.1f}) keyframe_t="
                  f"{[round(video_t0 + float(FRAME_RESCALE) * p, 1) for p in real_indices]}")

    PackedLayout.__init__ = patched_init
    return original_init


def _patch_extra_conds():
    """Patch MiniMaxH3.extra_conds to concatenate keyframe + ref latents.

    新版 ComfyUI 已原生拼接 keyframe + ref 的 latents，跳过 patch。
    """
    from comfy.model_base import MiniMaxH3

    if _is_new_comfyui():
        return None

    original_extra_conds = MiniMaxH3.extra_conds

    def patched_extra_conds(self, **kwargs):
        out = original_extra_conds(self, **kwargs)

        keyframes = kwargs.get("minimax_keyframes", None)
        refs = kwargs.get("minimax_refs", None)

        if keyframes is not None and refs is not None:
            payload = out['minimax_payload'].cond
            kf_v = [kf["latent"] for kf in keyframes]
            ref_v = [r["latent"] for r in refs if "latent" in r]
            ref_a = [r["audio_latent"] for r in refs
                     if r.get("audio_latent") is not None]
            payload["cond_video_latents"] = kf_v + ref_v
            payload["cond_audio_latents"] = ref_a

        return out

    MiniMaxH3.extra_conds = patched_extra_conds
    return original_extra_conds


_patches_applied = False

def apply_patches():
    """Apply monkey-patches. Safe to call multiple times."""
    global _patches_applied
    if _patches_applied:
        return
    try:
        _patch_packed_layout()
    except Exception as e:
        print(f"[H3-Auto] PackedLayout patch failed: {e}")
    try:
        _patch_extra_conds()
    except Exception as e:
        print(f"[H3-Auto] extra_conds patch failed: {e}")
    _patches_applied = True
