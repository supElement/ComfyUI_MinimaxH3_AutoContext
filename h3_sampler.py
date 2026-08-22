"""
h3_sampler.py — H3 分段推理采样器

核心改进：
1. 运行时自动应用 h3_patches (PackedLayout + extra_conds 补丁)
2. 使用 h3_conditioning 模块的统一 API (多帧 keyframe, ref/context 分离, encode_from_tokens_scheduled)
3. 参考图像/视频尺寸策略与官方对齐 (adapt_canvas + 32x 对齐 + match/max 模式)
4. 参考视频帧数吸附到 17n+5 网格 + 2fps 时间戳采样 (Qwen3-VL)
5. 段间音频 cosine 交叉淡化
6. 使用 vae.encode() / audio_vae.encode() (与官方一致)
7. context_frames 上限保护 (不超过 chunk_frames - 5)
8. 音频重采样到 VAE 采样率 (32kHz)
"""

import math
import re
import torch

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False

import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import comfy.nested_tensor

from . import h3_patches
from . import h3_conditioning
from . import h3_utils

VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STEREO = 2
SPATIAL_COMPRESSION = 16
CLIP_LENGTH = 17
TOKENS_PER_CLIP = 5
TOKEN_DROP = 3
AUDIO_LATENTS_PER_SEC = 40
AUDIO_SAMPLE_RATE = 32000
FPS_DEFAULT = 24
MAX_REF_SLOTS = 9

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
AUDIO_CROSSFADE_SAMPLES = 2048


def video_latent_frames(pixel_frames):
    """像素帧数 -> latent 帧数 (官方公式)"""
    return 2 if pixel_frames <= 5 else ((pixel_frames - 5) // CLIP_LENGTH) * TOKENS_PER_CLIP + 2


def audio_latent_frames(pixel_frames, fps):
    return max(1, round(pixel_frames / fps * AUDIO_LATENTS_PER_SEC))


# FRAME_PER_TOKEN = (1, 4, 4, 4, 4)：每 5 个 latent token 覆盖 17 个像素帧
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _pixels_for_tokens(n_tokens):
    """前 n_tokens 个 latent token 解码后产出的像素帧数。"""
    return sum(_FRAME_PER_TOKEN[i % 5] for i in range(max(0, int(n_tokens))))


def compute_seam_boundaries(seg_sizes, effective_context):
    """算出 merged latent 解码后各段接缝所在的像素帧号。

    必须与 _merge_segment_latents 的裁剪账目严格同源：非首段裁掉
    n_i = min(ctx_v_tokens, seg_tokens_i - 2) 个头部 token。

    由于 seg_tokens ≡ 2 (mod 5) 且 ctx_v_tokens ≡ 2 (mod 5)，
    裁剪后 token 相位守恒，因此 token 数可直接换算为像素帧号。

    返回 (boundaries, decoded_frames)：
    boundaries 中每个 b 是"下一段新增内容的首帧索引"；
    decoded_frames 是 merged latent 解码后的总像素帧数 (供下游校验 boundaries 是否过期)。
    """
    if not seg_sizes:
        return [], 0
    if len(seg_sizes) < 2:
        return [], _pixels_for_tokens(video_latent_frames(seg_sizes[0]))

    seg_tokens = [video_latent_frames(s) for s in seg_sizes]
    ctx_v = video_latent_frames(effective_context) if effective_context > 5 else 2
    ctx_v = max(2, ctx_v)

    boundaries = []
    cum_tokens = seg_tokens[0]
    for i in range(1, len(seg_tokens)):
        boundaries.append(_pixels_for_tokens(cum_tokens))
        n_i = min(ctx_v, seg_tokens[i] - 2)
        cum_tokens += seg_tokens[i] - n_i
    return boundaries, _pixels_for_tokens(cum_tokens)


_REF_MENTION_RE = re.compile(
    r'(?<![A-Za-z0-9<])'
    r'(?:'
    r'<\s*(?P<ntag>picture|video|audio|image)\s*(?P<nnum>\d+)\s*>'
    r'|(?P<atag>image|picture|video|audio|图像|图片|视频|音频)\s*(?P<anum>\d+)'
    r')'
    r'(?![A-Za-z0-9>])',
    re.IGNORECASE)

_REF_TAG = {"picture": "Picture", "video": "Video", "audio": "Audio"}

_SUBJECT_DEF_LINE = re.compile(r'^\s*<Subject\s*(\d+)>(.*)$')
_PICTURE_TAG = re.compile(r'<Picture\s*(\d+)>')
_SUBJECT_TAG = re.compile(r'<Subject\s*(\d+)>')


def _parse_ref_mentions(prompt):
    """解析提示词中的宽泛引用，返回 [(type, number, start, end, num_start, num_end), ...]。

    type ∈ {picture, video, audio}，number 为 1 基编号。
    支持 image1 / image 1 / picture1 / 图像1 / 图片1 / 视频1 / audio 1 / 音频1，
    以及原生标签 <Picture 1> / <Video 1> / <Audio 1> (大小写不敏感)。
    """
    mentions = []
    for m in _REF_MENTION_RE.finditer(prompt):
        ntag = m.group("ntag")
        if ntag is not None:
            kw = ntag.lower()
            num = int(m.group("nnum"))
            ns, ne = m.start("nnum"), m.end("nnum")
        else:
            kw = m.group("atag").lower()
            num = int(m.group("anum"))
            ns, ne = m.start("anum"), m.end("anum")
        if kw in ("image", "picture", "图像", "图片"):
            typ = "picture"
        elif kw in ("video", "视频"):
            typ = "video"
        else:
            typ = "audio"
        mentions.append((typ, num, m.start(), m.end(), ns, ne))
    return mentions


def _render_all_mentions(prompt, mentions, index_maps, fmt):
    """单次遍历渲染所有引用 (按位置排序)，避免多次替换导致的 span 偏移。

    mentions: [(type, num, start, end, num_start, num_end), ...]
    index_maps: {type: {orig_num: new_num}}
    fmt='official' -> 统一输出 <Picture N>/<Video N>/<Audio N>
    fmt='legacy'   -> 保留原始前缀/后缀，仅替换数字
    """
    if not mentions:
        return prompt
    out = []
    last = 0
    for typ, num, s, e, ns, ne in sorted(mentions, key=lambda m: m[2]):
        out.append(prompt[last:s])
        new_num = index_maps.get(typ, {}).get(num)
        if new_num is None:
            out.append(prompt[s:e])
        elif fmt == "official":
            out.append(f'<{_REF_TAG[typ]} {new_num}>')
        else:
            out.append(prompt[s:ns] + str(new_num) + prompt[ne:e])
        last = e
    out.append(prompt[last:])
    return "".join(out)


def _filter_refs_official(window_prompt, ref_img_data, ref_vid_data, ref_aud_latents):
    """官方标签体系 (<Subject N> / <Picture N>, 1-based) 的参考图过滤。

    - 从 subject_definitions 定义行 (行首 <Subject N> 且含 <Picture M>) 解析
      Subject -> Picture 映射 (一个 Subject 可来自多张 Picture, 不假设序号对应)
    - 定义行内的 <Picture M> 不算引用; 正文中的 <Subject N>/<Picture M> 才算
    - <Subject N> 引用通过映射展开为其来源 Picture
    - 未被引用 Subject 的定义行从提示词中删除 (避免与重映射后的编号冲突)
    - 传递的 Picture 按编号升序重排, 保持 1-based (与模型 presentation 对齐)
    """
    all_imgs = ref_img_data or []

    lines = window_prompt.split('\n')
    def_line_idx = set()
    subj_map = {}
    for i, line in enumerate(lines):
        m = _SUBJECT_DEF_LINE.match(line)
        if m and _PICTURE_TAG.search(line):
            def_line_idx.add(i)
            subj_map[int(m.group(1))] = [int(x) - 1 for x in _PICTURE_TAG.findall(line)]

    body = '\n'.join(line for i, line in enumerate(lines) if i not in def_line_idx)

    referenced_pics = set(int(x) - 1 for x in _PICTURE_TAG.findall(body))
    referenced_subjs = []
    for m in _SUBJECT_TAG.finditer(body):
        s = int(m.group(1))
        if s not in referenced_subjs:
            referenced_subjs.append(s)

    for s in referenced_subjs:
        if s in subj_map:
            referenced_pics.update(subj_map[s])
        else:
            print(f"[H3-Auto] 警告: <Subject {s}> 未在 subject_definitions 中定义, "
                  f"无法确定对应参考图")

    if not referenced_pics:
        return ref_img_data, ref_vid_data, ref_aud_latents, window_prompt

    missing = sorted(i for i in referenced_pics if i >= len(all_imgs))
    if missing:
        print(f"[H3-Auto] 警告: 引用了不存在的参考图 "
              f"{['<Picture ' + str(i + 1) + '>' for i in missing]} "
              f"(实际只有 {len(all_imgs)} 张)")

    valid = sorted(i for i in referenced_pics if i < len(all_imgs))
    if not valid:
        return ref_img_data, ref_vid_data, ref_aud_latents, window_prompt

    index_map = {orig: new for new, orig in enumerate(valid)}
    filtered_img = [all_imgs[i] for i in valid]

    def _remap_pic(m):
        orig = int(m.group(1)) - 1
        new = index_map.get(orig)
        if new is None:
            return m.group(0)
        return f'<Picture {new + 1}>'

    result_lines = []
    for i, line in enumerate(lines):
        if i in def_line_idx:
            s = int(_SUBJECT_DEF_LINE.match(line).group(1))
            if s not in referenced_subjs:
                continue
        result_lines.append(_PICTURE_TAG.sub(_remap_pic, line))
    new_prompt = '\n'.join(result_lines)

    skipped = [i for i in range(len(all_imgs)) if i not in referenced_pics]
    print(f"[H3-Auto] 参考图过滤(官方标签): "
          f"Subject引用={['<Subject ' + str(s) + '>' for s in referenced_subjs]} "
          f"传递={['<Picture ' + str(i + 1) + '>' for i in valid]} "
          f"跳过={[i + 1 for i in skipped]}")
    return filtered_img, ref_vid_data, ref_aud_latents, new_prompt


def _filter_pictures_simple(prompt, ref_img_data, fmt):
    """统一图片引用过滤 + 重映射 + 渲染 (1 基)。

    识别原生 <Picture N> 与别名 image/picture/图像/图片 (带数字)。
    只传被引用的图，编号压缩为连续 1..k。
    """
    all_imgs = ref_img_data or []
    mentions = [m for m in _parse_ref_mentions(prompt) if m[0] == "picture"]
    if not mentions:
        return all_imgs, prompt

    referenced = set()
    for _, num, *_ in mentions:
        if 1 <= num <= len(all_imgs):
            referenced.add(num)
        else:
            print(f"[H3-Auto] 警告: 引用了不存在的参考图 {num} (共 {len(all_imgs)} 张)")

    if not referenced:
        return all_imgs, prompt

    ordered = sorted(referenced)
    index_map = {orig: new for new, orig in enumerate(ordered, start=1)}
    filtered = [all_imgs[i - 1] for i in ordered]

    skipped = [i for i in range(1, len(all_imgs) + 1) if i not in referenced]
    print(f"[H3-Auto] 参考图过滤: 引用={ordered} 传递={ordered} 跳过={skipped}")

    new_prompt = _render_all_mentions(prompt, mentions, {"picture": index_map}, fmt)
    return filtered, new_prompt


def _filter_video_audio(prompt, ref_vid_data, ref_aud_data, fmt):
    """统一视频/音频引用过滤 + 重映射 + 渲染 (1 基)。

    视频音轨与独立音频按官方规则统一计数为 <Audio N>：
    音轨在前 (按视频顺序)，独立音频在后。视频与其配对音轨绑定 (不可分离)。
    """
    videos = ref_vid_data or []
    audios = ref_aud_data or []

    mentions = _parse_ref_mentions(prompt)
    video_mentions = [m for m in mentions if m[0] == "video"]
    audio_mentions = [m for m in mentions if m[0] == "audio"]

    if not video_mentions and not audio_mentions:
        return videos, audios, prompt

    audio_slots = []  # ("soundtrack", vid_idx) / ("standalone", aud_idx)
    for i, v in enumerate(videos):
        if v.get("audio_latent") is not None:
            audio_slots.append(("soundtrack", i))
    for j in range(len(audios)):
        audio_slots.append(("standalone", j))

    referenced_videos = set()
    for _, num, *_ in video_mentions:
        if 1 <= num <= len(videos):
            referenced_videos.add(num)
        else:
            print(f"[H3-Auto] 警告: 引用了不存在的参考视频 {num} (共 {len(videos)} 个)")

    referenced_audio_slots = set()
    for _, num, *_ in audio_mentions:
        if 1 <= num <= len(audio_slots):
            referenced_audio_slots.add(num - 1)
        else:
            print(f"[H3-Auto] 警告: 引用了不存在的参考音频 {num} (共 {len(audio_slots)} 个)")

    kept_videos = set(v - 1 for v in referenced_videos)
    for slot_idx in referenced_audio_slots:
        kind, idx = audio_slots[slot_idx]
        if kind == "soundtrack":
            kept_videos.add(idx)
    kept_videos = sorted(kept_videos)

    kept_audio_slots = []
    for slot_idx, (kind, idx) in enumerate(audio_slots):
        if kind == "soundtrack":
            if idx in kept_videos:
                kept_audio_slots.append(slot_idx)
        else:
            if slot_idx in referenced_audio_slots:
                kept_audio_slots.append(slot_idx)

    if not kept_videos and not kept_audio_slots:
        return videos, audios, prompt  # 引用全部越界 -> 全传 (与图片路径一致)

    video_map = {idx + 1: new for new, idx in enumerate(kept_videos, start=1)}
    audio_map = {slot + 1: new for new, slot in enumerate(kept_audio_slots, start=1)}

    filtered_videos = [videos[i] for i in kept_videos]
    filtered_audios = [audios[audio_slots[s][1]]
                       for s in kept_audio_slots if audio_slots[s][0] == "standalone"]

    passed_v = [i + 1 for i in kept_videos]
    passed_a = [audio_slots[s][1] + 1 for s in kept_audio_slots
                if audio_slots[s][0] == "standalone"]
    print(f"[H3-Auto] 参考视频/音频过滤: 视频传递={passed_v} 独立音频传递={passed_a}")

    new_prompt = _render_all_mentions(
        prompt, video_mentions + audio_mentions,
        {"video": video_map, "audio": audio_map}, fmt)
    return filtered_videos, filtered_audios, new_prompt


def _filter_refs_for_prompt(window_prompt, ref_img_data, ref_vid_data, ref_aud_latents,
                            fmt="official"):
    """统一引用过滤入口 (图片/视频/音频)。

    fmt:
      - official: 统一转官方标签 <Picture N>/<Video N>/<Audio N>；<Subject> 走官方映射
      - legacy:   保留原始写法 (image1/图像1/<Picture 1>)，仅重映射编号
      - raw:      完全不处理，全传所有参考、提示词原样
    """
    if fmt == "raw":
        return ref_img_data, ref_vid_data, ref_aud_latents, window_prompt

    if fmt == "official" and "<Subject" in window_prompt:
        filtered_img, _, _, window_prompt = _filter_refs_official(
            window_prompt, ref_img_data, ref_vid_data, ref_aud_latents)
    else:
        filtered_img, window_prompt = _filter_pictures_simple(
            window_prompt, ref_img_data, fmt)

    filtered_vid, filtered_aud, window_prompt = _filter_video_audio(
        window_prompt, ref_vid_data, ref_aud_latents, fmt)

    return filtered_img, filtered_vid, filtered_aud, window_prompt


def run_auto_context_generation(model, vae, audio_vae, clip,
                                first_frame, last_frame,
                                ref_images, ref_videos, ref_audios,
                                long_prompt, width, height,
                                total_frames, fps, chunk_frames, context_frames,
                                steps, cfg, sampler_name, scheduler, seed,
                                prompt_mode="auto", prompt_format="official",
                                clip_mode="Clip_Frame", clip_tag="段1",
                                crop_mode="stretch", ref_sync_mode="global",
                                decode_output=False,
                                drive_audio=None, audio_drive=False,
                                latent_input=None, sigmas=None, denoise=1.0,
                                lock_audio=True, video_context_denoise=0.0,
                                sampler=None):
    h3_patches.apply_patches()

    device = comfy.model_management.get_torch_device()
    total_seconds_for_prompt = total_frames / fps

    is_tag_mode = (clip_mode == "Clip_Tag")

    if is_tag_mode:
        # Clip_Tag 模式下 chunk_frames 不约束总时长：无明确时长的段默认用 total_frames 作为
        # 整段时长，使单段 (仅一个标签) 时总帧数贴合 total_frames，而非被 chunk_frames 顶成 90 帧。
        tag_schedule = h3_utils.build_tag_schedule(
            long_prompt, clip_tag, default_seconds=total_frames / fps)
        tag_segments = tag_schedule["segments"]
        tag_prefix = tag_schedule["prefix"]

        seg_target_frames = [int(d * fps) for _, d in tag_segments]
        chunks, seg_sizes = h3_utils.compute_tag_chunks(
            seg_target_frames, context_frames)

        effective_new = seg_sizes[0] + sum(
            s - context_frames for s in seg_sizes[1:]) if seg_sizes else 0
        total_frames = effective_new
        chunk_frames = seg_sizes[0] if seg_sizes else 0

        print(f"[H3-Auto] Clip_Tag 模式: {len(seg_sizes)} 段")
        print(f"[H3-Auto] 各段目标帧数={seg_target_frames} 实际帧数={seg_sizes}")
        print(f"[H3-Auto] 有效新增={effective_new}帧 ({effective_new/fps:.1f}s)")
    else:
        chunks, chunk_frames = h3_utils.compute_chunks(
            total_frames, chunk_frames, context_frames)
        seg_sizes = [end - start for start, end in chunks]

    min_seg = min(seg_sizes) if seg_sizes else chunk_frames
    max_context = min_seg - 5
    effective_context = max(5, min(context_frames, max_context))
    if effective_context != context_frames:
        print(f"[H3-Auto] context_frames {context_frames} -> {effective_context} (上限保护)")

    new_frames = [seg_sizes[0]] + [s - effective_context for s in seg_sizes[1:]]
    if is_tag_mode:
        print(f"[H3-Auto] 帧数账目: 各段新增={new_frames} 合计={sum(new_frames)}帧 "
              f"(实际输出={effective_new}帧)")
    else:
        print(f"[H3-Auto] 总帧数={total_frames} 分段={len(chunks)} 各段帧数={seg_sizes} (基准={chunk_frames})")
        print(f"[H3-Auto] 帧数账目: 各段新增={new_frames} 合计={sum(new_frames)}帧 "
              f"(目标={total_frames}, 超出部分从末段头部裁剪)")
        if len(seg_sizes) >= 2 and seg_sizes[-1] > chunk_frames:
            ext = seg_sizes[-1] - chunk_frames
            print(f"[H3-Auto] 末段加长 {ext} 帧 ({ext / chunk_frames * 100:.0f}%) 吸收锚定亏空 "
                  f"(<=30% 上限, 避免新增分段)")

    if not is_tag_mode:
        prompt_schedule = h3_utils.build_prompt_schedule(
            long_prompt, total_seconds_for_prompt, mode=prompt_mode)
        print(f"[H3-Auto] 提示词模式={prompt_schedule['mode']} "
              f"时间段数={len(prompt_schedule['segments'])} "
              f"全局段(前={bool(prompt_schedule['prefix'])} 后={bool(prompt_schedule['suffix'])})")
    else:
        prompt_schedule = None
    # 二采：空间分辨率以输入 latent 为准 (忽略 widget width/height，社区一采低清二采高清)
    is_second_pass = latent_input is not None
    if is_second_pass:
        v_in, _ = h3_conditioning.unpack_nested_latent(latent_input)
        if v_in is None:
            print("[H3-Auto] 警告: latent_input 无视频 latent，回退一采")
            is_second_pass = False
            latent_w = width // SPATIAL_COMPRESSION
            latent_h = height // SPATIAL_COMPRESSION
        else:
            latent_w = int(v_in.shape[4])
            latent_h = int(v_in.shape[3])
            # H3 patch_size=(1,2,2) 要求空间偶数；奇数时按边缘复制对齐到偶数，否则段间续接 keyframe 的 patchify 会崩溃
            if latent_w % 2 != 0 or latent_h % 2 != 0:
                print(f"[H3-Auto] 警告: 输入 latent 尺寸 {latent_w}x{latent_h} 为奇数，"
                      f"H3 需偶数 (patch_size=2)。已按边缘复制对齐到 "
                      f"{latent_w + (latent_w % 2)}x{latent_h + (latent_h % 2)}；"
                      f"建议改用 32 对齐分辨率 (如 1920x1088)")
                latent_input = _pad_latent_spatial_even(latent_input)
                v_in, _ = h3_conditioning.unpack_nested_latent(latent_input)
                latent_w = int(v_in.shape[4])
                latent_h = int(v_in.shape[3])
            width = latent_w * SPATIAL_COMPRESSION
            height = latent_h * SPATIAL_COMPRESSION
            print(f"[H3-Auto] 二采模式: 空间分辨率以输入 latent 为准 {width}x{height}")
    else:
        latent_w = width // SPATIAL_COMPRESSION
        latent_h = height // SPATIAL_COMPRESSION

    if first_frame is not None or last_frame is not None:
        print("[H3-Auto] 正在 VAE 编码首/尾帧...")
    first_latent = _encode_image(vae, first_frame, device, target_w=width, target_h=height,
                                 crop_mode=crop_mode)
    last_latent = _encode_image(vae, last_frame, device, target_w=width, target_h=height,
                                crop_mode=crop_mode)

    if ref_images:
        print(f"[H3-Auto] 正在 VAE 编码参考图 ({len(ref_images)} 张)...")
    ref_img_data = _prepare_ref_images(ref_images, vae, device, width, height, crop_mode)

    if ref_videos:
        if ref_sync_mode == "segmented":
            print(f"[H3-Auto] 正在预处理参考视频 ({len(ref_videos)} 个, segmented 模式跳过全量编码)...")
        else:
            print(f"[H3-Auto] 正在 VAE 编码参考视频 ({len(ref_videos)} 个)...")
    ref_vid_data = _prepare_ref_videos(ref_videos, vae, audio_vae, device, width, height, fps,
                                       crop_mode, pre_encode=(ref_sync_mode != "segmented"))

    if ref_audios:
        if ref_sync_mode == "segmented":
            print(f"[H3-Auto] 正在预处理独立参考音频 ({len(ref_audios)} 个, segmented 模式跳过全量编码)...")
        else:
            print(f"[H3-Auto] 正在 VAE 编码独立参考音频 ({len(ref_audios)} 个)...")
    ref_aud_data = _prepare_ref_audios(ref_audios, audio_vae, device,
                                       pre_encode=(ref_sync_mode != "segmented"))

    # 音频驱动 (Audio Drive)：把源音频锁进 latent (noise_mask=0)，输出音频=源音频本身
    drive_waveform = None
    drive_sr = AUDIO_SAMPLE_RATE
    if audio_drive and drive_audio is not None:
        # 音频可能是 dict，也可能是 Mapping 子类 (如 VHS 的 LazyAudioMap)，统一用 .get() 取值
        getter = getattr(drive_audio, "get", None)
        wf = getter("waveform") if callable(getter) else None
        if wf is None:
            print("[H3-Auto] 警告: audio_drive 已开启但 drive_audio 缺少波形数据，忽略音频锁定")
        else:
            sr = int(getter("sample_rate", AUDIO_SAMPLE_RATE))
            vae_sr = int(getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE))
            if sr != vae_sr and _HAS_TORCHAUDIO:
                try:
                    wf = torchaudio.functional.resample(wf, sr, vae_sr)
                except Exception as e:
                    print(f"[H3-Auto] 音频驱动重采样失败: {e}")
            if wf.numel() == 0:
                print("[H3-Auto] 警告: drive_audio 波形为空 (0 采样)，输出音频将为空")
            elif torch.count_nonzero(wf) == 0:
                print("[H3-Auto] 警告: drive_audio 波形全为零 (静音)，请检查源音频是否正确加载")
            drive_waveform = wf
            drive_sr = vae_sr
            print(f"[H3-Auto] 音频驱动已开启: 源音频 {wf.shape[-1] / drive_sr:.2f}s @ {drive_sr}Hz "
                  f"(shape={tuple(wf.shape)})")
    elif drive_audio is not None:
        print("[H3-Auto] 提示: drive_audio 已连接但 audio_drive=disable，未启用音频锁定。"
              "如需让 audio 端口输出源音频，请把 audio_drive 设为 enable")

    # 二采：把一采 merged latent 逆向切成逐段完整 latent (段间重叠头用前段尾部重建)
    first_pass_segments = None
    if is_second_pass:
        first_pass_segments = _split_first_pass_latent(
            latent_input, seg_sizes, effective_context, fps)
        if first_pass_segments is None or len(first_pass_segments) != len(chunks):
            print("[H3-Auto] 错误: 二采切分失败 (输入 latent 与分段账目不匹配)，回退一采")
            is_second_pass = False
            first_pass_segments = None

    all_segments = []
    all_x0 = []
    prev_x0 = None
    output_cursor = 0  # 累积"新增输出"帧数，用于音频驱动按输出时间轴切片

    overall_pbar = comfy.utils.ProgressBar(len(chunks) * (2 if decode_output else 1))

    for idx, (start_f, end_f) in enumerate(chunks):
        comfy.model_management.throw_exception_if_processing_interrupted()
        is_first_chunk = (idx == 0)
        is_last_chunk = (idx == len(chunks) - 1)
        seg_frames = end_f - start_f

        # 段间续接锚定用上一段的干净 x0 预测 (含残余噪声的 samples 不能直接当 keyframe)
        anchor_prev = None
        if not is_first_chunk:
            anchor_prev = prev_x0

        account = f"帧数={seg_frames}"
        if not is_first_chunk:
            account += f" (锚定={effective_context} + 新增={seg_frames - effective_context})"
        print(f"[H3-Auto] 生成段 {idx+1}/{len(chunks)} (帧{start_f}-{end_f}) {account}")

        if is_tag_mode:
            seg_text, _ = tag_segments[idx]
            # 用实际新增帧数渲染时间，保证提示词时间坐标与实际产出帧数一致 (总时长补偿后)
            seg_seconds = new_frames[idx] / fps
            window_prompt = h3_utils.render_tag_segment(
                seg_text, seg_seconds, prompt_format, tag_prefix)
        else:
            window_prompt = h3_utils.compose_window_prompt(
                prompt_schedule, start_f / fps, end_f / fps, fmt=prompt_format)

        seg_ref_vid, seg_ref_aud = ref_vid_data, ref_aud_data
        if ref_sync_mode == "segmented":
            if seg_ref_vid or seg_ref_aud:
                print(f"[H3-Auto] 段 {idx+1}/{len(chunks)}: 正在切片并 VAE 编码参考视频/音频...")
            seg_start_ratio = start_f / total_frames if total_frames > 0 else 0.0
            seg_end_ratio = end_f / total_frames if total_frames > 0 else 0.0
            seg_ref_vid = _slice_ref_videos_for_segment(
                ref_vid_data, seg_start_ratio, seg_end_ratio,
                vae, audio_vae, device, fps)
            seg_ref_aud = _slice_ref_audios_for_segment(
                ref_aud_data, seg_start_ratio, seg_end_ratio,
                audio_vae, device)

        seg_ref_img, seg_ref_vid, seg_ref_aud, window_prompt = _filter_refs_for_prompt(
            window_prompt, ref_img_data, seg_ref_vid, seg_ref_aud, fmt=prompt_format)

        print(f"[H3-Auto] ----- 段 {idx+1}/{len(chunks)} 完整提示词 ({len(window_prompt)}字) -----")
        print(window_prompt)
        print(f"[H3-Auto] {'=' * 50}")

        payload = h3_conditioning.build_conditioning_payload(
            seed=seed + idx, frame_count=seg_frames,
            first_latent=first_latent if is_first_chunk else None,
            last_latent=last_latent if is_last_chunk else None,
            prev_segment=anchor_prev,
            context_frames=effective_context, fps=fps,
            ref_img_data=seg_ref_img,
            ref_vid_data=seg_ref_vid,
            ref_aud_latents=[a.get("latent") for a in seg_ref_aud if a.get("latent") is not None],
            first_frame_pixel=first_frame if is_first_chunk else None,
            last_frame_pixel=last_frame if is_last_chunk else None,
        )

        positive = h3_conditioning.encode_text_with_references(
            clip, window_prompt, payload["ref_items_for_clip"], device,
            images_for_clip=payload.get("images_for_clip"))

        positive = h3_conditioning.inject_conditioning_data(positive, payload)

        drive_aud_latent = None
        if drive_waveform is not None and not is_second_pass:
            # 本段锁定的音频对应输出时间轴 [gen_start, gen_end)，含非首段的续接重演头
            gen_start = output_cursor - (effective_context if not is_first_chunk else 0)
            gen_end = gen_start + seg_frames
            drive_aud_latent = _encode_drive_audio(
                audio_vae, drive_waveform, drive_sr, gen_start / fps, gen_end / fps, device)
            if drive_aud_latent is None:
                print(f"[H3-Auto] 段 {idx+1}/{len(chunks)}: drive_audio 切片为空，本段不锁定音频")

        if is_second_pass:
            seg_latent_dict = first_pass_segments[idx]
            # 二采段: 非首段 video overlap 头按 mask 去噪 + 可选锁定音频 (首段无 overlap 头，仅锁音频)
            ov_t = 0
            if not is_first_chunk:
                ov_t = max(2, video_latent_frames(effective_context)) if effective_context > 5 else 2
            if ov_t > 0 or lock_audio:
                if ov_t > 0:
                    # 用二采本身推理后的干净 samples 尾部重写 overlap 头
                    # (而非 input latent 放大后的复制，避免接缝扭曲破碎)
                    seg_latent_dict = _copy_overlap_tail(
                        seg_latent_dict, all_segments[idx - 1], ov_t)
                seg_latent_dict = _with_locked_audio(
                    seg_latent_dict, overlap_video_tokens=ov_t,
                    lock_audio=lock_audio, overlap_video_denoise=video_context_denoise)
        else:
            # 一采段: 非首段把上一段视频尾部逐 token 复制进 overlap 头并按 mask 去噪 (而非 keyframe 重演)
            overlap_video = None
            if not is_first_chunk and idx > 0:
                overlap_video, _prev_a = h3_conditioning.unpack_nested_latent(
                    all_segments[idx - 1])
            seg_latent_dict = _make_h3_empty_latent(
                latent_w, latent_h, seg_frames, fps, drive_audio_latent=drive_aud_latent,
                overlap_video=overlap_video, overlap_video_denoise=video_context_denoise,
                overlap_frames=effective_context)

        segment_latent = _sample_segment(
            model, positive, seg_latent_dict, steps, cfg,
            sampler_name, scheduler, seed + idx,
            denoise=denoise, sigmas=sigmas,
            sampler_obj=sampler)

        samples_dict = {"samples": segment_latent["samples"]}
        x0 = segment_latent.get("x0")
        x0_dict = {"samples": x0} if x0 is not None else samples_dict
        prev_x0 = x0_dict
        all_segments.append(samples_dict)
        all_x0.append(x0_dict)
        output_cursor += new_frames[idx]
        overall_pbar.update(1)

    final_latent = _merge_segment_latents(all_segments, effective_context, fps)
    final_denoised = _merge_segment_latents(all_x0, effective_context, fps)

    # 段间接缝信息：供下游 Seam_Correction 节点精确定位边界 (token 相位换算，外部无法推算)
    seam_boundaries, seam_decoded = compute_seam_boundaries(seg_sizes, effective_context)
    seam_info = {
        "boundaries": seam_boundaries,
        "seg_sizes": [int(s) for s in seg_sizes],
        "effective_context": int(effective_context),
        "decoded_frames": int(seam_decoded),
    }
    if seam_boundaries:
        print(f"[H3-Auto] 段间接缝 (解码后像素帧号): {seam_boundaries} "
              f"/ 解码总帧数={seam_decoded} -> 已写入 info")

    # 音频驱动模式：准备"无损源音频" (裁齐到总时长)。
    # [注意] audio 输出端口已注释 (见 nodes.py)，这里的结果目前无人接收，
    # 仅为将来恢复端口预留；用户请把源音频直接接到视频合成节点。
    drive_final_audio = None
    if drive_waveform is not None:
        target_samples = int(round(total_frames / fps * drive_sr))
        drive_final_audio = {"waveform": _align_length(drive_waveform, target_samples, dim=-1),
                             "sample_rate": drive_sr}
        print(f"[H3-Auto] 音频驱动: 源音频已锁进 latent，视频照它生成")

    if not decode_output:
        print("[H3-Auto] 跳过内部解码，仅输出 latent (像素请接外部 VAE Decode)")
        return None, drive_final_audio, final_latent, final_denoised, seam_info

    print("[H3-Auto] 开始逐段 VAE 解码与拼接...")
    all_pixels, all_waveforms = [], []
    for idx, seg in enumerate(all_segments):
        comfy.model_management.throw_exception_if_processing_interrupted()
        pixels, waveform = _decode_segment(vae, audio_vae, seg, device)
        all_pixels.append(pixels.cpu())
        if waveform is not None:
            all_waveforms.append(waveform.cpu())
        overall_pbar.update(1)

    n_segs = len(all_pixels)
    head_trims = [0] * n_segs
    for i in range(1, n_segs):
        head_trims[i] = min(effective_context, all_pixels[i].shape[0] - 1)

    total_after = sum(all_pixels[i].shape[0] - head_trims[i] for i in range(n_segs))
    excess = total_after - total_frames
    if excess > 0 and n_segs > 1:
        last_avail = all_pixels[-1].shape[0] - head_trims[-1] - 22
        extra = max(0, min(excess, last_avail))
        head_trims[-1] += extra
        if extra > 0:
            print(f"[H3-Auto] 末段头部追加裁剪 {extra} 帧以保护尾帧锚点")

    final_pixels = torch.cat([all_pixels[i][head_trims[i]:] for i in range(n_segs)], dim=0)
    if final_pixels.shape[0] > total_frames:
        final_pixels = final_pixels[:total_frames]
    print(f"[H3-Auto] 视频拼接: 各段裁剪={head_trims} 最终={final_pixels.shape[0]}帧")

    if drive_final_audio is not None:
        final_audio = drive_final_audio
    else:
        final_audio = _crossfade_audio(all_waveforms, head_trims, effective_context,
                                       total_frames, fps)

    return final_pixels, final_audio, final_latent, final_denoised, seam_info


def _prepare_ref_images(ref_images, vae, device, gen_w, gen_h, crop_mode="stretch"):
    """预处理参考图片：按 crop_mode 缩放 + 32x 对齐 + VAE 编码"""
    result = []
    for img in (ref_images or []):
        if img is None:
            continue
        if img.dim() == 4 and img.shape[0] > 1:
            img = img[:1]
        elif img.dim() == 3:
            img = img.unsqueeze(0)

        H, W = img.shape[1], img.shape[2]
        if crop_mode == "none":
            tw = max(CANVAS_MULTIPLE, round(W / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(H / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        elif crop_mode == "center":
            scale = max(gen_w / W, gen_h / H)
            tw = max(CANVAS_MULTIPLE, round(W * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(H * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        else:  # stretch
            tw = gen_w
            th = gen_h

        resized = _resize_image(img, tw, th, "disabled" if crop_mode == "none" else crop_mode)
        lat = _encode_image(vae, resized, device)
        if lat is not None:
            result.append({"pixel": resized, "latent": lat})
    return result


def _prepare_ref_videos(ref_videos, vae, audio_vae, device, gen_w, gen_h, fps,
                        crop_mode="stretch", pre_encode=True):
    """预处理参考视频：按 crop_mode 空间对齐 + 帧数吸附 17n+5，保留源用于 segmented 切片。

    pre_encode=False 时跳过 VAE 编码 (segmented 模式逐段重编码，全量预编码无意义)，
    仅保留像素帧/波形供切片使用，节省一次整段视频的 VAE 编码。
    """
    result = []
    for v in (ref_videos or []):
        v_img = v.get("video")
        v_aud = v.get("audio")
        if v_img is None:
            continue

        if v_img.dim() == 5:
            v_img = v_img[0]
        elif v_img.dim() == 3:
            v_img = v_img.unsqueeze(0)

        T, H, W = v_img.shape[0], v_img.shape[1], v_img.shape[2]

        if crop_mode == "none":
            cw = max(CANVAS_MULTIPLE, round(W / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(H / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        elif crop_mode == "center":
            scale = max(gen_w / W, gen_h / H)
            cw = max(CANVAS_MULTIPLE, round(W * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(H * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        else:  # stretch
            cw, ch = gen_w, gen_h
        frames = _resize_image(v_img, cw, ch, "disabled" if crop_mode == "none" else crop_mode)

        n = frames.shape[0]
        if n < 5:
            print(f"[H3-Auto] 参考视频帧数 {n} < 5，跳过")
            continue
        while n % 17 != 5:
            n -= 1
        enc_frames = frames[:n]

        v_lat = a_lat = None
        if pre_encode:
            v_lat = _encode_video_latent(vae, enc_frames, device)
            a_lat = _encode_audio(audio_vae, v_aud, device) if v_aud is not None else None
            if v_lat is None:
                continue

        result.append({
            "pixel": frames,
            "video_latent": v_lat,
            "audio_latent": a_lat,
            "audio_dict": v_aud,
            "fps": fps,
        })
    return result


def _prepare_ref_audios(ref_audios, audio_vae, device, pre_encode=True):
    """预处理独立参考音频，保留源波形用于 segmented 切片。

    pre_encode=False 时跳过 VAE 编码 (segmented 模式逐段重编码)。
    """
    result = []
    for a in (ref_audios or []):
        a_lat = None
        if pre_encode:
            a_lat = _encode_audio(audio_vae, a, device)
            if a_lat is None:
                continue
        result.append({"latent": a_lat, "audio_dict": a})
    return result


def _adapt_canvas(width, height):
    """768 短边画布 + 768*1344 面积上限，逐轴 32 对齐 (官方 adapt_canvas)"""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize_image(image, width, height, crop="disabled"):
    """使用 comfy.utils.common_upscale 缩放图像 [B, H, W, C] -> [B, height, width, C]"""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_image(vae, image, device, target_w=None, target_h=None, crop_mode="disabled"):
    """编码单张图像为 latent。

    使用 vae.encode() (与官方一致)，自动处理设备/精度/归一化。
    Input: [B, H, W, C] in 0-1 range
    Output: latent tensor [B, C, 1, H/16, W/16] (或类似)
    """
    if image is None or vae is None:
        return None
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() == 5:
        image = image[:, 0]

    if target_w is not None and target_h is not None:
        h, w = image.shape[1], image.shape[2]
        if h != target_h or w != target_w:
            image = _resize_image(image, target_w, target_h, crop_mode)

    try:
        latent = vae.encode(image)
        if latent.dim() == 4:
            latent = latent.unsqueeze(2)
        return latent
    except Exception as e:
        print(f"[H3-Auto] 图像 VAE 编码失败: {e}")
        return None


def _encode_video_latent(vae, video, device):
    """编码视频序列为 latent。

    使用 vae.encode() (与官方一致)。
    Input: [T, H, W, C] in 0-1 range
    Output: latent tensor [1, C, T_lat, H/16, W/16]
    """
    if video is None or vae is None:
        return None
    try:
        latent = vae.encode(video)
        if latent.dim() == 4:
            latent = latent.unsqueeze(2)
        return latent
    except Exception as e:
        print(f"[H3-Auto] 视频 VAE 编码失败: {e}")
        return None


def _encode_audio(audio_vae, audio_dict, device):
    """编码音频为 latent (与官方 _encode_ref_audio 一致)。

    Input: {"waveform": [B, C, L], "sample_rate": int}
    Output: latent tensor [1, 32, 2, T]
    """
    if audio_dict is None or audio_vae is None:
        return None
    waveform = audio_dict.get("waveform")
    if waveform is None:
        return None

    sr = audio_dict.get("sample_rate", AUDIO_SAMPLE_RATE)
    vae_sr = getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE)

    if sr != vae_sr and _HAS_TORCHAUDIO:
        try:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        except Exception as e:
            print(f"[H3-Auto] 音频重采样失败: {e}")

    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)

    try:
        z = audio_vae.encode(waveform[:1].movedim(1, -1))
        return z
    except Exception as e:
        print(f"[H3-Auto] 音频 VAE 编码失败: {e}")
        return None


def _fit_audio_latent(encoded_audio, template_audio):
    """把编码后的音频 latent 对齐到模板音频 latent 的形状 [batch, channels, stereo, time]。

    与 VRGDG Audio Drive 的 _fit_audio_latent 一致：过长截断、过短补零、batch 对齐。
    通道/立体声布局不匹配时返回 None (表示无法锁定，调用方回退为普通生成)。
    """
    if encoded_audio is None or encoded_audio.ndim != 4 or template_audio.ndim != 4:
        return None
    if encoded_audio.shape[1:-1] != template_audio.shape[1:-1]:
        print(f"[H3-Auto] 警告: 音频 latent 布局不匹配，跳过锁定: "
              f"got {tuple(encoded_audio.shape)}, "
              f"expected 通道 {tuple(template_audio.shape[1:-1])}")
        return None

    target_batch = template_audio.shape[0]
    if encoded_audio.shape[0] == 1 and target_batch > 1:
        encoded_audio = encoded_audio.repeat(target_batch, 1, 1, 1)
    elif encoded_audio.shape[0] != target_batch:
        encoded_audio = encoded_audio[:target_batch]
        if encoded_audio.shape[0] != target_batch:
            return None

    target_t = template_audio.shape[-1]
    current_t = encoded_audio.shape[-1]
    if current_t > target_t:
        encoded_audio = encoded_audio[..., :target_t]
    elif current_t < target_t:
        padding = encoded_audio.new_zeros((*encoded_audio.shape[:-1], target_t - current_t))
        encoded_audio = torch.cat((encoded_audio, padding), dim=-1)

    return encoded_audio.to(device=template_audio.device, dtype=template_audio.dtype)


def _slice_waveform(waveform, sr, start_sec, end_sec):
    """按秒切波形 [..., L] -> [..., s:e]，越界自动 clamp。"""
    L = waveform.shape[-1]
    s = max(0, min(L, int(round(start_sec * sr))))
    e = max(s, min(L, int(round(end_sec * sr))))
    return waveform[..., s:e]


def _encode_drive_audio(audio_vae, drive_waveform, drive_sr, start_sec, end_sec, device):
    """切源音频到 [start_sec, end_sec) 并编码成 H3 音频 latent，空切片返回 None。"""
    wav = _slice_waveform(drive_waveform, drive_sr, start_sec, end_sec)
    if wav.shape[-1] == 0:
        return None
    return _encode_audio(audio_vae, {"waveform": wav, "sample_rate": drive_sr}, device)


def _decode_segment(vae, audio_vae, seg_latent, device):
    """解码一段 latent 为视频帧 + 音频波形"""
    v_lat, a_lat = h3_conditioning.unpack_nested_latent(seg_latent)

    pixels = None
    if v_lat is not None:
        if hasattr(vae, "patcher"):
            comfy.model_management.load_model_gpu(vae.patcher)
        vae_model = vae.first_stage_model
        v_dtype = next(vae_model.parameters()).dtype
        v_lat = v_lat.to(device=device, dtype=v_dtype)
        try:
            pixels = vae_model.decode(v_lat)
            if pixels.dim() == 5:
                pixels = pixels.permute(0, 2, 3, 4, 1)
            if pixels.shape[0] == 1:
                pixels = pixels[0]
        except Exception as e:
            print(f"[H3-Auto] 视频解码失败: {e}")
            pixels = None

    waveform = None
    if a_lat is not None and audio_vae is not None:
        if hasattr(audio_vae, "patcher"):
            comfy.model_management.load_model_gpu(audio_vae.patcher)
        audio_model = audio_vae.first_stage_model
        a_dtype = next(audio_model.parameters()).dtype
        a_lat = a_lat.to(device=device, dtype=a_dtype)
        try:
            waveform = audio_model.decode(a_lat)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
        except Exception as e:
            print(f"[H3-Auto] 音频解码失败: {e}")
            waveform = None

    return pixels, waveform


def _make_h3_empty_latent(latent_w, latent_h, pixel_frames, fps,
                          drive_audio_latent=None,
                          overlap_video=None, overlap_video_denoise=0.0,
                          overlap_frames=0):
    """构建 H3 空 latent (NestedTensor: video + audio)。

    - drive_audio_latent 非空时：把音频半区替换为源音频 latent，并设 noise_mask
      (视频=1 正常去噪，音频=0 锁定不生成)，实现"音频驱动视频"。
    - overlap_video 非空时：把上一段视频尾部逐 token 复制到段首 overlap 头，
      并用 noise_mask 冻结 (mask=overlap_video_denoise，0=精确保留、不去噪)。
      overlap 头长度与 _merge_segment_latents 的裁剪账目严格同源。
      冻结上一段真实结尾而非"重演"，消除接缝停顿/位置错位 (LongMedia frozen-overlap 同款)。
    """
    v_t = video_latent_frames(pixel_frames)
    a_t = audio_latent_frames(pixel_frames, fps)

    intermediate_device = comfy.model_management.intermediate_device()
    video_latent = torch.zeros([1, VIDEO_CHANNELS, v_t, latent_h, latent_w],
                               device=intermediate_device)
    audio_latent = torch.zeros([1, AUDIO_CHANNELS, AUDIO_STEREO, a_t],
                               device=intermediate_device)

    locked = False
    if drive_audio_latent is not None:
        fitted = _fit_audio_latent(drive_audio_latent, audio_latent)
        if fitted is not None:
            audio_latent = fitted
            locked = True

    # 段间续接 overlap 头：复制上一段视频尾部 (与 _merge_segment_latents 裁剪账目同源)
    ov_t = 0
    if overlap_video is not None and overlap_frames > 0:
        ctx_v_tokens = video_latent_frames(overlap_frames) if overlap_frames > 5 else 2
        ctx_v_tokens = max(2, ctx_v_tokens)
        ov_t = min(ctx_v_tokens, v_t - 2)
        ov_t = max(0, ov_t)
        if ov_t > 0:
            video_latent[:, :, :ov_t] = overlap_video[:, :, -ov_t:].to(
                device=video_latent.device, dtype=video_latent.dtype)
            print(f"[H3-Auto] 段间续接: overlap 头 mask={float(overlap_video_denoise):.2f} "
                  f"({ov_t} latent token)")

    try:
        combined = comfy.nested_tensor.NestedTensor((video_latent, audio_latent))
    except Exception:
        combined = (video_latent, audio_latent)

    result = {"samples": combined}
    if locked or ov_t > 0:
        video_mask = torch.ones_like(video_latent)
        audio_mask = torch.zeros_like(audio_latent) if locked else torch.ones_like(audio_latent)
        if ov_t > 0:
            video_mask[:, :, :ov_t] = float(overlap_video_denoise)
        try:
            result["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        except Exception:
            result["noise_mask"] = (video_mask, audio_mask)
    return result


def _sample_segment(model, positive, latent_dict, steps, cfg, sampler_name, scheduler, seed,
                    denoise=1.0, sigmas=None, sampler_obj=None):
    """采样一段 latent。denoise<1 或 sigmas 非空时作为二采 (img2img 从 latent_dict 起步)。

    - sampler_obj: 外部 SAMPLER 对象 (可选)，覆盖内置 sampler_name/scheduler
    """
    import latent_preview

    latent_tensor = latent_dict["samples"]
    noise = comfy.sample.prepare_noise(latent_tensor, seed)

    # sigma 优先级最高：传入 sigmas 时接管采样序列；denoise≠1 则 final_sigmas = sigmas * denoise
    use_custom_sigmas = sigmas is not None
    kdenoise = denoise
    if use_custom_sigmas:
        kdenoise = 1.0
        if denoise is not None and denoise < 0.9999:
            sigmas = sigmas * denoise

    # sigmas 传入时 steps 失效：预览步数用 sigma 长度，采样步数完全由 sigmas 决定
    n_steps = max(1, int(len(sigmas)) - 1) if use_custom_sigmas else steps
    x0_output = {}
    callback = latent_preview.prepare_callback(model, n_steps, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    negative = []
    denoise_mask = latent_dict.get("noise_mask")

    if sampler_obj is not None:
        # 外部 SAMPLER 对象 (SamplerCustom 同款)，需显式 sigmas
        if sigmas is None:
            _ks = comfy.samplers.KSampler(
                model, steps=steps, device=model.load_device,
                sampler=sampler_name, scheduler=scheduler,
                denoise=denoise, model_options=model.model_options)
            sigmas = _ks.sigmas
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler_obj, sigmas, positive, negative, latent_tensor,
            noise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    else:
        sampler = comfy.samplers.KSampler(
            model, steps=steps, device=model.load_device,
            sampler=sampler_name, scheduler=scheduler,
            denoise=kdenoise, model_options=model.model_options)
        if use_custom_sigmas:
            samples = sampler.sample(
                noise, positive, negative, cfg=cfg, latent_image=latent_tensor,
                force_full_denoise=True,
                denoise_mask=denoise_mask,
                sigmas=sigmas, callback=callback, disable_pbar=disable_pbar)
        else:
            samples = sampler.sample(
                noise, positive, negative, cfg=cfg, latent_image=latent_tensor,
                force_full_denoise=True, start_step=0, last_step=steps,
                denoise_mask=denoise_mask,
                callback=callback, disable_pbar=disable_pbar)

    # 提取最后一步去噪预测 x0 (干净内容)：含残余噪声的 samples 不能直接当段间 keyframe
    x0 = x0_output.get("x0")
    if x0 is not None:
        if hasattr(samples, "is_nested") and samples.is_nested \
                and not getattr(x0, "is_nested", False):
            latent_shapes = [t.shape for t in samples.unbind()]
            try:
                x0 = comfy.nested_tensor.NestedTensor(
                    comfy.utils.unpack_latents(x0, latent_shapes))
            except Exception:
                pass
        # x0 需 process_latent_out (H3 音频流除以 audio_scale)，与 SamplerCustomAdvanced 的 denoised_output 一致
        try:
            x0 = model.model.process_latent_out(x0.cpu())
        except Exception:
            pass
    return {"samples": samples, "x0": x0}


def _pad_latent_spatial_even(latent_dict):
    """把 latent 的 video 区 H/W 对齐到偶数 (H3 patch_size=(1,2,2) 要求)。

    主 latent 进入模型时会被 ComfyUI pad_to_patch_size 自动补偶，
    但段间续接 keyframe 走 _cond_video_rows 不 pad，奇数会 patchify 崩溃；
    因此二采入口先对齐，保证 keyframe 与主 latent 一致。

    对齐用「复制边缘 (replicate)」而非补零：零行会被 VAE 解码成底部亮斑/伪条带，
    复制相邻真实 latent 行则与内容连续，解码后无伪影。
    """
    v_lat, a_lat = h3_conditioning.unpack_nested_latent(latent_dict)
    if v_lat is None:
        return latent_dict
    h, w = int(v_lat.shape[3]), int(v_lat.shape[4])
    if h % 2 == 0 and w % 2 == 0:
        return latent_dict
    pad_h, pad_w = h % 2, w % 2
    # 5D latent [B,C,T,H,W] 用 replicate/reflect 时 pad 长度必须为 6 (补最后 3 维)，
    # 故 T 维补 0、只补空间 H/W： (W左, W右, H上, H下, T前, T后)
    pad = (0, pad_w, 0, pad_h, 0, 0)
    # replicate 要求补数 < 该维尺寸；极小 latent (单行/单列) 退化补零兜底
    mode = "replicate" if (h > pad_h and w > pad_w) else "constant"
    v_pad = torch.nn.functional.pad(v_lat, pad, mode=mode)
    if a_lat is not None:
        try:
            combined = comfy.nested_tensor.NestedTensor((v_pad, a_lat))
        except Exception:
            combined = (v_pad, a_lat)
    else:
        combined = v_pad
    result = dict(latent_dict)
    result["samples"] = combined
    return result


def _copy_overlap_tail(latent_dict, prev_latent_dict, ov_t):
    """把上一段 samples 的视频尾部复制到当前段 overlap 头 (二采段间续接用)。

    二采的 input latent 可能经过二次放大、结构不完美；段间续接应使用二采本身推理后的
    干净 samples 尾部 (完整去噪、正确结构)，而非 input latent 的放大复制。
    """
    v_cur, a_cur = h3_conditioning.unpack_nested_latent(latent_dict)
    v_prev, _a_prev = h3_conditioning.unpack_nested_latent(prev_latent_dict)
    if v_cur is None or v_prev is None:
        return latent_dict
    n = min(int(ov_t), int(v_cur.shape[2]) - 2, int(v_prev.shape[2]))
    if n <= 0:
        return latent_dict
    v_cur = v_cur.clone()
    v_cur[:, :, :n] = v_prev[:, :, -n:].to(device=v_cur.device, dtype=v_cur.dtype)
    if a_cur is not None:
        try:
            combined = comfy.nested_tensor.NestedTensor((v_cur, a_cur))
        except Exception:
            combined = (v_cur, a_cur)
    else:
        combined = v_cur
    result = dict(latent_dict)
    result["samples"] = combined
    return result


def _with_locked_audio(latent_dict, overlap_video_tokens=0,
                       lock_audio=True, overlap_video_denoise=0.0):
    """二采段 noise_mask：video overlap 头按 mask 去噪 + 可选锁定音频。

    - video: 前 overlap_video_tokens 个 token mask=overlap_video_denoise (0=精确保留、不去噪)，其余 1
    - audio: lock_audio=True 时 mask=0 (原样保留一采音频)，否则 1 (正常去噪)
    """
    samples = latent_dict.get("samples")
    if samples is None:
        return latent_dict
    if hasattr(samples, "is_nested") and samples.is_nested:
        tensors = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        tensors = list(samples)
    else:
        tensors = [samples]
    masks = []
    for t in tensors:
        if t.dim() == 5:
            m = torch.ones_like(t)
            n = min(int(overlap_video_tokens), int(t.shape[2]) - 2)
            if n > 0:
                m[:, :, :n] = float(overlap_video_denoise)
                print(f"[H3-Auto] 段间续接: overlap 头 mask={float(overlap_video_denoise):.2f} "
                      f"({n} latent token, 二采)")
            masks.append(m)
        else:
            masks.append(torch.zeros_like(t) if lock_audio else torch.ones_like(t))
    try:
        combined = comfy.nested_tensor.NestedTensor(tuple(masks))
    except Exception:
        combined = tuple(masks)
    result = dict(latent_dict)
    result["noise_mask"] = combined
    return result


def _split_first_pass_latent(merged_latent, seg_sizes, effective_context, fps):
    """把 merged latent 逆向切成逐段完整 latent (二采起点)，与 _merge_segment_latents 严格互逆。

    merge 时非首段裁掉头部 n_i 个 token (重叠区)，这里用前段尾部 n_i 个 token 重建该头，
    再拼上 merged 中本段新增部分。返回 [{"samples": NestedTensor}, ...] 或 None (账目不匹配)。
    """
    v_all, a_all = h3_conditioning.unpack_nested_latent(merged_latent)
    if v_all is None:
        return None

    n_seg = len(seg_sizes)
    seg_v_tokens = [video_latent_frames(s) for s in seg_sizes]
    seg_a_tokens = [audio_latent_frames(s, fps) for s in seg_sizes]

    ctx_v = video_latent_frames(effective_context) if effective_context > 5 else 2
    ctx_v = max(2, ctx_v)
    ctx_a = max(1, int(effective_context / fps * AUDIO_LATENTS_PER_SEC))

    # 账目校验：merged 各 token 数必须等于各段 (裁重叠后) 之和，否则 info/分段与输入 latent 不一致
    expected_v = seg_v_tokens[0] + sum(
        seg_v_tokens[i] - min(ctx_v, seg_v_tokens[i] - 2) for i in range(1, n_seg))
    if int(v_all.shape[2]) != expected_v:
        print(f"[H3-Auto] 二采切分失败: 视频 token 数 {int(v_all.shape[2])} ≠ 账目 {expected_v}")
        return None
    if a_all is not None:
        expected_a = seg_a_tokens[0] + sum(
            seg_a_tokens[i] - min(ctx_a, seg_a_tokens[i] - 1) for i in range(1, n_seg))
        if int(a_all.shape[-1]) != expected_a:
            print(f"[H3-Auto] 二采切分失败: 音频 token 数 {int(a_all.shape[-1])} ≠ 账目 {expected_a}")
            return None

    seg_v_lats = []
    v_cursor = 0
    for i in range(n_seg):
        v_t = seg_v_tokens[i]
        if i == 0:
            seg = v_all[:, :, v_cursor:v_cursor + v_t, :, :]
            v_cursor += v_t
        else:
            n_i = min(ctx_v, v_t - 2)
            head = v_all[:, :, v_cursor - n_i:v_cursor, :, :]
            body = v_all[:, :, v_cursor:v_cursor + (v_t - n_i), :, :]
            seg = torch.cat([head, body], dim=2)
            v_cursor += (v_t - n_i)
        seg_v_lats.append(seg)

    seg_a_lats = [None] * n_seg
    if a_all is not None:
        a_cursor = 0
        for i in range(n_seg):
            a_t = seg_a_tokens[i]
            if i == 0:
                seg = a_all[..., a_cursor:a_cursor + a_t]
                a_cursor += a_t
            else:
                n_a = min(ctx_a, a_t - 1)
                head = a_all[..., a_cursor - n_a:a_cursor]
                body = a_all[..., a_cursor:a_cursor + (a_t - n_a)]
                seg = torch.cat([head, body], dim=-1)
                a_cursor += (a_t - n_a)
            seg_a_lats[i] = seg

    result = []
    for i in range(n_seg):
        v = seg_v_lats[i]
        a = seg_a_lats[i]
        if a is not None:
            try:
                combined = comfy.nested_tensor.NestedTensor((v, a))
            except Exception:
                combined = (v, a)
        else:
            combined = v
        result.append({"samples": combined})
    return result


def _crossfade_audio(all_waveforms, head_trims, context_frames, total_frames, fps):
    """
    段间音频拼接，与视频账目严格一致：
    - 非首段音频的前 context_frames 对应锚定重叠区 -> 与上一段尾部做 cosine 长淡化
    - head_trims 中超出 context_frames 的部分 (末段保护尾帧的追加裁剪) -> 音频同步裁头
    总时长 = sum(段长) - sum(head_trims) 对应采样数，与视频逐帧对齐。
    """
    if not all_waveforms:
        return {"waveform": torch.zeros(1, 2, 1), "sample_rate": AUDIO_SAMPLE_RATE}

    sr = AUDIO_SAMPLE_RATE

    trimmed = []
    for i, w in enumerate(all_waveforms):
        extra = max(0, head_trims[i] - (context_frames if i > 0 else 0))
        cut = int(round(extra / fps * sr))
        if 0 < cut < w.shape[-1]:
            w = w[..., cut:]
        trimmed.append(w)

    overlap = int(round(context_frames / fps * sr))
    final_wave = trimmed[0]
    for i in range(1, len(trimmed)):
        ov = min(overlap, final_wave.shape[-1] // 2, trimmed[i].shape[-1] // 2)
        if ov > 0:
            final_wave = h3_utils.cosine_crossfade(final_wave, trimmed[i], ov)
        else:
            final_wave = torch.cat([final_wave, trimmed[i]], dim=-1)

    target_samples = int(total_frames / fps * sr)
    final_wave = _align_length(final_wave, target_samples, dim=-1)
    print(f"[H3-Auto] 音频拼接: 重叠淡化={overlap}采样({overlap/sr:.2f}s) "
          f"最终={final_wave.shape[-1]}采样({final_wave.shape[-1]/sr:.2f}s)")
    return {"waveform": final_wave, "sample_rate": sr}


def _align_length(tensor, target, dim):
    """对齐张量长度到 target (截断或补零)"""
    cur = tensor.shape[dim]
    if cur == target:
        return tensor
    if cur > target:
        idx = [slice(None)] * tensor.dim()
        idx[dim] = slice(0, target)
        return tensor[tuple(idx)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target - cur
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim)


def _slice_ref_videos_for_segment(ref_vid_data, start_ratio, end_ratio, vae, audio_vae, device, fps):
    """按当前段比例切参考视频，重新编码后返回 segmented ref 数据。"""
    result = []
    for entry in (ref_vid_data or []):
        pixel = entry.get("pixel")
        if pixel is None or pixel.shape[0] == 0:
            continue
        T = pixel.shape[0]
        s = max(0, min(T - 1, int(T * start_ratio)))
        e = max(s + 1, min(T, int(T * end_ratio)))
        sliced = pixel[s:e]

        n = sliced.shape[0]
        if n < 5:
            print(f"[H3-Auto] 分段参考视频帧数 {n} < 5，跳过")
            continue
        while n % 17 != 5:
            n -= 1
        enc_frames = sliced[:n]

        v_lat = _encode_video_latent(vae, enc_frames, device)
        if v_lat is None:
            continue

        a_lat = None
        aud_dict = entry.get("audio_dict")
        if aud_dict is not None:
            wav = aud_dict.get("waveform")
            if wav is not None and wav.shape[-1] > 0:
                L = wav.shape[-1]
                as_ = max(0, min(L, int(L * start_ratio)))
                ae = max(as_, min(L, int(L * end_ratio)))
                sliced_aud = {"waveform": wav[..., as_:ae],
                              "sample_rate": aud_dict.get("sample_rate", AUDIO_SAMPLE_RATE)}
                a_lat = _encode_audio(audio_vae, sliced_aud, device)

        result.append({
            "pixel": enc_frames,
            "video_latent": v_lat,
            "audio_latent": a_lat,
            "audio_dict": aud_dict,
            "fps": fps,
        })
    return result


def _slice_ref_audios_for_segment(ref_aud_data, start_ratio, end_ratio, audio_vae, device):
    """按当前段比例切独立参考音频，重新编码后返回 segmented ref 数据。"""
    result = []
    for entry in (ref_aud_data or []):
        aud_dict = entry.get("audio_dict")
        if aud_dict is None:
            continue
        wav = aud_dict.get("waveform")
        if wav is None or wav.shape[-1] == 0:
            continue
        L = wav.shape[-1]
        s = max(0, min(L, int(L * start_ratio)))
        e = max(s, min(L, int(L * end_ratio)))
        sliced_aud = {"waveform": wav[..., s:e],
                      "sample_rate": aud_dict.get("sample_rate", AUDIO_SAMPLE_RATE)}
        a_lat = _encode_audio(audio_vae, sliced_aud, device)
        if a_lat is not None:
            result.append({"latent": a_lat, "audio_dict": aud_dict})
    return result


def _merge_segment_latents(all_segments, effective_context, fps):
    """拼接各段 latent，裁掉非首段重演区，打包成 NestedTensor 输出。

    - video latent: 裁掉前 ctx_v_tokens 个 token (对应 effective_context 像素帧)
    - audio latent: 裁掉前 ctx_a_tokens 个 token (对应 effective_context 像素帧)
    - 不做末段额外裁剪 (保护尾帧锚点)，用户自行处理帧数对齐
    """
    v_lats = []
    a_lats = []
    for seg in all_segments:
        v_lat, a_lat = h3_conditioning.unpack_nested_latent(seg)
        if v_lat is not None:
            v_lats.append(v_lat.cpu())
        if a_lat is not None:
            a_lats.append(a_lat.cpu())

    if not v_lats:
        return {"samples": torch.zeros(1, VIDEO_CHANNELS, 2, 16, 16)}

    ctx_v_tokens = video_latent_frames(effective_context) if effective_context > 5 else 2
    ctx_v_tokens = max(2, ctx_v_tokens)

    merged_v = v_lats[0]
    for i in range(1, len(v_lats)):
        v = v_lats[i]
        n = min(ctx_v_tokens, v.shape[2] - 2)
        merged_v = torch.cat([merged_v, v[:, :, n:, :, :]], dim=2)

    merged_a = None
    if a_lats:
        ctx_a_tokens = max(1, int(effective_context / fps * AUDIO_LATENTS_PER_SEC))
        merged_a = a_lats[0]
        for i in range(1, len(a_lats)):
            a = a_lats[i]
            n = min(ctx_a_tokens, a.shape[-1] - 1)
            merged_a = torch.cat([merged_a, a[..., n:]], dim=-1)

    try:
        combined = comfy.nested_tensor.NestedTensor((merged_v, merged_a))
    except Exception:
        combined = (merged_v, merged_a)

    print(f"[H3-Auto] Latent 拼接: video={merged_v.shape[2]} tokens, "
          f"audio={merged_a.shape[-1] if merged_a is not None else 0} tokens")

    return {"samples": combined}
