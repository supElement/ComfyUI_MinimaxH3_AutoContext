"""
h3_utils.py
处理 MiniMax H3 的物理规则：
- 17n+5 VAE 时序网格吸附 (snap_to_grid 向下, align_frame_count 向上)
- video_latent_t / temporal_shape (官方公式)
- 视频/音频分块计算
- 音频交叉淡化
- 长提示词时间轴切片
"""

import torch
import numpy as np
import re
import math

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
AUDIO_LATENT_FPS = 40


def snap_to_grid(n_frames: int) -> int:
    """
    将任意帧数向下吸附到最近的合法 17n+5 网格点。
    例：输入 30 -> 返回 22；输入 60 -> 返回 56；输入 120 -> 返回 107
    """
    if n_frames < 5:
        return max(1, n_frames)
    n = (n_frames - 5) // 17
    return 17 * n + 5


def snap_to_grid_up(n_frames: int) -> int:
    """
    将帧数向上吸附到最近的 17n+5 网格点。
    例：输入 6 -> 返回 22；输入 50 -> 返回 56；输入 120 -> 返回 124
    """
    if n_frames <= 5:
        return 5
    n = (n_frames + 11) // 17
    return 17 * n + 5


def snap_to_grid_nearest(n_frames: int) -> int:
    """
    吸附到最近的 17n+5 网格点，等距时偏好向上 (减少分段数)。
    例：输入 120 -> 返回 124 (|124-120|=4 < |107-120|=13)
    """
    if n_frames <= 5:
        return 5
    down = snap_to_grid(n_frames)
    up = snap_to_grid_up(n_frames)
    if up - n_frames <= n_frames - down:
        return up
    return down


def align_frame_count(n: int) -> int:
    """向上吸附到 17n+5 (与 snap_to_grid_up 一致，保留旧名称兼容)"""
    return snap_to_grid_up(n)


def video_latent_t(frame_count: int) -> int:
    """
    像素帧数 -> latent 帧数 (官方公式)。
    frame_count <= 5 -> 2, 否则 ((n-5)//17)*5+2
    """
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int, fps: int = FPS):
    """
    根据目标帧数计算完整时序形状 (官方函数)。
    返回 (frame_count, latent_t, audio_t)
    """
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / fps
    audio_t = round(duration * AUDIO_LATENT_FPS)
    return frame_count, video_latent_t(frame_count), audio_t


def steps_for_frames(n: int):
    """计算覆盖 n 个像素帧所需的 latent step 数"""
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def compute_chunks(total_frames: int, chunk_frames: int, context_frames: int = 0):
    """
    分块计算，确保每段帧数符合 17n+5 网格，避免微小尾段。

    关键账目：非首段的前 context_frames 帧被续接锚定占用 (重演上一段尾部)，
    有效新增内容 = 段帧数 - context_frames。

    策略 (用户规则)：锚定亏空优先由**末段加长**吸收，
    仅当末段需加长超过基准段的 30% 时才新增一段。

    返回 (chunks, chunk): chunks 为名义时间轴上的 [(start, end), ...]
    """
    chunk = snap_to_grid_nearest(chunk_frames)
    if chunk < 5:
        chunk = 5
    context = max(0, min(context_frames, chunk - 5))

    if total_frames <= chunk:
        actual = snap_to_grid_up(max(5, total_frames))
        return [(0, actual)], chunk

    max_last = int(chunk * 1.3)

    n = math.ceil(total_frames / chunk)
    while n > 1:
        cov = (n - 2) * chunk + max_last - (n - 2) * context
        if cov >= total_frames:
            n -= 1
        else:
            break

    while True:
        covered = (n - 1) * chunk - (n - 2) * context if n >= 2 else 0
        need_last = total_frames - covered + (context if n >= 2 else 0)
        last = snap_to_grid_up(max(5, need_last))
        if last <= max_last:
            break
        n += 1
    sizes = [chunk] * (n - 1) + [last]

    while len(sizes) >= 2 and sizes[-1] - context < 22:
        merged = snap_to_grid_up(sizes[-2] + sizes[-1] - context)
        if merged > max_last:
            break
        sizes[-2] = merged
        sizes.pop()

    chunks = []
    start = 0
    for s in sizes:
        chunks.append((start, start + s))
        start += s

    return chunks, chunk


def cosine_crossfade(audio_a: torch.Tensor, audio_b: torch.Tensor,
                     overlap_samples: int) -> torch.Tensor:
    if overlap_samples <= 0:
        return torch.cat([audio_a, audio_b], dim=-1)

    overlap_samples = min(overlap_samples, audio_a.shape[-1], audio_b.shape[-1])

    t = torch.linspace(0, math.pi / 2, overlap_samples, device=audio_a.device,
                       dtype=audio_a.dtype)
    fade_out = torch.cos(t)
    fade_in = torch.sin(t)

    head = audio_a[..., :-overlap_samples]
    tail = audio_b[..., overlap_samples:]

    a_overlap = audio_a[..., -overlap_samples:] * fade_out
    b_overlap = audio_b[..., :overlap_samples] * fade_in
    middle = a_overlap + b_overlap

    return torch.cat([head, middle, tail], dim=-1)


_TIME_PATTERN = re.compile(
    r'(?:^|[\s，。；;,.：:、【】\[\]（）()<>「」『』])'
    r'(\d+(?:\.\d+)?)\s*[-–—~至到]\s*(\d+(?:\.\d+)?)\s*[s秒]'
    r'\s*[:：]?\s*'
)

_STRIP_CHARS = ' \t，,；;:：【】[]（）()「」『』'
_MARK_EDGE_CHARS = ' \t:：【】[]（）()「」『』'


def _parse_timeline_in_paragraph(para: str):
    """解析单个段落内的时间标记。无标记返回 None。
    返回 (preamble, [(start, end, label, bracket), ...])：
    - preamble: 首个时间标记之前的同行文本
    - label: 时间标记后的同行描述 (仅剥离结构字符, 内部标点原样保留)
    - bracket: 原标记是否带括号 (重组时保持原格式)
    """
    matches = list(_TIME_PATTERN.finditer(para))
    if not matches:
        return None

    preamble = para[:matches[0].start()].strip()

    segments = []
    for i, m in enumerate(matches):
        start = float(m.group(1))
        end = float(m.group(2))
        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(para)
        label = para[text_start:text_end].strip(_MARK_EDGE_CHARS)
        digit_pos = m.start(1)
        prev_char = para[digit_pos - 1] if digit_pos > 0 else ''
        bracket = bool(prev_char) and prev_char in '【[（(「『'
        segments.append((start, end, label, bracket))
    return preamble, segments


def _fulltext_segment(text: str):
    return {"start": 0.0, "end": float('inf'), "label": text.strip(),
            "content": "", "block": None, "bracket": False}


_SHOT_PATTERN = re.compile(
    r'\[Shot\s*(\d+)\](?:\s*At\s*(\d+):(\d+(?:\.\d+)?))?\s*,?\s*')

_LINE_START_TIME = re.compile(
    r'^\s*[【\[]?\s*\d+(?:\.\d+)?\s*[-–—~至到]\s*\d+(?:\.\d+)?\s*[s秒]')
_LINE_START_SHOT = re.compile(r'^\s*\[Shot\s*\d+\]')


def _starts_with_time_mark(para: str) -> bool:
    """行首即时间标记 (允许前置【/[), 用于区分全局区块内嵌的标记文本
    如 retention_analysis 的 "出现于 [Shot 1]")。"""
    return bool(_LINE_START_TIME.match(para) or _LINE_START_SHOT.match(para))


def _parse_shots_in_paragraph(para: str):
    """解析官方 Shot 格式: [Shot 1] (无时间戳, 视为 0 秒) / [Shot N] At MM:SS.mmm。
    返回 (preamble, [(start, content), ...]); end 一律为 inf, 由外层在遇到
    下一个 Shot 时回填 (最后一个回填为 total_seconds)。"""
    matches = list(_SHOT_PATTERN.finditer(para))
    if not matches:
        return None
    preamble = para[:matches[0].start()].strip()
    segments = []
    for i, m in enumerate(matches):
        if m.group(2) is not None:
            start = int(m.group(2)) * 60 + float(m.group(3))
        else:
            start = 0.0
        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(para)
        content = para[text_start:text_end].strip()
        segments.append((start, content))
    return preamble, segments


def parse_prompt_schedule(long_prompt: str):
    """整段文本按时间标记切分 (无标记则全文为一段)。兼容旧接口。"""
    parsed = _parse_timeline_in_paragraph(long_prompt)
    if parsed is None:
        return [_fulltext_segment(long_prompt)]
    preamble, segments = parsed
    result = []
    for s, e, label, bracket in segments:
        if preamble:
            head = preamble.strip(_MARK_EDGE_CHARS)
            label = (head + "，" + label) if label else head
        result.append({"start": s, "end": e, "label": label,
                       "content": "", "block": None, "bracket": bracket})
    return result


_CLAUSE_SPLIT = re.compile(r'[。；;！!？?\n]+')
_GLOBAL_MARK = re.compile(
    r'^\s*(?:[\[【(]\s*(?:全局|global)[^\]】)\n]*[\]】)]|(?:全局|global))\s*[:：]?\s*',
    re.IGNORECASE)


def _strip_global_mark(text: str) -> str:
    """剥掉全局标记前缀 (不发给模型)"""
    return _GLOBAL_MARK.sub('', text).strip()


def _is_block_title(text: str) -> bool:
    """纯标题行判定: 冒号结尾、冒号后无内容 (如 视频：/音频设计：)"""
    t = text.strip()
    return len(t) >= 2 and t[-1] in '：:'


def split_prompt_clauses(text: str):
    """按句读拆分提示词为有序子句"""
    parts = _CLAUSE_SPLIT.split(text)
    return [p.strip().strip(',，') for p in parts if p and p.strip().strip(',，')]


def _split_paragraphs(text: str):
    return [p.strip() for p in re.split(r'\n+', text) if p.strip()]


def build_prompt_schedule(long_prompt: str, total_seconds: float, mode: str = "auto", quiet: bool = False):
    """
    结构化构建提示词时间轴 (适配官方 skill 分节模板)。

    规则：按换行分段落 —
      - 含时间标记 (0-5s ...) 的段落 -> 时间轴段落，解析切片
      - 无时间标记的段落 -> 全局段落，拼入每个窗口
        (时间轴之前的全局段放在窗口提示词前面，之后的放后面)

    mode:
      - "timeline":   按时间标记切分 (无标记时降级为 global)
      - "global":     整段提示词用于所有窗口 (适合全程同质动作的一镜到底)
      - "sequential": 按句读顺序铺到时间轴；以 "全局:"/"[全局]"/"[global]" 开头
                      的段落不平铺，拼入每个窗口
      - "auto":       有时间标记 -> timeline，否则 -> global

    返回 dict: {"segments": [...], "prefix": str, "suffix": str, "mode": str}
    """
    def _result(segments, prefix, suffix, used):
        return {"segments": segments, "prefix": prefix, "suffix": suffix, "mode": used}

    paragraphs = _split_paragraphs(long_prompt)

    timeline_segs = []
    globals_before, globals_after = [], []
    seen_timeline = False
    in_global = False
    last_seg_idx = None
    current_block = None
    pending_title = None
    pending_lines = []

    def _append_global(text):
        (globals_after if seen_timeline else globals_before).append(text)

    def _flush_pending():
        nonlocal pending_title
        if pending_title is not None:
            _append_global(pending_title)
            pending_title = None
        while pending_lines:
            _append_global(pending_lines.pop(0))

    for para in paragraphs:
        if _GLOBAL_MARK.match(para):
            _flush_pending()
            in_global = True
            last_seg_idx = None
            current_block = None
            _append_global(_strip_global_mark(para))
            continue
        if in_global:
            if _starts_with_time_mark(para):
                pass
            elif _is_block_title(para):
                _flush_pending()
                in_global = False
                pending_title = para.strip()
                pending_lines.clear()
                continue
            else:
                _append_global(_strip_global_mark(para))
                continue
        parsed = _parse_timeline_in_paragraph(para)
        shot_parsed = None if parsed else _parse_shots_in_paragraph(para)
        if parsed or shot_parsed:
            used_pending = pending_title is not None
            if used_pending:
                line_block = pending_title
                current_block = pending_title
                pending_title = None
                preamble_texts = list(pending_lines)
                pending_lines.clear()
            else:
                line_block = current_block
                preamble_texts = []
            in_global = False
            if parsed:
                preamble, segs = parsed
                if preamble and not used_pending and _is_block_title(preamble):
                    line_block = preamble.strip()
                    current_block = line_block
                elif preamble:
                    head = preamble.strip(_MARK_EDGE_CHARS)
                    segs = [(s, e, (head + "，" + label) if label else head, br)
                            for s, e, label, br in segs]
                for s, e, label, bracket in segs:
                    timeline_segs.append({
                        "start": s, "end": e, "label": label,
                        "content": "", "block": line_block, "bracket": bracket,
                    })
            else:
                preamble, shot_segs = shot_parsed
                if preamble and not used_pending and _is_block_title(preamble):
                    line_block = preamble.strip()
                    current_block = line_block
                elif preamble:
                    preamble_texts.append(preamble)
                if (last_seg_idx is not None and shot_segs
                        and timeline_segs[last_seg_idx]["end"] == float('inf')):
                    timeline_segs[last_seg_idx]["end"] = shot_segs[0][0]
                for start, content in shot_segs:
                    timeline_segs.append({
                        "start": start, "end": float('inf'),
                        "label": "", "content": content,
                        "block": line_block, "bracket": False,
                    })
            seen_timeline = True
            last_seg_idx = len(timeline_segs) - 1
            if preamble_texts:
                seg = timeline_segs[last_seg_idx]
                pre = "\n".join(preamble_texts)
                seg["content"] = (pre + "\n" + seg["content"]) if seg["content"] else pre
            continue
        if _is_block_title(para) and not in_global:
            _flush_pending()
            pending_title = para.strip()
            pending_lines.clear()
            continue
        if pending_title is not None and not in_global:
            pending_lines.append(para.strip())
            continue
        if in_global:
            _append_global(_strip_global_mark(para))
        elif last_seg_idx is not None:
            seg = timeline_segs[last_seg_idx]
            seg["content"] = (seg["content"] + "\n" + para.strip()) if seg["content"] else para.strip()
        else:
            _append_global(_strip_global_mark(para))
    _flush_pending()

    for seg in timeline_segs:
        if seg["end"] == float('inf'):
            seg["end"] = total_seconds

    has_timeline = len(timeline_segs) > 0
    g_before = "\n".join(globals_before)
    g_after = "\n".join(globals_after)

    if mode == "timeline" or (mode == "auto" and has_timeline):
        if has_timeline:
            return _result(timeline_segs, g_before, g_after, "timeline")
        if not quiet:
            print("[H3-Auto] 提示: 未检测到时间标记 (如 '0-5s')，降级为 global 模式")
        return _result([_fulltext_segment(long_prompt)], "", "", "global")

    if mode == "sequential":
        narrative_paras = []
        seq_globals = []
        seq_segments = []
        in_global_section = False
        last_seq_idx = None
        seq_block = None
        seq_pending = None

        def _flush_seq_pending():
            nonlocal seq_pending
            if seq_pending is not None:
                (seq_globals if in_global_section else narrative_paras).append(seq_pending)
                seq_pending = None

        for para in paragraphs:
            if _GLOBAL_MARK.match(para):
                _flush_seq_pending()
                seq_globals.append(_strip_global_mark(para))
                in_global_section = True
                last_seq_idx = None
                seq_block = None
                continue
            parsed = _parse_timeline_in_paragraph(para)
            if parsed:
                preamble, segs = parsed
                line_block = seq_block
                if seq_pending is not None:
                    line_block = seq_pending
                    seq_block = seq_pending
                    seq_pending = None
                elif preamble and _is_block_title(preamble):
                    line_block = preamble.strip()
                    seq_block = line_block
                elif preamble:
                    head = preamble.strip(_MARK_EDGE_CHARS)
                    segs = [(s, e, (head + "，" + label) if label else head, br)
                            for s, e, label, br in segs]
                for s, e, label, bracket in segs:
                    seq_segments.append({
                        "start": s, "end": e, "label": label,
                        "content": "", "block": line_block, "bracket": bracket,
                    })
                in_global_section = False
                last_seq_idx = len(seq_segments) - 1
                continue
            if _is_block_title(para) and not in_global_section:
                _flush_seq_pending()
                seq_pending = para.strip()
                continue
            _flush_seq_pending()
            if in_global_section:
                seq_globals.append(para)
            elif last_seq_idx is not None:
                seg = seq_segments[last_seq_idx]
                seg["content"] = (seg["content"] + "\n" + para.strip()) if seg["content"] else para.strip()
            else:
                narrative_paras.append(para)
        _flush_seq_pending()
        clauses = []
        for para in narrative_paras:
            clauses.extend(split_prompt_clauses(para))
        if clauses:
            step = total_seconds / len(clauses)
            for i, c in enumerate(clauses):
                seq_segments.append({
                    "start": i * step, "end": (i + 1) * step, "label": c,
                    "content": "", "block": None, "bracket": False,
                })
        if not seq_segments:
            return _result([_fulltext_segment(long_prompt)], "", "", "global")
        return _result(seq_segments, "", "\n".join(seq_globals), "sequential")

    return _result([_fulltext_segment(long_prompt)], "", "", "global")


def _snap_half(t: float) -> float:
    """时间吸附到 0.5 秒网格"""
    return round(t * 2) / 2


def _fmt_time(t: float) -> str:
    """整数秒不带小数点，否则保留一位小数"""
    return str(int(t)) if float(t).is_integer() else f"{t:.1f}"


_CLAUSE_KEEP = re.compile(r'[^。；;！!？?，,、\n]+[。；;！!？?，,、]*')


def _split_clauses_keep_punct(text: str):
    """按标点切分子句，子句尾部标点原样保留"""
    return [c for c in _CLAUSE_KEEP.findall(text) if c.strip()]


def _filter_clauses_for_window(text: str, seg_s: float, seg_e: float,
                               clip_s: float, clip_e: float) -> str:
    """子句均匀铺到段时间轴，保留中点落在 [clip_s, clip_e) 内的子句。
    无标点可切 (单个子句) 时整段保留。"""
    clauses = _split_clauses_keep_punct(text)
    if len(clauses) <= 1:
        return text
    step = (seg_e - seg_s) / len(clauses)
    if step <= 0:
        return text
    kept = []
    for i, c in enumerate(clauses):
        mid = seg_s + (i + 0.5) * step
        if clip_s <= mid < clip_e:
            kept.append(c)
    if not kept:
        clip_mid = (clip_s + clip_e) / 2
        idx = min(range(len(clauses)),
                  key=lambda i: abs(seg_s + (i + 0.5) * step - clip_mid))
        kept = [clauses[idx]]
    return "".join(kept)


def _render_segment(seg, rel_s: float, rel_e: float, clip_s: float, clip_e: float) -> str:
    """渲染单个时间段: 时间转窗口内相对值，内容按标点切分筛选，格式保持原文样式"""
    time_str = f"{_fmt_time(rel_s)}-{_fmt_time(rel_e)}秒"
    label = seg["label"]
    content = seg["content"]
    if seg["bracket"]:
        head = f"【{time_str}：{label}】" if label else f"【{time_str}】"
        if content:
            body = _filter_clauses_for_window(content, seg["start"], seg["end"], clip_s, clip_e)
            return head + "\n" + body
        return head
    if content:
        body = _filter_clauses_for_window(content, seg["start"], seg["end"], clip_s, clip_e)
        head = f"{time_str}：{label}" if label else time_str
        return head + "\n" + body
    body = _filter_clauses_for_window(label, seg["start"], seg["end"], clip_s, clip_e)
    return f"{time_str}：{body}" if body else time_str


def _fmt_shot_time(t: float) -> str:
    """秒 -> MM:SS.mmm (官方切镜时间格式, 如 00:03.500)"""
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:06.3f}"


def _render_segment_official(seg, rel_s: float, clip_s: float, clip_e: float,
                             shot_idx: int) -> str:
    """官方格式渲染单个段: [Shot 1] 无时间戳, 后续 [Shot N] At MM:SS.mmm"""
    label = seg["label"]
    content = seg["content"]
    if content:
        body = _filter_clauses_for_window(content, seg["start"], seg["end"], clip_s, clip_e)
        text = (label + "：" + body) if label else body
    else:
        text = _filter_clauses_for_window(label, seg["start"], seg["end"], clip_s, clip_e)
    if shot_idx == 1:
        return f"[Shot 1] {text}"
    return f"[Shot {shot_idx}] At {_fmt_shot_time(rel_s)}, {text}"


def _compose_official(selected, schedule, win_start_sec: float) -> str:
    """官方格式组装:
    - 视频段 -> integrated_multimodal_description: ([Shot N] At MM:SS.mmm 结构)
    - 音频段 (块标题含"音") -> overall_soundscape: (去时间前缀, 合并为连续段落)
    - 配乐段 (块标题含"配乐"或"music") -> non_diegetic_music: (去时间前缀, 合并)
    - 全局段落拼在最后
    """
    shot_lines = []
    sound_parts = []
    music_parts = []
    for seg, cs, ce in selected:
        b = seg["block"]
        if b and ("配乐" in b or "music" in b.lower()):
            body = _filter_clauses_for_window(
                seg["content"] if seg["content"] else seg["label"],
                seg["start"], seg["end"], cs, ce)
            if body.strip():
                music_parts.append(body.strip())
            continue
        if b and ("音" in b or "soundscape" in b.lower() or "sound" in b.lower()):
            body = _filter_clauses_for_window(
                seg["content"] if seg["content"] else seg["label"],
                seg["start"], seg["end"], cs, ce)
            if body.strip():
                sound_parts.append(body.strip())
            continue
        shot_idx = len(shot_lines) + 1
        rel_s = cs - win_start_sec
        shot_lines.append(_render_segment_official(seg, rel_s, cs, ce, shot_idx))

    parts = []
    if shot_lines:
        parts.append("integrated_multimodal_description: " + " ".join(shot_lines))
    if sound_parts:
        parts.append("overall_soundscape: " + "".join(sound_parts))
    if music_parts:
        parts.append("non_diegetic_music: " + "".join(music_parts))
    base = "\n\n".join(parts)
    all_parts = [p for p in (base, schedule.get("prefix", ""), schedule.get("suffix", "")) if p]
    return "\n\n".join(all_parts)


def compose_window_prompt(schedule, win_start_sec: float, win_end_sec: float,
                          fmt: str = "legacy") -> str:
    """
    组合窗口提示词：
    - 时间轴段与窗口求交集，时间标记转为窗口内相对时间 (0.5 秒网格)
    - 段内容按标点切分子句，中点落在交集内的保留 (无标点则整段保留)
    - 按块标题 (视频：/音频设计：等) 分组输出，视频/音频不混排
    - 全局段落 (已剥离【全局】标记，其余标点原样) 拼在最后

    fmt:
      - "legacy":   【0-3秒：label】+ 块标题 (自定义格式, 相对时间吸附 0.5 秒网格)
      - "official": [Shot N] At MM:SS.mmm + integrated_multimodal_description /
                    overall_soundscape 字段 (MiniMax H3 官方训练格式,
                    时间为帧对齐精确值, 不吸附)
    """
    if schedule.get("mode") == "global":
        seg = schedule["segments"][0]
        text = seg["label"] + ("\n" + seg["content"] if seg["content"] else "")
        parts = [p for p in (text, schedule.get("prefix", ""), schedule.get("suffix", "")) if p]
        return "\n\n".join(parts)

    segments = schedule["segments"]
    win_dur = max(1e-6, win_end_sec - win_start_sec)

    selected = []
    for seg in segments:
        cs = max(seg["start"], win_start_sec)
        ce = min(seg["end"], win_end_sec)
        if ce - cs >= 0.25 * win_dur:
            selected.append((seg, cs, ce))

    if not selected and segments:
        win_mid = (win_start_sec + win_end_sec) / 2
        best = None
        for seg in segments:
            if seg["start"] <= win_mid < seg["end"]:
                best = seg
                break
        if best is None:
            best = min(segments,
                       key=lambda s: abs((s["start"] + s["end"]) / 2 - win_mid))
        cs = max(best["start"], win_start_sec)
        ce = min(best["end"], win_end_sec)
        selected = [(best, cs, ce)]

    if fmt == "official":
        return _compose_official(selected, schedule, win_start_sec)

    block_order = []
    block_lines = {}
    for seg, cs, ce in selected:
        rel_s = _snap_half(cs - win_start_sec)
        rel_e = _snap_half(ce - win_start_sec)
        text = _render_segment(seg, rel_s, rel_e, cs, ce)
        b = seg["block"]
        if b not in block_lines:
            block_lines[b] = []
            block_order.append(b)
        block_lines[b].append(text)

    parts = []
    for b in block_order:
        lines = block_lines[b]
        if b:
            parts.append(b + "\n" + "\n".join(lines))
        else:
            parts.append("\n".join(lines))

    base = "\n\n".join(p for p in parts if p)
    all_parts = [p for p in (base, schedule.get("prefix", ""), schedule.get("suffix", "")) if p]
    return "\n\n".join(all_parts)


def get_prompt_for_window(segments, win_start_sec: float, win_end_sec: float) -> str:
    """旧接口: 纯文本拼接 (不含块结构/时间标记)。新代码请用 compose_window_prompt。"""
    def _triplet(seg):
        if isinstance(seg, dict):
            text = seg["label"] + ("\n" + seg["content"] if seg["content"] else "")
            return seg["start"], seg["end"], text
        return seg

    triplets = [_triplet(s) for s in segments]
    win_dur = max(1e-6, win_end_sec - win_start_sec)
    win_mid = (win_start_sec + win_end_sec) / 2.0

    significant = []
    for start, end, text in triplets:
        overlap = min(end, win_end_sec) - max(start, win_start_sec)
        if overlap >= 0.25 * win_dur:
            significant.append(text)

    if significant:
        return ", ".join(significant)

    for start, end, text in triplets:
        if start <= win_mid < end:
            return text
    closest = min(triplets, key=lambda s: abs((s[0] + s[1]) / 2 - win_mid))
    return closest[2]


# ==================== Clip_Tag 模式 ====================

_TAG_TRAILING_SEPARATORS = ':：，,。；; \t—–-'
_TAG_DURATION_PATTERN = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*[s秒]\s*[:：\s]*')
_TAG_BRACKET_RE = re.compile(r'[【】\[\]「」『』（）()<>]')


def _parse_tag_line_duration(text):
    """从标签行剩余内容中解析时长，支持单时长/时间范围/带括号。

    优先级: 时间范围 (0-3秒/3-8s) > 单时长 (3秒/5s)
    带括号的先剥括号再解析。
    返回 (duration, remaining_text)，无时长返回 (None, text)
    """
    cleaned = _TAG_BRACKET_RE.sub('', text).strip()

    m = _TIME_PATTERN.search(cleaned)
    if m:
        start = float(m.group(1))
        end = float(m.group(2))
        remaining = (cleaned[:m.start()] + cleaned[m.end():]).strip()
        return end - start, remaining

    m = _TAG_DURATION_PATTERN.match(cleaned)
    if m:
        return float(m.group(1)), cleaned[m.end():].strip()

    return None, text


def _parse_tag_pattern(clip_tag_input):
    """解析用户填写的标签模板，拆成 (prefix, suffix)。

    从右往左：先跳过尾部非数字 (suffix)，再找连续数字，剩下的是 prefix。
        '段1'       -> ('段', '')
        'A01'       -> ('A', '')
        '[片段001]' -> ('[片段', ']')
    """
    s = clip_tag_input.strip()
    i = len(s) - 1
    while i >= 0 and not s[i].isdigit():
        i -= 1
    if i < 0:
        raise ValueError(f"标签模板 '{clip_tag_input}' 未包含数字序号")
    num_end = i + 1
    while i >= 0 and s[i].isdigit():
        i -= 1
    num_start = i + 1
    prefix = s[:num_start]
    suffix = s[num_end:]
    return prefix, suffix


def _split_by_tag(long_prompt, prefix, suffix):
    """按标签行切分提示词。

    标签必须独占一行 (行首允许空格)，正则: ^prefix(\\d+)suffix + 可选分隔符 + 可选同行内容
    标签后跳过分隔符取段内容，同行内容若以时长开头则提取时长。

    返回 (prefix_text, [(seg_no, tag_duration, seg_text), ...])
    - prefix_text: 第一个标签之前的内容
    - tag_duration: 标签行紧跟的时长 (秒)，None 表示未指定
    - seg_text: 段内容 (已去标签行、去标签行时长标记)
    """
    esc_prefix = re.escape(prefix)
    esc_suffix = re.escape(suffix)
    tag_re = re.compile(
        r'^[ \t]*' + esc_prefix + r'(\d+)' + esc_suffix + r'[ \t]*'
        r'([' + re.escape(_TAG_TRAILING_SEPARATORS) + r']*)[ \t]*(.*)$',
        re.MULTILINE)

    matches = list(tag_re.finditer(long_prompt))
    if not matches:
        return long_prompt, []

    prefix_text = long_prompt[:matches[0].start()]

    segments = []
    for i, m in enumerate(matches):
        seg_no = int(m.group(1))
        rest_of_line = m.group(3) or ''

        tag_duration, rest_of_line = _parse_tag_line_duration(rest_of_line)

        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(long_prompt)
        following = long_prompt[content_start:content_end]

        if rest_of_line and following:
            seg_text = rest_of_line + '\n' + following
        elif rest_of_line:
            seg_text = rest_of_line
        else:
            seg_text = following

        seg_text = seg_text.strip('\n').strip()
        segments.append((seg_no, tag_duration, seg_text))

    return prefix_text, segments


def _compute_segment_duration(seg_text, default_seconds):
    """段内时间标记的最大结束值；无标记则用默认值。"""
    max_end = None
    for m in _TIME_PATTERN.finditer(seg_text):
        end = float(m.group(2))
        if max_end is None or end > max_end:
            max_end = end
    if max_end is not None:
        return max_end
    return default_seconds


def build_tag_schedule(long_prompt, clip_tag_input, default_seconds):
    """Clip_Tag 模式：按自定义标签切分提示词，确定每段时长。

    时长三层优先级：
      1. 标签行紧跟的时长 (如 '段1 5秒' -> 5秒)
      2. 段内时间标记的最大结束值 (如 【0-2秒】+【2-5秒】-> 5秒)
      3. default_seconds (即 chunk_seconds)

    返回 {"segments": [(seg_text, seg_seconds), ...], "prefix": str}
    """
    prefix, suffix = _parse_tag_pattern(clip_tag_input)
    prefix_text, raw_segments = _split_by_tag(long_prompt, prefix, suffix)

    if not raw_segments:
        print(f"[H3-Auto] Clip_Tag: 未在提示词中找到标签 '{clip_tag_input}'，降级为整段")
        return {"segments": [(long_prompt.strip(), default_seconds)], "prefix": ""}

    segments = []
    for seg_no, tag_duration, seg_text in raw_segments:
        if tag_duration is not None:
            duration = tag_duration
        else:
            duration = _compute_segment_duration(seg_text, default_seconds)

        if duration > 15:
            print(f"[H3-Auto] 警告: 段 {seg_no} 时长 {duration}s 超过 15s "
                  f"(MiniMax H3 官方推荐时长，特殊情况下可生成更长)")

        segments.append((seg_text, duration))

    total = sum(d for _, d in segments)
    durations_str = ", ".join(f"{d}s" for _, d in segments)
    print(f"[H3-Auto] Clip_Tag 分段: {len(segments)} 段 [{durations_str}]，合计 {total}s")
    if prefix_text.strip():
        print(f"[H3-Auto] Clip_Tag 全局前缀: {prefix_text.strip()[:80]}...")

    return {"segments": segments, "prefix": prefix_text.strip()}


def _snap_to_17_multiple_nearest(n_frames):
    """吸附到最近的 17 的倍数 (非首段新增帧数用，无 5 帧头部)。"""
    n_frames = max(5, int(n_frames))
    k = round(n_frames / 17.0)
    return max(17, k * 17)


def compute_tag_chunks(seg_target_frames_list, context_frames):
    """Clip_Tag 模式的分块计算 (每段新增帧数贴目标 + 总时长锚定补偿)。

    帧数账目 (H3 17n+5 网格约束)：
    - 首段 (无锚定): 段长 = 新增 = snap_to_grid_nearest(target)  (17n+5 形式)
    - 非首段: 段长 = 新增 + context_frames；由于 context_frames 取 17n+5 形式，
      新增必然是 17 的倍数 (段长 17n+5 − context 17n+5 = 17 倍数)，如 68/85/102
    - 总时长补偿: 各段新增之和与目标总帧数之差，按 17 帧粒度从未段起逐段 ±17，
      使总时长尽量贴近用户目标 (不再像旧版那样每段都向下缩水 4 帧)

    返回 (chunks, seg_sizes)
    - chunks: [(start, end), ...] 名义累积坐标
    - seg_sizes: [seg_size, ...] 实际生成帧数
    """
    targets = [max(5, int(t)) for t in seg_target_frames_list]
    n = len(targets)
    if n == 0:
        return [], []

    new_frames = [snap_to_grid_nearest(targets[0])]
    for t in targets[1:]:
        new_frames.append(_snap_to_17_multiple_nearest(t))

    target_total = sum(targets)
    total_new = sum(new_frames)
    diff = target_total - total_new

    if diff != 0 and n >= 2:
        k = int(round(diff / 17.0))
        step = 17 if k > 0 else -17
        idx = n - 1
        remaining = abs(k)
        while remaining > 0 and idx >= 0:
            if new_frames[idx] + step >= 5:
                new_frames[idx] += step
                remaining -= 1
            idx -= 1
        total_new = sum(new_frames)

    seg_sizes = [new_frames[0]]
    for i in range(1, n):
        seg_sizes.append(new_frames[i] + context_frames)

    chunks = []
    start = 0
    for s in seg_sizes:
        chunks.append((start, start + s))
        start += s

    print(f"[H3-Auto] Clip_Tag 分块: 目标总帧数={target_total} "
          f"实际输出={total_new} (偏差 {total_new - target_total:+d}帧)")

    return chunks, seg_sizes


def render_tag_segment(seg_text, seg_seconds, fmt, prefix=""):
    """渲染 Clip_Tag 模式下的单段提示词。

    - raw: 去标签后原样输出 (段内相对时间标记保持不变)
    - legacy/official: 对单段独立调 build_prompt_schedule + compose_window_prompt
      时间坐标自动变段内相对 (窗口从 0 到 seg_seconds)
    - prefix (全局前缀) 拼在每段末尾
    """
    if fmt == "raw":
        parts = [p for p in (seg_text, prefix) if p]
        return "\n\n".join(parts)

    schedule = build_prompt_schedule(seg_text, seg_seconds, mode="timeline", quiet=True)
    window_prompt = compose_window_prompt(schedule, 0.0, seg_seconds, fmt=fmt)
    if prefix:
        window_prompt = window_prompt + "\n\n" + prefix
    return window_prompt