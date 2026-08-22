<div align="center">

[![Chinese](https://img.shields.io/badge/Language-Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

> One-click MiniMax H3 long-video auto-generation node: **Segmented Inference + Inter-segment Continuation Anchoring + Prompt Timeline Slicing + Second Pass + Seam Correction**.

To address VRAM limitations, long videos are split into multiple segments for independent inference; an overlapping enhancement method ensures seamless transitions between segments, while prompts are automatically segmented along the time axis to align content with the prompt rhythm. A two-stage processing workflow is supported (low-resolution initial sampling followed by high-resolution secondary sampling).

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## 📖 Table of Contents

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Second Pass & SplitSigmas High/Low Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Notes (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter-group node: centrally manages prompt / segmentation / resolution / audio parameters, outputs `parameter`, and previews "expected segmentation" in real time |
| **Minimax_H3_AutoContext_Sampler** | Main node: segmented inference + continuation anchoring + sampling (shared by pass 1 / pass 2) |
| **Minimax_H3_Seam_Correction** | Seam correction node: pixel-domain correction of inter-segment seams in the decoded video |

> Usage: `parameter node --parameter--> main node`. Fill in the prompt in the parameter node; the main node receives it via `parameter` (required).

## <a id="features"></a> ✨ Core Features

### 🧩 Segmented Inference

- Split into multiple segments by `total_frames` / `chunk_frames` (frame unit); frame counts are recommended as 5, 22, 39, 56, 73, 90…
- The last segment is automatically extended to avoid a tiny tail segment
- `fps` is only used for audio synchronization and seconds conversion in the prompt

### 🔗 Inter-segment Continuation

- **Overlapping Enhancement**: non-first segments automatically "relay" the previous segment's ending, so the new content continues naturally from where the previous segment left off, eliminating pauses or position drift at the seams
- The previous segment's ending is passed as a motion reference to the current segment, helping continue the motion direction and speed
- The previous segment's audio is also passed as "previous content", helping the sound continue naturally
- Inter-segment audio is smoothly cross-faded, aligned with the video frame count

> Frame-count rule: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90… (a multiple of 17 plus 5); the node aligns them automatically, so manual calculation is usually unnecessary.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **auto** | Paragraphs with time marks (e.g. `0-5s`) are auto-split; unmarked paragraphs (style / sfx / banned items) are merged into every window |
| **timeline** | Force split by time marks |
| **global** | The whole prompt is used for all windows (suited to a one-shot scene with uniform action) |
| **sequential** | Laid onto the timeline in sentence reading order; paragraphs starting with `global:` / `[global]` are merged into every window |

### 🏷️ Clip_Tag Tag Segmentation Mode

- Split the prompt by user-defined tags (e.g. `段1` / `段2` / `段3`); each segment = one chunk = that segment's full prompt
- Segment duration is determined by the prompt content (three-level priority):
  1. Duration right after the tag line (e.g. `段1:0-5秒` → 5 s; `段1:3-8秒` → 5 s)
  2. The maximum end value of in-segment time marks (e.g. `【0-2秒】` + `【2-5秒】` → 5 s)
  3. `total_frames / fps` default fallback (single segment matches `total_frames`)
- In-segment time marks are **relative time** (each segment starts at 0), not global absolute time
- Non-first segments generate an extra overlap region for continuity, automatically trimmed after generation
- The total duration is automatically aligned to the target total frames, getting as close as possible to the expected duration
- Tags themselves are stripped during inference; the remaining prompt content is output according to `prompt_format`

### 🎯 Intelligent Reference Filtering (Image / Video / Audio)

- Automatically detects which reference images/videos/audios are used in each segment's prompt and passes only those to that segment
- A video is bound to its paired audio track, avoiding image/sound crosstalk

### 🖌️ Second Pass

- Connecting pass-1 latent to the main node's `latent_input` (directly or via a latent upscaler) enters second-pass mode
- Second-pass resolution **follows the input latent** (ignores width/height), enabling low-res pass 1 → high-res pass 2
- `denoise` controls redraw strength; `sigmas` supports custom sigma schedules (same as `SamplerCustomAdvanced`)
- `lock_audio`: pass 2 only redraws video and reuses pass-1 audio

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- When enabled, the video follows this audio track, and the output audio = the source audio itself (lip-sync / rhythm driven by it)

## <a id="install"></a> 📦 Installation

### Method One: Manual Installation

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method Two: Install via Manager

Search `ComfyUI_MinimaxH3_AutoContext` in ComfyUI Manager and click Install.

## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter (parameter-group node)

| Parameter | Default | Description |
|------|--------|------|
| long_prompt | — | Prompt (passed to the main node for inference; also used for the "expected segmentation" preview) |
| prompt_mode | auto | Prompt timeline mode (only effective for Clip_Frame) |
| clip_mode | Clip_Frame | Segmentation mode: Clip_Frame / Clip_Tag |
| clip_tag | 段1 | Clip_Tag split tag template (must end with a number) |
| prompt_format | official | Prompt output format: official / legacy / raw |
| crop_mode | stretch | Reference image / first-last frame / reference video scaling & cropping: center / stretch / none |
| ref_sync_mode | segmented | Whether reference video/audio is sliced per segment: global / segmented |
| width × height | 960×544 | Pass-1 resolution (overridden by latent_input in pass 2) |
| total_frames | 362 | Total generated frames (17n+5); in Clip_Tag mode, segments without an explicit duration fall back to total_frames, and the result is ultimately overridden by the sum of the segments |
| fps | 24 | Frame rate; used for audio sync and prompt seconds conversion |
| chunk_frames | 90 | Frames generated per segment (17n+5) |
| context_frames | 22 | Inter-segment continuation frames (17n+5: 5/22/39/56…) |
| lock_audio | enable | Lock the audio region; only redraw video |
| audio_drive | disable | Audio drive switch |


> The node displays the "expected segmentation" preview in real time (frontend JS calculation; not involved in inference).

### Minimax_H3_AutoContext_Sampler (main node)

| Parameter | Default | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | required | Parameter-group input (from the parameter node) |
| sampler | optional | External sampler object (SAMPLER); overrides the built-in sampler_name/scheduler |
| sigmas | optional | Custom sigma schedule (SIGMAS); highest priority |
| latent_input | optional | Pass-2 input latent (connecting it enables pass 2) |
| info | optional | Parameter inheritance input (chained multi-pass; keeps segmentation consistent) |
| first_frame / last_frame | optional | First / last frame anchoring (FL2VA) |
| video_context_denoise | 0.0 | Inter-segment continuity strength (non-first segments only): 0 = continue exactly from the previous segment's ending, 1 = regenerate, intermediate = soft blend. With SplitSigmas in pass 2, set 1 to avoid artifacts |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw strength (1 = full resample, lower preserves more of the original structure) |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | optional | Reference materials (Autogrow dynamic ports) |
| drive_audio | optional | Audio drive source |

## <a id="output"></a> 📤 Output

| Output | Description |
|------|------|


| **latent** | Spliced audio/video latent; connect to VAE Decode, or upscale then connect to pass 2 |
| **denoised_latent** | Clean latent output, used for pass-2 relay / preview |
| **info** | Segmentation parameters (Dict), passed to the next main node's info input to keep chained passes' segmentation consistent |



## <a id="second-pass"></a> 🔄 Second Pass & SplitSigmas High/Low Frequency

### Basic second pass (low-res pass 1 → high-res pass 2)

```
parameter node ──parameter──> main node (pass 1, 864×480)
    └─ latent / denoised_latent ──> [split AV] ──> video_latent ──> latent upscale ──> [merge AV] ──> main node (pass 2).latent_input
pass-2 node: shared parameter (or info inheritance); optional denoise 0.4~0.6
```

- Pass-2 resolution follows `latent_input`, ignoring the parameter node's width/height

### SplitSigmas high/low frequency (save time, improve clarity)

> ⚠️ **Audio constraint**: high/low frequency **only applies to video** (audio inter-segment continuity needs full sampling), so keep audio fully sampled.

```
Pass-1 node: full sampling (do not connect high_sigmas; audio fully denoised)
          → denoised_latent → split & upscale video (audio untouched) → merge → pass-2.latent_input
Pass-2 node: sigmas ← low_sigmas (only run the low-sigma range to add detail)
          lock_audio = True (reuse the complete pass-1 audio)
          video_context_denoise = 1.0 (redraw the continuation region together with the new content to avoid artifacts)
```

> 💡 **Pass-2 `video_context_denoise`**: with SplitSigmas, setting 0 (exact continuation) may cause artifacts at the boundary between the continued region and the redrawn new content; setting 1.0 redraws the continuation region in sync and avoids them. If the seam is slightly discontinuous, lower it to 0.3~0.5 as a compromise. Pass 1 keeps the default 0.

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

- Input `images` (decoded full video frames, connect to VAE Decode output), output corrected `images`
- **Two-stage correction architecture**:
  - **[Stage 1] Exposure/Color — whole-segment global alignment (the main workhorse)**: an inter-segment seam is essentially a "step" in the temporal signal (level shift / DC step), so the correction must be applied to the entire segment rather than locally blended near the seam (the latter only turns a hard step into a ramp over a few frames, producing a breathing/pumping feel). The transform uses the closed-form MKL (Monge–Kantorovich Linear) solution: the optimal 3×3 linear transform matching mean + covariance, which handles cross-channel color casts (warm/green) and is strictly stronger than per-channel gain/offset
  - **[Stage 2] Motion/Flash — local seam warp/blend**: GPU optical-flow alignment + local blending, run after Stage 1 — once the DC step is removed, the residual at the boundary is the real structural discontinuity
- Seam boundaries are detected **purely from content** (no dependency on the sampler's `info`): precise localization of single-frame transients (flash / segment-head warm-up) + exposure steps (windowed quantile step detection), with no reliance on the segment ledger
- **Shot gate** (`cut_detection`): runs frame-by-frame cut detection over the whole video before correcting; exposure/color/motion corrections apply only within the same shot, and real cuts (including cuts inside a segment) are skipped, so corrections never leak into other shots
- The three treatments are controlled independently and can each be turned off: `fix_color_preset` (exposure/color), `fix_motion_preset` (seam continuity), `fix_flash` (flash frames)

| Parameter | Default | Description |
|------|------|------|
| fix_color_preset | medium | Exposure/color level: off = no processing; low = per-channel brightness gain to remove boundary steps (strength halved, most conservative, never introduces hue shift); medium = MKL linear color transfer (recommended starting point; only fixes the level jump at seams, fully preserving the picture's own brightness changes); high = same as medium but with a longer statistics window (more stable when there is large motion across the boundary); max = full-video per-frame level normalization (the only level that removes "intra-segment drift", but it also flattens the picture's own legitimate brightness changes — nightfall/lights off/tunnel — because the two are the same signal in pixels and cannot be distinguished; ineffective on frames already crushed to near-black, and the log reports the affected frame count) |
| fix_motion_preset | off | Seam-continuity level: off = no processing (recommended default; first try the color level only); low/medium/high/max = optical-flow alignment + local blending — the higher the level, the more frames and the stronger the blend, the smoother the seam but the higher the risk of slight blur or pumping |
| fix_flash | false | Fix flash frames separately (1–2 frame transient brightness jumps at a boundary). It is a separate switch because flash frames go through the local temporal blending logic, which is completely different from whole-segment color transforms; also, when the picture itself has legitimate fast brightness changes (lightning, explosions, light switches), suppressing flash frames flattens those effects too. Still effective when fix_motion_preset=off (uses medium-level parameters) |
| blend_frames | 2 | Seam level-ramp window (frames, 0–8): after exposure alignment, pulls the brightness transition of blend_frames frames on each side of the boundary onto a smooth ramp; larger = smoother and more natural, but too large on motion-heavy shots brings slight blur/breathing; 0 = off |
| cut_detection | true | Shot-detection gate: when enabled, runs frame-by-frame cut detection over the whole video (adjacent-frame RGB histogram correlation); exposure/color/motion corrections apply only within the same shot and real cuts (including cuts inside segments) are skipped; disabled = old behavior, process all segment boundaries |
| cut_threshold | 0.6 | Cut-detection threshold (adjacent-frame RGB histogram correlation, 0–1): correlation below this is judged a cut; higher is more aggressive (fast motion within a shot may be misjudged as a cut), lower is more conservative (cuts may be missed) |
| use_gpu | true | Use CUDA GPU for statistics, color transforms, and optical flow |
| debug | false | Print diagnostics: full-video luma curve, per-segment head/mid/tail drift, per-frame luma around boundaries, 3×3 block luma difference — to identify the cause (step / intra-segment drift / segment-head transient / spatial unevenness). Set all three processing items to off and enable this for "diagnose only, no modification" |

> Usage: `VAE Decode → H3_Seam_Correction → Save/Video`.

> ⚠️ Note: This node only performs frame-level seam correction and cannot fix artifacts produced upstream in the second pass. 

## <a id="prompt-examples"></a> ✍️ Prompt Writing Examples

### Timeline Mode (auto / timeline)

```text
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape
```

> Paragraphs containing `0-5s` marks are split by time; unmarked paragraphs (style / sfx / banned items) are automatically merged into every window.

### Global Mode (global)

> The entire prompt is used for all segments; suited to a one-shot scene with uniform action throughout.

### Clip_Tag Mode (Segment by Tag)

> Set `clip_mode` to `Clip_Tag` and fill in the tag template (`clip_tag`) (must end with a number).

**Tag Template Examples**

| Template | Match |
|------|------|
| `段1` | `段1` / `段2` / `段3` (prefix "段"+number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A"+number) |
| `[片段001]` | `[片段001]` / `[片段002]` (prefix "[片段"+number+suffix"]") |

**Tag Writing**: a tag occupies a line alone as the split point; a line break after the tag is recommended. It also works without a line break (the separator is skipped and the segment content is taken):

```text
段1:3s
视频：
...
音频设计：
...


段2:3-8s
视频：
0-2秒：
...
2-5秒：
...
音频设计：
0-5秒：...
```

**Segment Duration Rules** (three-level priority):

1. Duration right after the tag line: `段1:0-5秒` → 5 s; `段1:3-8秒` → 5 s (the duration mark is removed from the prompt)
2. In-segment time marks, 0-based: `【0-2秒】` + `【2-5秒】` → 5 s
3. Neither present → `chunk_frames / fps` fallback

**prompt_format Selection**

- `official` / `legacy`: in-segment time marks are automatically rendered as in-segment relative coordinates
- `raw`: tags are removed and output as-is; time marks stay unchanged (suited to structured prompts generated by large models)

## <a id="limitations"></a> 📝 Prompt Notes (Node Limitations)

> The following notes **do not apply** to simple, always-valid prompt scenarios (i.e., all segments share the same prompt, global mode),
> e.g. talking-head digital humans (of course, dialogue needs segmentation), videos with little change in shot/composition, or character-replacement scenarios where the prompt stays generic.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented inference (chunking), you must follow the **temporal exclusivity** principle — each segment's prompt can only describe the **new changes** "happening" in that segment relative to the end of the previous segment.

- **Segmentation is a "relay"**: when the Nth segment is generated, its starting visual state (position, pose, camera position) is fully provided implicitly by the "anchoring frames (Context Frames)" at the end of the previous segment. You don't need to re-describe that starting state in the prompt.
- **No "retrospection" or "overlap"**: the Nth segment's prompt must never re-describe actions or camera moves already completed in the N-1th segment. Repeating them makes the model receive instructions that logically conflict with the anchoring frames (instruction conflict), causing stuttering, confused motion logic, or repeated actions.
- **Zero the boundary**: when switching segments, clear the previous segment's "action in progress". The new segment's prompt should be like "new instructions after pressing the shutter" — only describe the displacement, action, or new elements occurring in the new time window.

**❌ Incorrect Writing (Conflict/Overlap)**

```text
Segment 1: 3 seconds
"Object A moves toward position B" 
Segment 2: 3-6 seconds
"After object A moves to position B, it turns around at position B" 
```

> Problem analysis: at the end of segment 1, the anchoring frame shows object A has already arrived at position B and just stopped. But segment 2's prompt forcibly requires "object A to move to position B", which conflicts with the "already arrived" static result of the anchoring frame; the model tries to "move again", causing glitching or skipped frames.

**✅ Correct Writing (Seamless Progression)**

```text
Segment 1: 3 seconds
"Object A moves toward position B and finally stops at position B" (emphasizing action closure)
Segment 2: 3-6 seconds
"After standing still, object A slowly turns its direction" (directly describing the new action after the previous segment ends)
```

> Correct logic: segment 2 completely abandons describing the "moving process", treats "stopped at B" as a given fact, and only describes the following "turning" new action — the model can then continue seamlessly using the anchoring frames.

> 🚀 **One-sentence summary**: the end of the previous segment is the "result", and the start of the next segment is the "new action after the result" — don't write the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Reference Declaration

> When using segmented inference together with reference images/videos (image1, video1, etc.), you must follow the **per-segment reference declaration** principle — every segment's prompt must independently and completely declare all the reference materials it needs; references are not "remembered" or "inherited" into the next segment.

- **No global memory**: the node parses the reference tags written in the current segment's prompt to precisely decide which materials that segment needs. Writing image1 in the previous segment only means the previous segment used it; the next segment is scanned anew.
- **Not written, not passed**: if the Nth segment doesn't mention image1 again, that segment won't receive that reference image, causing character/object inconsistency.

**❌ Incorrect Writing (Implicit Inheritance)**

```text
Segment 1 [3s]: image1 is object A; object A is moving forward.
Segment 2 [3-6s]: Object A stops and turns to face the camera. (image1 not written)
```

**✅ Correct Writing (Explicit Per Segment)**

```text
Segment 1 [3s]: image1 is object A; object A is moving forward.
Segment 2 [3-6s]: image1 is object A; object A stops and turns to face the camera.
```
