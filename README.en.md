<div align="center">

[![Chinese](https://img.shields.io/badge/Language-Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

A one-click automated generation node for long videos using MiniMax H3, featuring: segmented inference, seamless inter-segment anchoring, timeline-based prompt slicing, secondary sampling, and seam correction. To handle VRAM limitations, the process splits long videos into independent inference segments and achieves seamless transitions via overlap-based enhancement. Prompts are automatically sliced ​​along the timeline to align generated content with the prompt's rhythm; audio-visual references undergo corresponding slicing and alignment, ensuring only the relevant reference for the current segment is used during inference. Secondary sampling is supported, as are video extension (continuation), video prepending, and dual-video merging. The system supports latent cache management, allowing for quick resumption after interruptions by skipping already-processed segments; cache files are stored segment-by-segment, enabling the reuse of existing latent cache files provided the upstream sampling parameters remain unchanged.

Note: Changing models—or acceleration nodes like LoRA or SageAttention—will not be detected by the latent check mechanism; therefore, the latent cache must be deleted. Two methods exist to delete the latent cache:
- Enable the `clear_cache` parameter on the `Minimax_H3_AutoContext_Sampler` node; this forces the node to regenerate its cache file when sampling begins.
- Manually delete the corresponding folder in the cache directory (`\ComfyUI\output\cache`); the folder is named "node_" followed by the "Node ID".
- `ignore_latent_hash`: Ignores hash verification for the `input_latent` port. Use case: Certain latent processing nodes (e.g., `Minimax H3 Latent Upscaler (3D)`) alter latent metadata, causing minor changes that render the latent cache unusable and waste inference time; setting this to `true` is recommended in such cases. I have only tested this with the `Minimax_H3-LatentUpscaler_Adv` node from my other repository (`github.com/supElement/ComfyUI_Element_easy`); other similar nodes have not been tested. When using latent processing nodes that do not alter latent noise characteristics, the `ignore_latent_hash` parameter can be set to `false`.

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## Bug fixes and optimizations

V0.7.1

- The `video_guide` parameter has been added to optimize video continuation, video extrapolation, and dual-video bridging (generating intermediate segments), with support for segmentation. Note: When this parameter is not set to `none`, the reference length at the corresponding reference port of the sampling node is forcibly truncated to the value specified by the `context_frames` parameter. The reference logic remains the same as for standard references (i.e., the reference is utilized only if declared in the prompt).

V0.6.5

- Optimized latent cache handling logic: removed the manual cache directory specification and switched to automatically assigning a unique cache directory (format: "node + node ID") for each node. This prevents accidental overwriting of latent caches between sampling nodes.
- Implemented segmented cache and validation logic: if an upstream node merely adds a prompt or a new segment—without altering prompts for existing segments submitted to the sampler or changing parameters linked to the sampling node—the existing cache remains valid and is reused, while a new latent cache is automatically created for the added segment. Downstream - sampling nodes (e.g., a second sampling pass) also retain and utilize existing latent caches, creating new caches only for the newly added segments.
- The position of a prompt change determines which latent caches can be reused; segments following the modified one are forced to regenerate, and downstream nodes apply the same logic.
- `ignore_latent_hash`: Ignores hash verification for the `input_latent` port. Use case: Some latent processing nodes (such as the Minimax H3 Latent Upscaler (3D) node) alter latent metadata, causing minor changes that render the latent cache unusable and waste inference time; setting this to `true` is recommended in such cases. I have only tested this with the `Minimax_H3-LatentUpscaler_Adv` node from my other repository (github.com/supElement/ComfyUI_Element_easy) and have not tested similar nodes; for latent processing nodes that do not alter latent noise characteristics, the `ignore_latent_hash` parameter can be set to `false`.

V0.5.8

- Refined hash verification parameters to resolve tensor mismatch errors caused by changes in parameters of nodes upstream of the sampler.
- `Minimax_H3_Seam_Correction` node: Removed the shot detection model—which previously caused a "white screen" in the sampling node's preview—and replaced it with the `PySceneDetect` method (CPU-only, avoiding potential contamination).

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
| **Clip_Tag** | Splits prompts based on user-defined tags (e.g., `段1`/`段2`), with each tag corresponding to a separate video segment; segment duration is determined by the prompt content (priority order: duration specified after the tag > in-segment timestamps > `total_frames/fps` fallback). |
| **timeline** | Splits prompts based on explicit timestamps (e.g., `0-2s`/`2-6s`), with each time interval corresponding to a video segment; segment duration = interval length × `fps`, automatically snapped to a valid grid; **ignores `total_frames` and `chunk_frames`**, with total duration determined entirely by the prompts. Global segments (marked `【Global】`) remain in their original positions and are not extracted separately. |
| **sequential** | Distributes prompts evenly across the video timeline based on sentence/phrase order without splitting the prompts themselves; video segmentation still follows `chunk_frames`. |
| **global** | Applies the entire prompt block to all video segments (after stripping the `【Global】` tag); video segmentation follows `chunk_frames`. |

> In `Clip_Tag` and `timeline` modes, the `total_frames` and `chunk_frames` parameters are ignored (segment duration is determined by the prompts) and are only used as a fallback if the mode degrades (e.g., if no tags or timestamps are detected).

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

### Minimax_H3_AutoContext_parameter (Parameter Group Node)

| Parameter | Default | Description |
|-----------|---------|-------------|
| long_prompt | — | Prompt (passed to the main node for inference and used for the "estimated segmentation" preview) |
| **clip_mode** | `Clip_Tag` | Method for mapping prompts to video segments: `Clip_Tag` / `timeline` / `sequential` / `global`. `total_frames` and `chunk_frames` are ignored in `Clip_Tag` and `timeline` modes. |
| clip_tag | `段1` | Clip_Tag segmentation label template (must end with a numeric index); effective only when `clip_mode=Clip_Tag` |
| prompt_format | `official` | Prompt output format: `official` / `legacy` / `raw` | `official` uses the MiniMax H3 official [Shot] format; `legacy` uses the old-style timestamp format; `raw` outputs as-is (for Clip_Tag mode) |
| crop_mode | `stretch` | Scaling/cropping for reference images, start/end frames, or reference videos: `center` / `stretch` / `none` |
| ref_sync_mode | `segmented` | Whether to slice reference video/audio by segment: `global` (use full source for each segment) / `segmented` (slice based on segment time ratios) |
| width × height | 960×544 | Resolution for the first pass (overridden by `latent_input` during the second pass) |
| total_frames | 362 | Total frames to generate (17n+5); acts as a fallback in `Clip_Tag`/`timeline` modes (used only if tags/timestamps are missing), but is ultimately overridden by the sum of individual segments |
| fps | 24 | Frame rate; used for audio synchronization and converting prompt timings to seconds |
| chunk_frames | 90 | Frames generated per segment (17n+5); applies only in `sequential` / `global` modes |
| context_frames | 22 | Continuity frames between segments (17n+5: 5/22/39/56…); recommended value is 22 or higher |
| lock_audio | `true` | Locks the audio region during the second pass (noise_mask audio=0): resamples video only, keeping the first-pass audio unchanged |
| audio_drive | `false` | Audio-driven mode toggle; when enabled, video generation follows `drive_audio` |
| video_guide | `none` | Video extension parameter; supports segmentation. `none`: Disabled(Do not modify the video reference logic); `pre_guide`: Video continuation (uses `ref_video_0` or `ref_video_audio_0` ports); `post_guide`: Video extension/push-forward (uses `ref_video_0` or `ref_video_audio_0` ports); `pre_post_guide`: Bridging between two videos (uses `ref_video_0`/`ref_video_audio_0` and `ref_video_1`/`ref_video_audio_1` ports). The number of anchor frames is determined by `context_frames`. Note: When set to anything other than `none`, the reference at the corresponding port of the sampling node is forcibly truncated to the value specified by `context_frames`. The reference logic remains the same as for standard references (i.e., the reference is utilized only if declared in the prompt). |

> Real-time preview of "estimated segments" displayed on the node (calculated via frontend JS; does not affect the inference process).

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
| enable_cache | true | Stores/reads latent cache; automatically creates a folder named "node+[node ID]" in the "\ComfyUI\output\cache" directory. Existing latent cache files are overwritten if upstream nodes or parameters change. |
| clear_cache | false | Forces the latent cache file to be rebuilt. |
| ignore_latent_hash | false | Ignores hash verification for the `input_latent` port. Useful when certain latent processing nodes alter latent metadata, causing minor changes that render the cache unusable and waste inference time; setting this to `true` is recommended in such cases. |
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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fix_color_preset` | `"medium"` | **Color/Exposure correction level**<br>`off`: disabled;<br>`low`: per‑channel luminance gain with half‑strength correction, most conservative, no color cast;<br>`medium`: per‑channel luminance gain, only corrects seam level jumps (recommended);<br>`high`: MKL linear color transfer with a longer statistical window, more stable with large motion;<br>`max`: frame‑by‑frame luminance normalisation across the whole clip, eliminates gradual drift within segments, but flattens intentional brightness changes (e.g. sunset/tunnel entry); near‑black frames are ineffective (logged). |
| `fix_motion_preset` | `"off"` | **Seam continuity (optical flow alignment + blending) level**<br>`off`: disabled (recommended to test with colour correction first);<br>`low/medium/high/max`: higher levels blend more frames with stronger effect, but may introduce slight blur or pumping. |
| `fix_flash` | `false` | **Flash frame handling** (transient brightness jumps at boundaries). Independent switch, uses temporal fusion logic. May suppress legitimate rapid changes like lightning or explosions. Works even when `fix_motion_preset=off`. |
| `flash_threshold` | `0.30` | Threshold for transient correction (abnormal pixel ratio). Lower values are more aggressive (correct more frames). Recommended range `0.20` – `0.40`. |
| `cut_threshold` | `15.0` | Sensitivity threshold for PySceneDetect (range `5.0` – `50.0`). Lower values are more sensitive. Recommended `10` – `20`. Only effective when `cut_detection=true`. |
| `blend_frames` | `2` | Level transition window around seams (frames, 0–8): after exposure alignment, smooths the brightness transition over `blend_frames` on each side of the boundary. Higher values give smoother transitions but may cause slight blur/breathing with fast motion; `0` disables. |
| `use_gpu` | `true` | Use CUDA GPU for statistics, colour transforms, and optical flow (falls back to CPU if unavailable). |

⚠️ Removed the shot‑detection model dependency; now uses PySceneDetect (pure CPU, no potential pollution).

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
