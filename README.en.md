<div align="center">

[![中文](https://img.shields.io/badge/Language-Simplified%20Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 Long Video Automation Generation Node: **Segmented Inference + Inter-segment Anchor Alignment + Prompt Timeline Slicing + Secondary Sampling (SS) + Seam Correction**.
Under limited GPU memory, long videos are divided into multiple segments for independent inference, and seamless splicing is achieved through superposition enhancement. At the same time, prompts are automatically sliced along the time axis to align the generated content with the prompt rhythm; the same slicing and alignment are performed on the audio-visual references, and only the references referenced in the current segment participate in the inference. It supports secondary sampling. Supports video extension, video push forward, and double video splicing.
Supports latent caching and retrieval, which is convenient for quickly skipping the already inferred segments and caching files when the inference is interrupted for some reason, storing the cache files in segments, and reading the existing latent cache files without changing the upstream parameters of the sampling node.

Note: When changing the model, including lora, sageattention, and other acceleration nodes, the latent detection will not detect the change, so it is necessary to delete the latent cache. Two methods to delete the latent cache:
- Enable the clear_cache parameter on the Minimax_H3_AutoContext_Sampler node, which will force the node to rebuild the cache file at the start of sampling.
- Manually delete the corresponding folder in the cache directory (\ComfyUI\output\cache), which is named "node_" + "node ID".

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## Bug Fixes and Optimizations

V0.6.5

- Optimized the latent cache processing logic, removed the manual specification of the cache directory, and changed it to automatically specify a unique cache directory for each node ("node + node ID"), to prevent misoperations that cause the latent cache logic of sampling nodes to overlap.
- Built cache and verification logic in segmented mode, if the upstream node only adds prompts or increases the number of segments without changing the other segments of prompts submitted to the sampling, and the other parameters associated with the sampling node do not change, the existing corresponding cache is still considered valid and called, and the new added segments will also automatically build latent cache. The downstream sampling node (SS) will also retain the existing latent cache and call it, and only create the cache for the added segments.
- The change of prompt position determines which latent caches can be reused, and the segments after the changed prompt must be rebuilt. The downstream nodes use the same processing logic.
- ignore_latent_hash, ignore the hash value check of the input port input_latent. Practical scenarios: some latent processing nodes will change the latent judgment information (such as: Minimax H3 Latent Upscaler (3D) node), causing the latent to be slightly changed and causing the latent cache to be unavailable, wasting inference time. At this time, it is recommended to set it to true. I only tested the Minimax_H3-LatentUpscaler_Adv node in my another repository github.com/supElement/ComfyUI_Element_easy extension, and similar nodes have not been tested. When using latent processing nodes that do not change the latent noise features, you can set the ignore_latent_hash parameter to false.

V0.5.8
- Improved the hash value detection parameter, solving the error of tensor mismatch caused by the change of parameter of the upstream node of the sampler.
- Minimax_H3_Seam_Correction node, remove the lens detection model, the detection model will cause the preview of the sampling node to "go blank", replace it with PySceneDetect method (pure CPU, no potential pollution).

## 📖 Directory

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Second Pass and SplitSigmas High and Low Frequencies](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Considerations (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: centrally manages prompts/segments/resolution/audio, outputs `parameter`, and previews the "expected segmentation" in real time |
| **Minimax_H3_AutoContext_Sampler** | Main node: segmented inference + continuation anchoring + sampling (common for first and second passes) |
| **Minimax_H3_Seam_Correction** | Seam correction node: pixel domain correction for the seam of the decoded video |

> Usage: `parameter node --parameter--> main node`. Prompts are filled in the parameter node, and the main node receives `parameter` (required) through `parameter`.

## <a id="features"></a> ✨ Core Features

### 🧩 Segmented Inference

- Split into multiple segments according to `total_frames` / `chunk_frames` (frame unit), frame number recommendations: 5, 22, 39, 56, 73, 90...
- The last segment is automatically extended to fill, avoiding overly short tail segments
- `fps` is only used for audio synchronization and prompt second calculation

### 🔗 Segment Continuation

- **Superposition Enhancement**: Non-first segment automatically "re接力" the end of the previous segment, new content is naturally continued from the end of the previous segment, eliminating pauses or position jumps at the junction
- The end of the previous segment is passed as a motion reference to the current segment to help continue the direction and speed of motion
- The audio of the previous segment is also passed as "previous content" to help the sound continue naturally
- Segment audio is smoothly faded, aligned with video frame number

> Frame number rules: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90... (multiple of 17 plus 5), the node will automatically align, generally no need to calculate manually.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **Clip_Tag** | Split prompts according to user-defined tags (such as `Segment 1`/`Segment 2`), each tag corresponds to an independent video segment; segment duration is determined by the prompt content (tag time > segment time tag > `total_frames/fps` as a floor). |
| **timeline** | Split prompts according to explicit time marks (such as `0-2s`/`2-6s`), each time interval corresponds to a video segment; segment duration = interval length × `fps` and automatically adsorbs to a legal grid; **ignore `total_frames` and `chunk_frames`**. The global segment (`【Global】`) remains in the original position and will not be concentrated extracted. |
| **sequential** | Distribute prompts uniformly to the entire video time axis according to the order of reading, do not split the prompt itself; video segmentation still follows `chunk_frames`. |
| **global** | The entire prompt is used for all video segments (after stripping the `【Global】` tag), and video segmentation is carried out according to `chunk_frames`. |

> In `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` parameters are ignored (segment length is determined by the prompt), and these values are only used when the mode degrades (such as no tags or time marks detected).

### 🏷️ Clip_Tag Tag Segmentation Mode

- Split prompts according to user-defined tags (such as `Segment 1`/`Segment 2`/`Segment 3`), each segment = one chunk = all prompts of this segment
- Segment duration is determined by the prompt content (three levels of priority):
  1. Time duration immediately following the tag line (such as `Segment 1:0-5 seconds` → 5 seconds; `Segment 1:3-8 seconds` → 5 seconds)
  2. The maximum end value of the time mark within the segment (such as `【0-2 seconds】`+`【2-5 seconds】` → 5 seconds)
  3. `total_frames / fps` default value as a floor (total frame number of a single segment conforms to `total_frames`)
- Segment time marks are relative time (each segment starts from 0), not global absolute time
- A segment of overlapping frames is generated to be used for splicing, and