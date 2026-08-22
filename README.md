<div align="center">

[![中文](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

> 一键式 MiniMax H3 长视频自动化生成节点：**分段推理 + 段间续接锚定 + 提示词时间轴切片 + 二次采样（二采）+ 接缝修正**。

在显存有限的情况下，将长视频拆分为多段独立推理，通过叠加增强方法实现段间无缝衔接，同时按时间轴自动切分提示词，让每段生成内容与提示词节奏对齐。支持二采（低清一采 + 高清二采）。

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## 📖 目录

- [节点列表](#nodes)
- [核心特性](#features)
- [安装](#install)
- [节点参数](#params)
- [输出](#output)
- [二采与 SplitSigmas 高低频](#second-pass)
- [接缝修正节点](#seam)
- [提示词写法示例](#prompt-examples)
- [提示词注意事项（节点的局限性）](#limitations)

## <a id="nodes"></a> 🧩 节点列表

| 节点 | 说明 |
|------|------|
| **Minimax_H3_AutoContext_parameter** | 参数组节点：集中管理提示词/分段/分辨率/音频等参数，输出 `parameter`，并实时预览「预计分段」 |
| **Minimax_H3_AutoContext_Sampler** | 主节点：分段推理 + 续接锚定 + 采样（一采/二采共用） |
| **Minimax_H3_Seam_Correction** | 接缝修正节点：对解码后视频的段间接缝做像素域修正 |

> 用法：`parameter 节点 --parameter--> 主节点`。提示词在 parameter 节点填写，主节点通过 `parameter`（必选）接收。

## <a id="features"></a> ✨ 核心特性

### 🧩 分段推理

- 按 `total_frames` / `chunk_frames`（帧单位）拆分为多段，帧数建议取 5、22、39、56、73、90…
- 末段自动加长补齐，避免出现过短的尾段
- `fps` 仅用于音频同步和提示词内秒数换算

### 🔗 段间续接

- **叠加增强**：非首段自动"接力"上一段的结尾画面，新增内容从上一段结束的位置和动作自然延续，消除接缝处的停顿或位置跳变
- 上一段结尾作为运动参考传给当前段，帮助延续运动方向与速度
- 上一段音频也作为"之前的内容"传入，帮助声音自然延续
- 段间音频平滑淡化，与视频帧数对齐

> 帧数规则：`total_frames` / `chunk_frames` / `context_frames` 都取 5、22、39、56、73、90…（17 的倍数加 5），节点会自动对齐，一般无需手动计算。

### ⏱️ 提示词时间轴

| 模式 | 说明 |
|------|------|
| **auto** | 段落含时间标记（如 `0-5s`）自动切分，无标记段落（风格/音效/禁止项）拼入每个窗口 |
| **timeline** | 强制按时间标记切分 |
| **global** | 整段提示词用于所有窗口（适合全程同质动作的一镜到底） |
| **sequential** | 按句读顺序铺到时间轴，以 `全局:`/`[全局]` 开头的段落拼入每个窗口 |

### 🏷️ Clip_Tag 标签分段模式

- 按用户自定义标签（如 `段1`/`段2`/`段3`）切分提示词，每段 = 一个 chunk = 该段全部提示词
- 段时长由提示词内容决定（三层优先级）：
  1. 标签行紧跟的时长（如 `段1:0-5秒` → 5 秒；`段1:3-8秒` → 5 秒）
  2. 段内时间标记的最大结束值（如 `【0-2秒】`+`【2-5秒】` → 5 秒）
  3. `total_frames / fps` 默认值兜底（单段时总帧数贴合 `total_frames`）
- 段内时间标记是**相对时间**（每段从 0 开始），不是全局绝对时间
- 非首段会多生成一段重叠帧用于衔接，生成后自动裁掉
- 总时长自动对齐目标总帧数，尽量贴近预期时长
- 推理时标签本身会被去除，其余提示词内容根据 `prompt_format` 输出

### 🎯 参考智能过滤（图 / 视频 / 音频）

- 自动识别每段提示词里用到的参考图/视频/音频，只把被引用到的素材传给该段
- 视频与其配对音轨绑定，避免画面/声音串扰

### 🖌️ 二次采样（二采）

- 主节点 `latent_input` 接入一采 latent（或经 latent 放大节点）即进入二采模式
- 二采分辨率**以输入 latent 为准**（忽略 width/height），实现低清一采 → 高清二采
- `denoise` 控制重绘强度；`sigmas` 支持自定义 sigma 序列（与 `SamplerCustomAdvanced` 同款）
- `lock_audio`：二采只重画视频、复用一采音频

### 🎵 音频驱动（Audio Drive）

- `drive_audio`（AUDIO，可选）+ `audio_drive` 开关
- 开启后视频跟随这条音频生成，输出音频 = 源音频本身（口型/节奏由它驱动）

## <a id="install"></a> 📦 安装

### 方式一：手动安装（Manual Installation）

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### 方式二：通过 Manager 安装（Install using Manager）

在 ComfyUI Manager 中搜索 `ComfyUI_MinimaxH3_AutoContext`，点击 Install。


## <a id="params"></a> ⚙️ 节点参数

### Minimax_H3_AutoContext_parameter（参数组节点）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| long_prompt | — | 提示词（传给主节点推理，同时用于「预计分段」预览） |
| prompt_mode | auto | 提示词时间轴模式（仅 Clip_Frame 生效） |
| clip_mode | Clip_Frame | 分段模式：Clip_Frame / Clip_Tag |
| clip_tag | 段1 | Clip_Tag 分割标签模板（必须以数字序号结尾） |
| prompt_format | official | 提示词输出格式：official / legacy / raw |
| crop_mode | stretch | 参考图/首尾帧/参考视频缩放裁剪：center / stretch / none |
| ref_sync_mode | segmented | 参考视频/音频是否按段切片：global / segmented |
| width × height | 960×544 | 一采分辨率（二采时被 latent_input 覆盖） |
| total_frames | 362 | 生成总帧数（17n+5）；Clip_Tag 下无明确时长的段以 total_frames 兜底，最终被各段之和覆盖 |
| fps | 24 | 帧率，用于音频同步和提示词秒数换算 |
| chunk_frames | 90 | 每段生成帧数（17n+5） |
| context_frames | 22 | 段间续接帧数（17n+5：5/22/39/56…） |
| lock_audio | enable | 锁定音频区，只重画 video |
| audio_drive | disable | 音频驱动开关 |


> 节点上实时显示「预计分段」预览（前端 JS 计算，不参与推理）。

### Minimax_H3_AutoContext_Sampler（主节点）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 模型组件 |
| parameter | 必选 | 参数组输入（来自 parameter 节点） |
| sampler | 可选 | 外部采样器对象（SAMPLER），覆盖内置 sampler_name/scheduler |
| sigmas | 可选 | 自定义 sigma 序列（SIGMAS），优先级最高 |
| latent_input | 可选 | 二采输入 latent（接入即开启二采） |
| info | 可选 | 参数继承输入（多采串联，保证分段一致） |
| first_frame / last_frame | 可选 | 首/尾帧锚定（FL2VA） |
| video_context_denoise | 0.0 | 段间续接强度（仅非首段）：0=精确延续上一段结尾，1=重新生成，中间值=软混合。二采接 SplitSigmas 时建议设 1 避免花屏 |
| seed | 0 | 随机种子（control_after_generate） |
| steps / cfg | 30 / 1.0 | 采样步数 / CFG |
| sampler_name / scheduler | euler / simple | 内置采样器 / 调度器 |
| denoise | 1.0 | 重绘强度（1=全量重采样，越小保留越多原结构） |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | 可选 | 参考素材（Autogrow 动态端口） |
| drive_audio | 可选 | 音频驱动源 |

## <a id="output"></a> 📤 输出

| 输出 | 说明 |
|------|------|
| **latent** | 拼接后的音视频 latent，接 VAE Decode，或放大后接二采 |
| **denoised_latent** | 干净的 latent 输出，用于二采接力 / 预览 |
| **info** | 分段参数（Dict），传给下一个主节点的 info 输入，保证多采分段一致 |



## <a id="second-pass"></a> 🔄 二采与 SplitSigmas 高低频

### 基础二采（低清一采 → 高清二采）

```
parameter 节点 ──parameter──> 主节点(一采, 864×480)
    └─ latent / denoised_latent ──> [分离 AV] ──> video_latent ──> latent 放大 ──> [合并 AV] ──> 主节点(二采).latent_input
二采节点: parameter 共用 (或 info 继承)，可选 denoise 0.4~0.6
```

- 二采分辨率以 `latent_input` 为准，忽略 parameter 的 width/height

### SplitSigmas 高低频（省时间提清晰度）

> ⚠️ **音频约束**：高低频**只对视频生效**（音频段间衔接需要完整采样），音频请保持完整采样。

```
一采节点: 完整采样 (不接 high_sigmas，audio 完整去噪)
          → denoised_latent → 分离放大 video (audio 不动) → 合并 → 二采.latent_input
二采节点: sigmas ← low_sigmas (只跑低 sigma 段提细节)
          lock_audio = True (复用一采完整音频)
          video_context_denoise = 1.0 (续接区随新增区一起重绘，避免花屏)
```

> 💡 **二采 `video_context_denoise`**：接 SplitSigmas 时若设 0（精确延续），续接区与已重绘的新增区在边界可能花屏；设 1.0 让续接区同步重绘即可避免。接缝略有不连续时可降到 0.3~0.5 折中。一采保持默认 0。

## <a id="seam"></a> 🧵 接缝修正节点（Minimax_H3_Seam_Correction）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| fix_color_preset | medium | 色彩/曝光处理档位。 off：不处理；low：逐通道亮度增益，修正量减半，最保守，无色偏；medium：逐通道亮度增益，仅修接缝电平跳变（推荐）；high：MKL 线性色彩迁移，统计窗口更长，运动大时更稳；max：全片逐帧亮度归一化，消除段内渐变漂移，但会削平画面本身明暗变化（如天黑/进隧道），近黑帧无效（日志报告）。 |
| fix_motion_preset | off | 接缝连续性档位：off=不处理（推荐默认，先只用色彩档位看效果）；low/medium/high/max=光流对齐+局部融合，档位越高参与融合的帧数与强度越大，但越容易带来轻微糊感或 pumping |
| fix_flash | false | 闪帧处理（边界处瞬时亮度突跳）。独立开关，采用接缝时域融合逻辑。若画面有闪电、爆炸等合理快速明暗变化，抑制会削平这些效果。fix_motion_preset=off 时仍生效。|
| flash_threshold | 0.30 | 瞬态修正筛选阈值（异常像素占比）,值越小越激进（修正更多帧）,推荐 0.20~0.40。|
| blend_frames | 2 | 接缝处电平渐变窗口（帧，0~8）：曝光对齐后，把边界前后各 blend_frames 帧的亮度过渡按平滑斜坡拉平；值越大过渡越缓、越自然，但运动大的镜头过大会带来轻微糊感/呼吸感；0=关闭 |
| cut_detection | true | 镜头检测闸门：开启后对整个视频逐帧做切镜检测（相邻帧 RGB 直方图相关度），曝光/色彩/运动修正只在同一镜头内部进行，真实切镜处（含分段内部切镜）直接跳过；关闭则回到旧行为，处理全部分段边界 |
| use_gpu | true | 使用 CUDA GPU 进行统计、色彩变换与光流 |
| debug | false | 打印诊断 |

⚠️模型（检测分镜）：将models/transnetv2文件夹中的模型文件，拷贝到comfyUI的模型目录中（".\ComfyUI\models\transnetv2\transnetv2.pth"）

> 用法：`VAE Decode → H3_Seam_Correction → Save/Video`。

> ⚠️ 提示：本节点只做画面接缝修正，无法修复二采上游产生的伪影。

## <a id="prompt-examples"></a> ✍️ 提示词写法示例

### 时间轴模式（auto / timeline）

```text
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape
```

> 含 `0-5s` 标记的段落按时间切分，无标记段落（风格/音效/禁止项）自动拼入每个窗口。

### 全局模式（global）

> 整段提示词用于所有分段，适合全程同质动作的一镜到底。

### Clip_Tag 模式（按标签分段）

> `clip_mode` 设为 `Clip_Tag`，`clip_tag` 填写标签模板（必须以数字序号结尾）。

**标签模板示例**

| 模板 | 匹配 |
|------|------|
| `段1` | `段1` / `段2` / `段3`（前缀"段"+数字） |
| `A01` | `A01` / `A02` / `A03`（前缀"A"+数字） |
| `[片段001]` | `[片段001]` / `[片段002]`（前缀"[片段"+数字+后缀"]"） |

**标签写法**：标签独占一行作为分割点，标签后推荐换行。不换行时也能处理（跳过分隔符取段内容）：

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

**段时长规则**（三层优先级）：

1. 标签行紧跟的时长：`段1:0-5秒` → 5 秒；`段1:3-8秒` → 5 秒（时长标记会从提示词中去除）
2. 段内时间标记 0 基：`【0-2秒】`+`【2-5秒】` → 5 秒
3. 都没有 → `chunk_frames / fps` 兜底

**prompt_format 选择**

- `official` / `legacy`：段内时间标记自动转为段内相对坐标渲染
- `raw`：去标签后原样输出，时间标记保持不变（适合用大模型生成的结构化提示词）

## <a id="limitations"></a> 📝 提示词注意事项（节点的局限性）

> 以下注意事项**不适用于**简单的、始终有效的提示词场景（即所有分段共用同一个提示词，global 模式），
> 比如：口播数字人（当然，台词需要分段）、视频中的镜头/构图变化不大，或视频替换角色等提示词通用的场景。

### 1️⃣ 核心原则：时序排他性

> 使用分段推理（Chunk）时，请务必遵守**时序排他性**原则——每一段提示词只能描述该段"正在发生"的、相对于上一段末尾的**新变化**。

- **分段即"接力"**：第 N 段生成时，它的起始画面状态（位置、动作姿态、镜头位置）完全由上一段末尾的"锚定帧（Context Frames）"隐式提供。你不需要在提示词里重复描述这个起始状态。
- **禁止"回叙"与"重叠"**：第 N 段的提示词绝对不可以重复描述第 N-1 段已经完成的动作或镜头运动。如果重复描述，模型接收到的指令就会与锚定帧的画面产生逻辑冲突（指令冲突），导致生成画面卡顿、运动逻辑错乱或动作重复。
- **边界清零**：切换分段时，请把上一段的"正在进行的动作"清零。新的一段提示词，应当像"按下快门后的新指令"一样，仅针对当前新时间段内发生的位移、动作或新元素出现。

**❌ 错误写法（冲突重叠）**

```text
段1：3秒
"物体 A 向位置 B 移动"
段2：3-6秒
"物体 A 移动到位置 B 后，正在位置 B 转身"
```

> 问题分析：第 1 段结束时，锚定帧显示物体 A 已经到达位置 B 且刚停稳。但第 2 段提示词强行要求"物体 A 移动到位置 B"，这与锚定帧"已到达"的静态结果冲突，模型会试图"重新移动"，造成鬼畜或跳帧。

**✅ 正确写法（无缝递进）**

```text
段1：3秒
"物体 A 向位置 B 移动，并最终停在位置 B"（强调动作闭环）
段2：3-6秒
"站稳后，物体 A 缓慢转动方向"（直接描述上一段结束后的新动作）
```

> 正确逻辑：第 2 段完全抛弃对"移动过程"的描述，默认"停在 B 点"是既定事实，只描述接下来的"转动"新动作，模型就能利用锚定帧完美续接。

> 🚀 **总结成一句话**：上一段的末尾是"结果"，下一段的开头是"结果之后的新动作"，别把"导致结果的过程"写到下一段里去。

### 2️⃣ 核心原则：逐段引用声明

> 使用分段推理并搭配参考图/视频（image1、video1 等）时，请务必遵守**逐段引用声明**原则——每一段提示词都必须独立且完整地声明该段所需要的全部引用素材，引用不会被"记忆"或"继承"到下一段。

- **无全局记忆**：节点解析当前段提示词内写明的引用标签，精准判断该段需要哪些素材。上一段写了 image1，只代表上一段调用了它；到了下一段，会重新扫描。
- **不写则不传**：如果第 N 段没再写 image1，该段就不会传入这张参考图，导致角色/物体不一致。

**❌ 错误写法（隐式继承）**

```text
段1[3秒]：image1 是物体A，物体A正在向前移动。
段2[3-6秒]：物体A停下，转身看向镜头。（没写 image1）
```

**✅ 正确写法（逐段显式）**

```text
段1[3秒]：image1 是物体A，物体A正在向前移动。
段2[3-6]：image1 是物体A，物体A停下，转身看向镜头。
```
