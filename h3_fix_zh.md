# MinimaxH3 修脸流水线说明 (Face_Cut → Face_Resample → Face_Blend)

三个节点串联, 对主采样节点 (Minimax_H3_AutoContext_Sampler) 的输出做面部修复:

| 节点 | 步骤 | 职责 |
|---|---|---|
| Minimax_H3_Face_Cut | ① 检测与裁剪 | 分镜 + YOLO 检测 + 可选 SeC-4B 追踪, 逐帧平滑窗口裁成统一 res² 小图 |
| Minimax_H3_Face_Resample | ② 精修 | 主采样同款模型做块级 img2img 重采样 (块结构镜像主采样分段 + 块间锚定) |
| Minimax_H3_Face_Blend | ③ 贴回 | 精修脸按几何账本与 mask 逐像素贴回原画面 (零 VAE) |

## 一、标准接线

| 从 | 到 |
|---|---|
| 主采样 info | Face_Cut.info 与 Face_Resample.info |
| 主采样输出画面 | Face_Cut.images (推荐 images 模式) 与 Face_Blend.images |
| Face_Cut.crop_images | Face_Resample.crop_images |
| Face_Cut.face_pack | Face_Resample.face_pack |
| Face_Cut.shot_info | parameter 节点 (分配镜头提示词) |
| Face_Resample.images | Face_Blend.canvas |
| Face_Resample.bbox | Face_Blend.bbox |
| Face_Cut.masks | Face_Blend.masks (**直连**, 不经过 Resample) |
| Face_Blend.images | 最终输出画面 |


## 二、Face_Cut — 检测与裁剪

流程: PySceneDetect 分镜 → YOLO 检测 → 可选 SeC-4B 身份追踪 → 逐帧平滑窗口裁剪。
模型下载地址：[huggingface](https://huggingface.co/cglearned/Minimax_H3_Face_Cut/tree/main)

### 模型放置目录

| 模型 | 对应参数 | 放置目录 | 说明 |
|---|---|---|---|
| YOLO 人脸检测 | face_model | ComfyUI/models/elementEasy/ | .pt/.pth/.onnx/.engine/.torchscript; 不选直接报错 |
| SeC-4B 身份追踪 | sec_model | ComfyUI/models/sams/ | 建议 fp16; 选 None = 单脸模式 |
| 放大模型 | upscale_model | ComfyUI/models/upscale_models/ | 仅图像超分 (ESRGAN/RealESRGAN/UltraSharp 类); 避开 GFPGAN/CodeFormer 人脸修复类 |

模型放入目录后需**刷新或重启 ComfyUI**, 下拉框才会出现新文件。
PySceneDetect 是 Python 依赖而非模型: 缺失时 pip install scenedetect, 自动降级为仅上游分段隔离 (只警告不报错)。

### 关键参数

| 参数 | 说明 |
|---|---|
| images / latent | 二选一。images 模式推荐: 外部画面原生尺寸检测, 完全不碰 latent; latent 模式用 VAE 解码探针帧 |
| yolo_threshold | 检测置信度 (默认 0.3), 漏检多就调低 |
| shot_threshold | 分镜阈值, 越高切点越少 (默认 40) |
| upscale_model | 可选 SR 链: 反复放大至 ≥ 画布边长 (≤3 次) 再 lanczos 到精确尺寸, 修小脸高倍放大的振铃色斑; None = 仅 lanczos |
| res / expand | 画布边长 (默认 512) / 裁剪窗口余量 % (默认 20) |
| skip_ratio | 脸 ≥ res×此比例 → 跳过重采样 (默认 0.8) |
| sec_model / sec_threshold / max_identities | None = 单脸 (每帧最大脸, 无 mask); 选权重 = 多身份追踪 + SeC mask |


## 三、Face_Resample — 画布精修

裁剪行当画布, 按块 img2img 精修; 块结构镜像主采样分段, 块间续接锚定。

**两种接线模式**: 集成模式 (info 接主采样; model/vae/clip 来自 info.h3_runtime, 本地端口忽略) / 独立模式 (info 留空, 本地 model/vae/clip 必填, 提示词与分段来自 parameter)。parameter 端口两种模式都必接。

| 参数 | 说明 |
|---|---|
| sigmas | 精修 σ 阶梯 (调度器输出); 步数越少修脸越保守 |
| seed | 块采样种子 (每块自动偏移) |
| color_match | 逐子轨 Reinhard 色彩匹配回源裁剪 (默认开) |
| ref_images | 参考图; 提示词声明 Picture 标签才传递 |


块级原理: 同身份且时间连续的子轨合并为一条序列 → 按主采样真实段边界切块 (提示词按块中点精确映射到段) → 每块编码补齐 17n+5 网格 (重复末帧, 解码后裁掉) → 边界窗口大跳变时按追踪模式决定锚定或禁用。

## 四、Face_Blend — 贴回合成

逐子轨缩放 + 像素贴回, **零 VAE** (全程像素域, 无编解码质量损失)。

| 端口 / 参数 | 说明 |
|---|---|
| images | 原视频画面 |
| canvas | Face_Resample 输出的精修画布 |
| bbox | Face_Resample 的 face_pack。⚠ 不能接 shot_info (镜头表不是几何包); 接 Face_Cut 的 face_pack 也可以 (自动用 crop_off 兜底并警告, 但推荐接 Resample) |
| masks (可选) | Face_Cut 的 SeC mask 直连, 行 1:1 对齐 |
| use_sec_mask | **默认 False** (羽化框整窗贴回)。True 时 mask 来源优先级: masks 端口 > pack 内置 > 回退羽化框并警告 |
| feather_px | 羽化像素 (默认 16, 0–128): 框模式羽化矩形边; mask 模式 = 腐蚀 feather/2 + 高斯模糊 σfeather/2 (小脸被腐蚀掏空时回退原 mask) |


## 五、常见问题

| 现象 | 处理 |
|---|---|
| 改参数/代码后结果没变 | 缓存命中 — 勾 clear_cache 跑一次; 改代码后务必清缓存 |
| Resample 报缺输入 | 集成模式没接 info / 独立模式没接 model+vae+clip |
| Blend 报 bbox 端口错误 | 接成了 shot_info — 必须接 face_pack |
| 想只贴人脸、背景不动 | Blend 的 use_sec_mask=True + Face_Cut 的 masks 直连 (默认羽化框会替换整个窗口内容) |
| 贴回边缘生硬 | 加大 feather_px; 单脸模式无 mask, 羽化框回退属正常 |
| UNCOVERED 警告 | 该区间整段无脸, 原样透传; 意外出现时查 YOLO 阈值 |
| masks 全零 | 单脸模式正常; multi_sec 模式全零查上方 SeC 日志 |
| 小脸修完有色斑 | Face_Cut 选 RealESRGAN_x4plus 类 SR; 若仍有则源于多次 VAE 往返, 降步数观察 |
| SR 加载失败 | 自动回退 lanczos; 仅支持图像超分模型 (RIFE 等视频类不支持) |
| 看逐子轨详细日志 | h3_facefix.py 顶部 _VERBOSE = True |
| 缓存位置 | output/cache/node_<节点id>/, 可整目录手动删除 |

