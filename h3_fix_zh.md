# MinimaxH3 修脸流水线说明 (Face_Cut → Face_Resample → Face_Blend)

### 保持面部特征

**方法一: 参考图 (推荐 — 精修的同时保身份)**

通过 ref_images 端口传入参考图, 并在提示词中用 Picture 标签声明引用:

- 标签必须写在**该角色出现的分段**的提示词里 — Resample 的提示词按分段生效, 写在哪个段, 参考图就只传给映射到该段的块;
- 多角色视频必须在提示词中描述参考图对应画面中的哪一位角色 (供模型消歧); 单人视频可省略;
- 技巧: 参考图可直接取源视频中该角色最清晰的一帧人脸 (从 Face_Cut 的 crop 输出中挑选), 身份一致性最稳。

**方法二: 降低重采样强度 (零额外配置, 以修复效果换身份保留)**

降噪强度由 σ 阶梯的**起始值**决定 (img2img 加噪量 = 起始 σ × 噪声), 降低起始 σ 即可让精修更保守:

- 用 SplitSigmas 节点: low_sigmas 端口 → Face_Resample 的 sigmas 端口 (low 侧自带末值 0, 天然满足端口要求), 并**增大 SplitSigmas 的 step**;
- step 必须**小于**总采样步数, 且给 low 侧至少留几步 — step 逼近总步数时 low_sigmas 趋于只剩末值 0, 精修完全退化 (极端时接近原样透传);
- 新版 ComfyUI 可改用 SplitSigmasDenoise 节点, 直接按 denoise 值切分, 免去 step 与 σ 的换算。

**两者关系**: 互补而非二选一。降低起始 σ 保留的是**原始裁剪**的身份 (瑕疵与色斑也一并留下); 参考图保留的是**参考图**的身份 (照常精修, 同时把脸往参考拉)。推荐 ref_images 打底 + 适度降低起始 σ 兜底。

> 注: σ 阶梯变化会自动使 Resample 缓存失效, 无需手动 clear_cache。


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

流程: PySceneDetect 分镜 → YOLO 检测 → 可选 SeC-4B 身份追踪 → 逐帧平滑窗口裁剪。多人场景: 身份仲裁按逐像素 mask 比对 (box 仅作回退), 追踪缺口内的检出会先归还给既有轨迹、再考虑新建身份 — 交叉换位/短暂互相遮挡不再把同一身份切成数块; 残留小缺口由轨迹内插值桥接 (默认上限 24 帧)。
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
| upscale_model | 可选 SR 链: 反复放大至 ≥ 画布边长 (≤2 次) 再 lanczos 到精确尺寸, 修小脸高倍放大的振铃色斑; None = 仅 lanczos |
| res / expand | 画布边长 (默认 512) / 裁剪窗口余量 % (默认 20) |
| skip_ratio | 脸 ≥ res×此比例 → 跳过重采样 (默认 0.8) |
| sec_model / sec_threshold / max_identities | None = 单脸 (每帧最大脸, 无 mask); 选权重 = 多身份追踪 + SeC mask; SeC-4B 追踪结束后始终自动卸载 |
| sr_batch | 裁剪放大阶段每次 SR 前向的帧数 (默认 4, 1–16)。越大越快但显存峰值越高; 16GB 用 4, 24GB+ 可 8~16。只影响速度与显存, 不影响结果 |
| unload_main_models | 检测与 SeC 追踪完成后、裁剪/SR 放大前, 把 H3 主模型/VAE/CLIP 移出显存 (默认开; 12~16GB 显存建议开启)。SR 模型经 spandrel 直进显存、不受 ComfyUI 模型管理调度, 主模型驻留会把显存挤过物理上限; Windows 会以"共享 GPU 内存"溢出掩盖 OOM, 表现为速度骤降。无论此项开关, SeC-4B 追踪结束后始终卸载 |


## 三、Face_Resample — 画布精修

裁剪行当画布, 按块 img2img 精修; 块结构镜像主采样分段, 块间续接锚定。

**两种接线模式**: 集成模式 (info 接主采样; model/vae/clip 来自 info.h3_runtime, 本地端口忽略) / 独立模式 (info 留空, 本地 model/vae/clip 必填, 提示词与分段来自 parameter)。parameter 端口两种模式都必接。

| 参数 | 说明 |
|---|---|
| sigmas | 精修 σ 阶梯 (调度器输出); 步数越少修脸越保守 |
| seed | 块采样种子 (每块自动偏移); 控件 control_after_generate 保持 fixed, 否则行缓存永不命中 |
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
| 两人交叉换位时同一身份被切碎 / 凭空多出短命身份 | v20.4 起自动修复: mask 仲裁 + 缺口归属 + 轨迹内插值桥接。SeC 日志出现 “N 帧归属现有轨迹” 即修复生效。仍被切碎时调低 sec_threshold |
| 16GB 卡裁剪/SR 阶段速度骤降但无 OOM 报错 | Windows 把显存溢出静默转入"共享 GPU 内存"而不报错。保持 unload_main_models 开启, 并把 sr_batch 降到 2 |
| Face_Cut 勾了 clear_cache 但 Face_Resample 仍用旧结果 | clear_cache 只清该节点自己的缓存。Face_Resample 的裁剪行缓存在裁剪像素变化 (内容哈希) 或 σ 阶梯变化时自动失效; 节点id变化后需删除各节点缓存目录 output/cache/node_<节点id>/ |
| 升级节点后旧工作流端口标红/被重置 | v20.3 起移除 sec_auto_unload (SeC 现在始终自动卸载), 新增 unload_main_models 与 sr_batch, 需重新接线勾选 |
