"""
seam_correction.py - H3 视频接缝修正节点 (像素域)

参考 ComfyUI-H3-Continuum 的 v3/video_seam.py 的接缝分析 + 修正思路：
- 自动检测段间接缝 (帧间亮度跳变)
- 分类: 场景切换 / 瞬态闪光 / 曝光渐变 / 运动卡顿
- 对曝光/闪光类跳变做 lerp 修正，保留真正的场景切换
"""

import torch
import torch.nn.functional as F
from comfy_api.latest import io

_LONG_SIDE = 192
_LUMA_W = (0.2126, 0.7152, 0.0722)


def _downsample_rgb(frames, long_side=_LONG_SIDE):
    """降采样到长边 long_side 以内，加速分析。输入 [T,H,W,C] in 0-1。"""
    rgb = frames[..., :3].clamp(0.0, 1.0).permute(0, 3, 1, 2)
    h, w = rgb.shape[-2], rgb.shape[-1]
    scale = min(1.0, long_side / float(max(h, w)))
    if scale < 1.0:
        rgb = F.interpolate(rgb, scale_factor=scale, mode="bilinear", align_corners=False)
    return rgb.permute(0, 2, 3, 1)


def _luma(frames):
    w = frames.new_tensor(_LUMA_W)
    return torch.sum(frames[..., :3] * w, dim=-1)


def _frame_delta(a, b):
    return float(torch.mean(torch.abs(a - b)).item())


def _detect_seams(images, threshold):
    """检测候选接缝位置：帧间 luma 差异显著跳变 (且为局部峰值)。"""
    if images.shape[0] < 2:
        return []
    small = _downsample_rgb(images)
    y = _luma(small)
    deltas = [float(torch.mean(torch.abs(y[i] - y[i - 1])).item())
              for i in range(1, y.shape[0])]
    seams = []
    for i, d in enumerate(deltas):
        if d < threshold:
            continue
        # 局部峰值: 显著高于邻域
        left = deltas[i - 1] if i - 1 >= 0 else 0.0
        right = deltas[i + 1] if i + 1 < len(deltas) else 0.0
        if d >= left * 1.3 and d >= right * 1.3:
            seams.append(i + 1)  # 边界帧下标 (后段首帧)
    return seams


def _classify_boundary(images, boundary, threshold):
    """对边界分类: 'scene_cut' / 'flash' / 'exposure' / 'motion' / 'clean'。

    返回 (classification, delta)。
    """
    small = _downsample_rgb(images)
    y = _luma(small)
    if boundary < 1 or boundary >= small.shape[0]:
        return "clean", 0.0

    prev = y[boundary - 1]
    curr = y[boundary]
    delta = float(torch.mean(torch.abs(prev - curr)).item())

    # 场景切换: 跳变巨大且后段内部稳定 (持续变化)
    if boundary + 2 < small.shape[0]:
        cur_internal = 0.5 * (
            float(torch.mean(torch.abs(y[boundary] - y[boundary + 1])).item())
            + float(torch.mean(torch.abs(y[boundary + 1] - y[boundary + 2])).item())
        )
    else:
        cur_internal = delta
    if delta >= 0.12 and cur_internal <= delta * 0.35 and delta >= threshold * 1.5:
        return "scene_cut", delta

    # 曝光渐变: 跳变明显，后段亮度持续漂移 (非快速恢复)
    if boundary + 3 < small.shape[0]:
        after = [float(torch.mean(torch.abs(prev - y[boundary + k])).item())
                 for k in range(1, 4)]
        if all(a > delta * 0.9 for a in after):
            return "exposure", delta

    # 瞬态闪光: 跳变后快速恢复到接近前段
    if boundary + 2 < small.shape[0]:
        d2 = float(torch.mean(torch.abs(prev - y[boundary + 1])).item())
        if d2 < delta * 0.6:
            return "flash", delta

    # 运动卡顿: 跳变中等，且前后帧运动幅度不匹配 (简化: 中等 delta)
    if delta >= threshold and delta < 0.12:
        return "motion", delta

    return "clean", delta


def _correct_frame_lerp(images, boundary, frames=1):
    """对边界帧做 lerp 过渡，前段尾帧和后段后续帧之间插值。"""
    out = images.clone()
    if boundary < 1 or boundary >= out.shape[0]:
        return out
    prev = out[boundary - 1]
    recovery = boundary + frames
    if recovery >= out.shape[0]:
        return out
    target = out[recovery]
    for offset in range(frames):
        idx = boundary + offset
        if idx >= out.shape[0]:
            break
        alpha = float(offset + 1) / float(frames + 1)
        blend = torch.lerp(prev, target, alpha)
        orig = out[idx]
        blend_mean = blend.mean(dim=(0, 1), keepdim=True)
        orig_mean = orig.mean(dim=(0, 1), keepdim=True)
        out[idx] = (orig + blend_mean - orig_mean).clamp(0.0, 1.0)
    return out


class H3SeamCorrection(io.ComfyNode):
    """H3 视频段间接缝修正：自动检测跳变并分类修正。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3SeamCorrection",
            display_name="Minimax_H3_Seam_Correction",
            category="MinimaxH3_AutoContext",
            description="H3 视频段间接缝像素域修正：自动检测帧间亮度跳变，分类 (场景切换/瞬态闪光/曝光渐变/运动卡顿) 并做 lerp 修正",
            inputs=[
                io.Image.Input("images",
                    tooltip="解码后的视频帧 (拼接后的完整视频)"),
                io.Boolean.Input("fix_flash", default=True,
                    tooltip="修正瞬态闪光/曝光跳变 (边界帧 lerp 过渡)"),
                io.Boolean.Input("fix_exposure", default=True,
                    tooltip="修正曝光渐变 (边界帧 lerp 过渡)"),
                io.Boolean.Input("fix_motion", default=False,
                    tooltip="修正运动卡顿 (边界帧 lerp 过渡，可能影响正常运动，谨慎开启)"),
                io.Boolean.Input("fix_scene_cut", default=False,
                    tooltip="修正场景切换 (默认关闭：场景切换是内容变化，不应被抹平)"),
                io.Float.Input("threshold", default=0.05, min=0.0, max=1.0, step=0.01,
                    tooltip="接缝检测阈值 (帧间亮度差异)。值越小越敏感，检测到的接缝越多"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, images, fix_flash=True, fix_exposure=True,
                fix_motion=False, fix_scene_cut=False, threshold=0.05) -> io.NodeOutput:
        if images is None or images.shape[0] < 2:
            return io.NodeOutput(images)

        images = images.clone()
        seams = _detect_seams(images, threshold)
        if not seams:
            print("[H3-Seam] 未检测到明显接缝")
            return io.NodeOutput(images)

        actions = []
        for b in seams:
            kind, delta = _classify_boundary(images, b, threshold)
            if kind == "clean":
                actions.append((b, kind, "无修正"))
                continue
            should_fix = {
                "flash": fix_flash,
                "exposure": fix_exposure,
                "motion": fix_motion,
                "scene_cut": fix_scene_cut,
            }.get(kind, False)
            if should_fix:
                images = _correct_frame_lerp(images, b, frames=1)
                actions.append((b, kind, f"lerp 修正 (delta={delta:.3f})"))
            else:
                actions.append((b, kind, f"跳过 (开关关闭, delta={delta:.3f})"))

        print(f"[H3-Seam] 检测到 {len(seams)} 处候选接缝:")
        for b, kind, act in actions:
            print(f"[H3-Seam]   帧{b}: {kind} -> {act}")
        return io.NodeOutput(images)
