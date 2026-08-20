"""
seam_correction.py - Minimax H3 Seam Correction
GPU optical-flow alignment + seam warp/blend

Requires:
- PyTorch
- ComfyUI
- torchvision (for torchvision.ops / transforms is not required here)
- CUDA for GPU optical flow

This implementation uses a lightweight GPU Lucas-Kanade-style
iterative flow solver implemented entirely with torch ops.
It avoids OpenCV, so it can stay on CUDA and does not depend on
cv2.remap / CUDA OpenCV availability.

Main features:
1. Seam detection
2. GPU dense optical-flow estimation around each seam
3. Warp the right-side reference toward the left-side reference
4. Multi-frame temporal seam blending
5. Local color correction
6. GPU checkbox
7. Adjustable flow strength / iterations / seam window
"""

import torch
import torch.nn.functional as F
from comfy_api.latest import io

_LONG_SIDE = 320
_FLOW_SCALE = 0.25
_LUMA_W = (0.2126, 0.7152, 0.0722)


# ============================================================
# Basic utilities
# ============================================================

def _work_device(images, use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return images.device


def _downsample_rgb(frames, long_side=_LONG_SIDE):
    rgb = frames[..., :3].clamp(0.0, 1.0).permute(0, 3, 1, 2)
    h, w = rgb.shape[-2:]
    scale = min(1.0, long_side / float(max(h, w)))

    if scale < 1.0:
        rgb = F.interpolate(
            rgb,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )

    return rgb.permute(0, 2, 3, 1)


def _luma(frames):
    w = frames.new_tensor(_LUMA_W)
    return torch.sum(frames[..., :3] * w, dim=-1)


# ============================================================
# Seam detection
# ============================================================

def _detect_seams(images, threshold, device):
    if images.shape[0] < 2:
        return []

    small = _downsample_rgb(images).to(device)
    y = _luma(small)

    deltas = torch.mean(
        torch.abs(y[1:] - y[:-1]),
        dim=(1, 2),
    )

    values = deltas.detach().float().cpu().tolist()

    seams = []

    for i, d in enumerate(values):
        if d < threshold:
            continue

        left = values[i - 1] if i > 0 else 0.0
        right = values[i + 1] if i + 1 < len(values) else 0.0

        if d >= left * 1.3 and d >= right * 1.3:
            seams.append(i + 1)

    return seams


def _classify_boundary(images, boundary, threshold, device):
    small = _downsample_rgb(images).to(device)
    y = _luma(small)

    if boundary < 1 or boundary >= small.shape[0]:
        return "clean", 0.0

    prev = y[boundary - 1]
    curr = y[boundary]

    delta = float(
        torch.mean(torch.abs(prev - curr)).item()
    )

    if boundary + 2 < small.shape[0]:
        cur_internal = 0.5 * (
            torch.mean(
                torch.abs(y[boundary] - y[boundary + 1])
            )
            +
            torch.mean(
                torch.abs(y[boundary + 1] - y[boundary + 2])
            )
        )
        cur_internal = float(cur_internal.item())
    else:
        cur_internal = delta

    if (
        delta >= 0.12
        and cur_internal <= delta * 0.35
        and delta >= threshold * 1.5
    ):
        return "scene_cut", delta

    if boundary + 3 < small.shape[0]:
        after = [
            float(
                torch.mean(
                    torch.abs(prev - y[boundary + k])
                ).item()
            )
            for k in range(1, 4)
        ]

        if all(a > delta * 0.9 for a in after):
            return "exposure", delta

    if boundary + 2 < small.shape[0]:
        d2 = float(
            torch.mean(
                torch.abs(prev - y[boundary + 1])
            ).item()
        )

        if d2 < delta * 0.6:
            return "flash", delta

    if threshold <= delta < 0.12:
        return "motion", delta

    return "clean", delta


# ============================================================
# GPU optical flow
# ============================================================

def _image_to_gray(image):
    """
    [H,W,C] -> [1,1,H,W]
    """
    return (
        image[..., :3]
        .mul(image.new_tensor(_LUMA_W))
        .sum(dim=-1)
        .unsqueeze(0)
        .unsqueeze(0)
    )


def _make_flow_grid(h, w, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    gx = 2.0 * xx / max(w - 1, 1) - 1.0
    gy = 2.0 * yy / max(h - 1, 1) - 1.0

    return torch.stack((gx, gy), dim=-1).unsqueeze(0)


def _warp(image, flow):
    """
    image: [1,C,H,W]
    flow:  [1,2,H,W] pixel displacement

    output(x) = image(x + flow(x))
    """
    _, _, h, w = image.shape

    base = _make_flow_grid(
        h,
        w,
        image.device,
        image.dtype,
    )

    dx = flow[:, 0] * 2.0 / max(w - 1, 1)
    dy = flow[:, 1] * 2.0 / max(h - 1, 1)

    grid = base.clone()
    grid[..., 0] += dx
    grid[..., 1] += dy

    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _gradient_x(x):
    return F.pad(
        x[..., :, 1:] - x[..., :, :-1],
        (0, 1, 0, 0),
    )


def _gradient_y(x):
    return F.pad(
        x[..., 1:, :] - x[..., :-1, :],
        (0, 0, 0, 1),
    )


def _smooth_flow(flow):
    return F.avg_pool2d(
        F.pad(flow, (1, 1, 1, 1), mode="replicate"),
        kernel_size=3,
        stride=1,
    )


@torch.no_grad()
def _gpu_optical_flow(
    source,
    target,
    iterations=12,
    pyramid_levels=3,
):
    """
    Lightweight dense optical flow.

    source: frame A [H,W,C]
    target: frame B [H,W,C]

    Returns flow [1,2,H,W] mapping source -> target.

    This is deliberately conservative: the flow is used to align
    seam frames, not to perform full video optical-flow generation.
    """

    src = _image_to_gray(source)
    tgt = _image_to_gray(target)

    original_h, original_w = src.shape[-2:]

    # Build a small pyramid.
    src_pyr = [src]
    tgt_pyr = [tgt]

    for _ in range(max(1, pyramid_levels - 1)):
        h, w = src_pyr[-1].shape[-2:]

        if min(h, w) < 64:
            break

        src_pyr.append(
            F.avg_pool2d(
                src_pyr[-1],
                kernel_size=2,
                stride=2,
            )
        )

        tgt_pyr.append(
            F.avg_pool2d(
                tgt_pyr[-1],
                kernel_size=2,
                stride=2,
            )
        )

    flow = None

    for level in reversed(range(len(src_pyr))):
        s = src_pyr[level]
        t = tgt_pyr[level]

        h, w = s.shape[-2:]

        if flow is None:
            flow = torch.zeros(
                (1, 2, h, w),
                device=s.device,
                dtype=s.dtype,
            )
        else:
            flow = F.interpolate(
                flow,
                size=(h, w),
                mode="bilinear",
                align_corners=True,
            ) * 2.0

        # Image gradients of target.
        ix = _gradient_x(t)
        iy = _gradient_y(t)

        for _ in range(iterations):
            warped = _warp(s, flow)

            error = t - warped

            # Robust normalization prevents large exposure jumps
            # from generating excessive flow.
            scale = error.abs().mean().clamp_min(0.01)
            error = error / (1.0 + 4.0 * scale)

            ix_w = _warp(ix, flow)
            iy_w = _warp(iy, flow)

            denom = (
                ix_w * ix_w
                + iy_w * iy_w
                + 0.01
            )

            du = error * ix_w / denom
            dv = error * iy_w / denom

            update = torch.cat((du, dv), dim=1)

            # Spatial regularization.
            update = 0.65 * update + 0.35 * _smooth_flow(update)

            flow = flow + update

            # Prevent unstable jumps.
            flow = flow.clamp(-32.0, 32.0)

    # At this point flow is at original resolution.
    if flow.shape[-2:] != (original_h, original_w):
        flow = F.interpolate(
            flow,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=True,
        )

    return flow


# ============================================================
# Local color matching
# ============================================================

def _local_color_match(
    orig,
    reference,
    grid=8,
    strength=0.65,
):
    if strength <= 0:
        return orig

    x = orig[..., :3].permute(2, 0, 1).unsqueeze(0)
    r = reference[..., :3].permute(2, 0, 1).unsqueeze(0)

    h, w = x.shape[-2:]

    gh = max(1, min(grid, h))
    gw = max(1, min(grid, w))

    xm = F.adaptive_avg_pool2d(x, (gh, gw))
    rm = F.adaptive_avg_pool2d(r, (gh, gw))

    diff = F.interpolate(
        rm - xm,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    corrected = (x + diff * strength).clamp(0.0, 1.0)

    out = orig.clone()
    out[..., :3] = corrected.squeeze(0).permute(1, 2, 0)

    return out


# ============================================================
# Seam warp + blend
# ============================================================

@torch.no_grad()
def _warp_blend_seam(
    images,
    boundary,
    window,
    flow_strength,
    blend_strength,
    flow_iterations,
    flow_pyramid,
    device,
):
    """
    Seam correction strategy:

        left frame
           \
            optical flow
             \
              aligned right reference
                   |
             local color match
                   |
             temporal seam blend

    Only a small window around the boundary is modified.
    """

    out = images.clone()

    if boundary < 1 or boundary >= out.shape[0]:
        return out

    window = max(1, int(window))

    left_idx = boundary - 1
    right_idx = min(
        boundary + window,
        out.shape[0] - 1,
    )

    left = out[left_idx].to(device)

    right = out[right_idx].to(device)

    h, w = left.shape[:2]

    # Flow is calculated at reduced resolution to save VRAM.
    scale = min(
        1.0,
        768.0 / float(max(h, w)),
    )

    if scale < 1.0:
        left_small = F.interpolate(
            left[..., :3].permute(2, 0, 1).unsqueeze(0),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )

        right_small = F.interpolate(
            right[..., :3].permute(2, 0, 1).unsqueeze(0),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )

        left_flow = left_small.squeeze(0).permute(1, 2, 0)
        right_flow = right_small.squeeze(0).permute(1, 2, 0)

        flow_small = _gpu_optical_flow(
            right_flow,
            left_flow,
            iterations=flow_iterations,
            pyramid_levels=flow_pyramid,
        )

        flow = F.interpolate(
            flow_small,
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        ) / max(scale, 1e-6)
    else:
        flow = _gpu_optical_flow(
            right,
            left,
            iterations=flow_iterations,
            pyramid_levels=flow_pyramid,
        )

    # Limit flow influence.
    flow = flow * float(flow_strength)

    # Convert right reference to NCHW.
    right_rgb = right[..., :3].permute(2, 0, 1).unsqueeze(0)

    warped_right = _warp(
        right_rgb,
        flow,
    )

    warped_right = (
        warped_right.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
    )

    start = max(
        0,
        boundary - window,
    )

    end = min(
        out.shape[0],
        boundary + window + 1,
    )

    for idx in range(start, end):
        orig = out[idx].to(device)

        # 0 at the outer edge, 1 close to seam.
        dist = abs(
            idx - (boundary - 0.5)
        )

        strength = max(
            0.0,
            1.0 - dist / float(window + 0.5),
        )

        # Temporal position.
        if idx < boundary:
            alpha = 0.08 * strength
        else:
            denom = max(
                1,
                right_idx - boundary,
            )

            alpha = (
                float(idx - boundary + 1)
                / float(denom + 1)
            )

            alpha = min(
                1.0,
                alpha,
            )

        # Construct aligned reference.
        if idx < boundary:
            reference = torch.lerp(
                left,
                warped_right,
                0.15 * strength,
            )
        else:
            reference = torch.lerp(
                left,
                warped_right,
                alpha,
            )

        # Spatial color correction.
        matched = _local_color_match(
            orig,
            reference,
            grid=8,
            strength=blend_strength * strength,
        )

        # Keep original detail.
        mix = (
            0.30 * strength
            if idx != boundary
            else 0.60
        )

        result = torch.lerp(
            orig,
            matched,
            mix,
        )

        out[idx] = result.to(out.device)

    return out


# ============================================================
# ComfyUI Node
# ============================================================

class H3SeamCorrection(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3SeamCorrection",
            display_name="Minimax_H3_Seam_Correction",
            category="MinimaxH3_AutoContext",

            description=(
                "Minimax H3 接缝连续性增强："
                "GPU 光流对齐 + seam warp/blend + "
                "局部颜色匹配 + 多帧过渡"
            ),

            inputs=[

                io.Image.Input(
                    "images",
                    tooltip="拼接后的完整视频帧",
                ),

                io.Boolean.Input(
                    "fix_flash",
                    default=True,
                    tooltip="修正瞬态闪光/亮度跳变",
                ),

                io.Boolean.Input(
                    "fix_exposure",
                    default=True,
                    tooltip="修正曝光渐变",
                ),

                io.Boolean.Input(
                    "fix_motion",
                    default=True,
                    tooltip="修正运动接缝",
                ),

                io.Boolean.Input(
                    "fix_scene_cut",
                    default=False,
                    tooltip="修正场景切换，通常不要开启",
                ),

                io.Float.Input(
                    "threshold",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="接缝检测阈值",
                ),

                io.Int.Input(
                    "seam_window",
                    default=3,
                    min=1,
                    max=12,
                    step=1,
                    tooltip="接缝前后参与融合的帧数",
                ),

                io.Float.Input(
                    "flow_strength",
                    default=0.75,
                    min=0.0,
                    max=1.5,
                    step=0.05,
                    tooltip="GPU 光流位移补偿强度",
                ),

                io.Float.Input(
                    "blend_strength",
                    default=0.65,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="局部颜色/亮度连续性修正强度",
                ),

                io.Int.Input(
                    "flow_iterations",
                    default=12,
                    min=4,
                    max=30,
                    step=1,
                    tooltip="光流迭代次数，越高越慢",
                ),

                io.Int.Input(
                    "flow_pyramid",
                    default=3,
                    min=1,
                    max=5,
                    step=1,
                    tooltip="光流金字塔层数",
                ),

                io.Boolean.Input(
                    "use_gpu",
                    default=True,
                    tooltip="使用 CUDA GPU 进行光流和接缝处理",
                ),
            ],

            outputs=[
                io.Image.Output(
                    display_name="images"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        fix_flash=True,
        fix_exposure=True,
        fix_motion=True,
        fix_scene_cut=False,
        threshold=0.05,
        seam_window=3,
        flow_strength=0.75,
        blend_strength=0.65,
        flow_iterations=12,
        flow_pyramid=3,
        use_gpu=True,
    ) -> io.NodeOutput:

        if images is None or images.shape[0] < 2:
            return io.NodeOutput(images)

        device = _work_device(
            images,
            use_gpu,
        )

        if device.type != "cuda":
            print(
                "[H3-Seam] WARNING: GPU disabled or "
                "CUDA unavailable; optical flow will run "
                "on the input device."
            )

        print(
            "[H3-Seam] "
            f"device={device}, "
            f"window={seam_window}, "
            f"flow_strength={flow_strength:.2f}, "
            f"blend_strength={blend_strength:.2f}, "
            f"iterations={flow_iterations}"
        )

        seams = _detect_seams(
            images,
            threshold,
            device,
        )

        if not seams:
            print(
                "[H3-Seam] 未检测到明显接缝"
            )
            return io.NodeOutput(images)

        out = images.clone()

        actions = []

        for b in seams:

            kind, delta = _classify_boundary(
                out,
                b,
                threshold,
                device,
            )

            should_fix = {
                "flash": fix_flash,
                "exposure": fix_exposure,
                "motion": fix_motion,
                "scene_cut": fix_scene_cut,
            }.get(kind, False)

            if not should_fix:
                actions.append(
                    (
                        b,
                        kind,
                        f"跳过 delta={delta:.3f}",
                    )
                )
                continue

            out = _warp_blend_seam(
                out,
                b,
                window=seam_window,
                flow_strength=flow_strength,
                blend_strength=blend_strength,
                flow_iterations=flow_iterations,
                flow_pyramid=flow_pyramid,
                device=device,
            )

            actions.append(
                (
                    b,
                    kind,
                    (
                        "GPU optical-flow "
                        f"warp/blend "
                        f"delta={delta:.3f}"
                    ),
                )
            )

        print(
            f"[H3-Seam] "
            f"检测到 {len(seams)} 处候选接缝"
        )

        for b, kind, action in actions:
            print(
                f"[H3-Seam] "
                f"帧{b}: {kind} -> {action}"
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {
    "H3SeamCorrection": H3SeamCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SeamCorrection":
        "Minimax_H3_Seam_Correction",
}
