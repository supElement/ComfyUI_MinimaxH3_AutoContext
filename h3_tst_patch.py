"""h3_tst_patch.py - H3 时间状态传输 (TST) 注意力校正节点
使用说明：
- 工作流接法: 模型加载 -> [其它 attention patch 节点 (如有)] -> 本节点 -> 采样节点
  链式合并: 挂载本节点时会拾取模型上已有的其它 optimized_attention_override
  (如 Kijai sage attention patch) 并保留在最内层，校正后转交，双方共存。
  注意方向性: 本节点必须放在其它直接赋值型 override 节点的下游 (更靠近采样器)，
  否则对方后挂载时会把本节点顶掉。官方 ModelAttentionBackend 走独立注册表机制，
  不占用本键，任意位置兼容
- 长度 guard 会拒绝短于视频段的 attention 调用 (如 token-refiner)，
  保证只有真正的视频注意力推进层计数器
- TST 会改变输出结果，启用后二采缓存语义已变 (本节点会把 tau 写入 transformer_options,
  主节点可读取该值写入缓存指纹, 见 patch_model 内注释)
"""
import math
from functools import partial

import torch
import comfy.patcher_extension
from comfy_api.latest import io

try:
    from comfy.ldm.modules.attention import AttentionTensorContainer
except ImportError:
    AttentionTensorContainer = None


# ==================== 核心工具 ====================

def _unwrap(tensor):
    """H3 把 q/k/v 包在 AttentionTensorContainer 里；wrap_attn 调用 override 前
    已 take() 解包，因此这里通常收到普通 tensor。保留一层防御性解包。"""
    if AttentionTensorContainer is not None and isinstance(tensor, AttentionTensorContainer):
        return tensor.peek()
    return tensor


def _spectral_tension(q, k, v0, frames, rows_per_frame, eps=1e-8):
    """单头符号谱张力 (帧级传输算子)。
    q, k: [1, heads, S, head_dim] (post-norm post-RoPE)；视频行从 v0 开始。
    返回 [heads] fp32；T>0 过度混合，T<0 碎片化 (论文 Eqs. 1-3)。
    """
    heads = q.shape[1]
    head_dim = q.shape[-1]
    qv = q[0, :, v0:].reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    kv = k[0, :, v0:].reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    a = (qv @ kv.transpose(-2, -1) * head_dim ** -0.5).softmax(dim=-1).clamp(min=eps)
    log_f = math.log(frames)
    h_row = -(a * a.log()).sum(dim=-1).mean(dim=-1) / log_f
    gram = a @ a.transpose(-2, -1)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=eps)
    eig = torch.linalg.eigvalsh(gram / trace[:, None, None]).clamp(min=eps)
    eig = eig / eig.sum(dim=-1, keepdim=True).clamp(min=eps)
    h_vn = -(eig * eig.log()).sum(dim=-1) / log_f
    return h_row - h_vn


# ==================== attention override ====================

def _attention_override(state, inner_override=None):
    """挂到 transformer_options["optimized_attention_override"] 的回调。
    链式合并: 工作流中已挂的其它 override 作为 inner_override 保留在最内层，
    TST 校正 q 后原样转交 (含 q/k/v/mask 全部参数)，双方都生效。
    只处理长度匹配视频段的 attention 调用 (token-refiner 等短序列调用被
    长度 guard 拒绝，不推进层计数器，但仍会转交给内层 override)。"""

    def _dispatch(func, q, k, v, heads, mask, **kwargs):
        if inner_override is not None:
            return inner_override(func, q, k, v, heads, mask=mask, **kwargs)
        return func(q, k, v, heads, mask=mask, **kwargs)

    def override(func, q, k, v, heads, mask=None, **kwargs):
        frames = state["frames"]
        rows = state["rows_per_frame"]
        qt = _unwrap(q)
        s = qt.shape[2]
        if frames is None or frames < 2 or s <= frames * rows:
            return _dispatch(func, q, k, v, heads, mask, **kwargs)

        state["calls_this_forward"] += 1

        layers = state["layers"]
        layer = (state["calls_this_forward"] - 1) % max(1, layers)
        total_steps = state["total_steps"]
        step = min(state["step"], total_steps - 1)
        layer_weight = 0.5 - 0.5 * math.cos(math.pi * layer / (layers - 1)) if layers > 1 else 1.0
        step_weight = 0.5 + 0.5 * math.cos(math.pi * step / (total_steps - 1)) if total_steps > 1 else 1.0

        tension = _spectral_tension(qt, _unwrap(k), s - frames * rows, frames, rows)
        gamma = torch.exp(tension * (state["tau"] * layer_weight * step_weight))

        if state["tau"] != 0.0:
            qt[0, :, s - frames * rows:].mul_(gamma.to(qt.dtype)[:, None, None])

        return _dispatch(func, q, k, v, heads, mask, **kwargs)

    return override


# ==================== 每 forward 包装 ====================

def _forward_wrapper(state, executor, x, timestep, context, transformer_options,
                     minimax_payload=None, **kwargs):
    """DIFFUSION_MODEL wrapper：每个 forward 开始时复位计数器并重推视频形状。"""
    if state["calls_this_forward"] > 0:
        state["layers"] = state["calls_this_forward"]
    state["calls_this_forward"] = 0

    video = x[0] if isinstance(x, (list, tuple)) else x
    if isinstance(video, torch.Tensor) and video.ndim == 5:
        state["frames"] = int(video.shape[2])
        state["rows_per_frame"] = ((video.shape[3] + 1) // 2) * ((video.shape[4] + 1) // 2)
    else:
        state["frames"] = None

    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is not None:
        sigma = float(timestep.flatten()[0]) / 1000.0
        state["total_steps"] = max(1, int(sigmas.numel()) - 1)
        state["step"] = int((sigmas.float().cpu() - sigma).abs().argmin())

    return executor(x, timestep, context, transformer_options,
                    minimax_payload=minimax_payload, **kwargs)


# ==================== 对外接口 ====================

def patch_model(model, tau):
    """给 ModelPatcher 挂 TST wrapper + attention override (clone 后修改，不影响原模型)。"""
    state = {
        "tau": float(tau),
        "frames": None,
        "rows_per_frame": None,
        "step": 0,
        "total_steps": 1,
        "layers": 50,  # 默认猜测值，首个 forward 后由实际 attention 调用数修正
        "calls_this_forward": 0,
    }
    patched = model.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "h3_tst_patch", partial(_forward_wrapper, state))
    to = dict(patched.model_options.get("transformer_options") or {})
    to["optimized_attention_override"] = _attention_override(
        state, to.get("optimized_attention_override"))
    to["h3_tst_tau"] = float(tau)
    patched.model_options["transformer_options"] = to
    return patched


# ==================== 节点定义 ====================

class H3TSTPatch(io.ComfyNode):
    """H3 TST 注意力校正节点：基于谱张力的 per-head 自适应 query 温度，
    抑制时序闪烁与小脸崩坏 (碎片化/过度混合两类时序状态均被拉回稳态)。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3TSTPatch",
            display_name="Minimax_H3_TST_AttentionPatch",
            category="MinimaxH3_AutoContext",
            description="H3 时间状态传输 (TST) 注意力校正：谱张力诊断帧级传输算子状态，"
                        "自适应缩放视频行 query。与分段推理节点串联使用，"
                        "对本节点之后所有段的所有 attention 调用生效。"
                        "与官方 ModelAttentionBackend 天然兼容；"
                        "与其它 attention patch 节点链式共存 (本节点需放在其下游)",
            inputs=[
                io.Model.Input("model", tooltip="输入模型 (MiniMax H3)，输出补丁后的模型接到采样节点"),
                io.Float.Input("tau", default=0.2, min=0.0, max=1.0, step=0.01,
                               tooltip="校正强度。0=透传不挂载 (可作 A/B 基线)。"
                                       "常用范围 0.1~0.3，小脸崩坏明显时可试 0.3。"
                                       "实际校正量 = tau * 层权重(深层更强) * 步权重(早期更强)"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, tau=0.2) -> io.NodeOutput:
        if tau == 0.0:
            print("[H3-Auto] TST: tau=0，直接透传模型 (未挂载任何补丁)")
            return io.NodeOutput(model)
        patched = patch_model(model, tau)
        print(f"[H3-Auto] TST 注意力校正已挂载: tau={tau}")
        print("[H3-Auto] 提示: TST 会改变输出，采样节点缓存需重新生成 (与换 LoRA/加速节点后相同)")
        return io.NodeOutput(patched)
