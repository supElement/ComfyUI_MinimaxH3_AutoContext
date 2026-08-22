"""
ComfyUI-H3-Auto-Context
一键式 MiniMax H3 长视频自动化生成节点
"""

from .nodes import H3AutoContextSampler, H3ParameterNode

# 独立导入 seam_correction，失败不影响其他节点
try:
    from .seam_correction import H3SeamCorrection
except ImportError:
    H3SeamCorrection = None
    print("[H3-AutoContext] 警告：H3SeamCorrection 加载失败，请检查依赖（如 transnetv2-pytorch）。")

NODE_CLASS_MAPPINGS = {
    "H3AutoContextSampler": H3AutoContextSampler,
    "H3Parameter": H3ParameterNode,
}
if H3SeamCorrection is not None:
    NODE_CLASS_MAPPINGS["H3SeamCorrection"] = H3SeamCorrection

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]