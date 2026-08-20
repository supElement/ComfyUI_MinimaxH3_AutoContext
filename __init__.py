"""
ComfyUI-H3-Auto-Context
一键式 MiniMax H3 长视频自动化生成节点
"""

from .nodes import H3AutoContextSampler, H3ParameterNode
from .seam_correction import H3SeamCorrection

NODE_CLASS_MAPPINGS = {
    "H3AutoContextSampler": H3AutoContextSampler,
    "H3Parameter": H3ParameterNode,
    "H3SeamCorrection": H3SeamCorrection,
}


WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]