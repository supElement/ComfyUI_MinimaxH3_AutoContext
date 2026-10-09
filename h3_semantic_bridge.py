"""h3_semantic_bridge.py — MiniMax H3 语义桥

移植自: https://github.com/Speach1sdef178/MiniMax-H3-Semantic-Bridge
       (nodes.py 中的 SemanticStudent / _rms_normalize / _magnitude_match / _apply_distilled_bridge)

公式: C = H + alpha * (S - H)
  H = 原始 H3 条件张量 [B, T, 5120]
  S = 学生适配器 (5120→512→512→5120 SiLU MLP, ~11MB) 的输出
      输入前先做 RMS 归一化, 输出后按 magnitude_match 模式幅度对齐回 H
alpha: 官方起点 0.10, 官方 A/B 示例用 0.15
magnitude_match: per_token (官方推荐) / global / none

适配器位置: ComfyUI/models/semantic_bridge/MiniMaxH3_SemanticBridge_v1.safetensors
只变换 conditioning 张量, 不修改 H3 DiT 权重, 不触碰 minimax_keyframes/minimax_refs 等键。
与上游差异: 校验失败时打印警告并返回原 conditioning (软失败), 而非抛 RuntimeError,
避免长视频分段生成被单段异常中断。
"""
import os
import gc
import torch
import torch.nn as nn
from safetensors.torch import load_file

import folder_paths

# ---- 注册 models/semantic_bridge 目录 (与上游一致) ----
DISTILLED_DIR = os.path.join(folder_paths.models_dir, "semantic_bridge")
# os.makedirs(DISTILLED_DIR, exist_ok=True)
# try:
    # folder_paths.add_model_folder_path("semantic_bridge", DISTILLED_DIR)
# except Exception:
    # pass

_STUDENT_CACHE = {}


def _list_safetensors():
    try:
        files = folder_paths.get_filename_list("semantic_bridge")
    except Exception:
        files = []
    files = [x for x in files if x.lower().endswith(".safetensors")]
    return sorted(files) if files else ["NO_DISTILLED_ADAPTER_FOUND.safetensors"]


def _full_adapter_path(name):
    path = None
    try:
        path = folder_paths.get_full_path("semantic_bridge", name)
    except Exception:
        pass
    if path and os.path.isfile(path):
        return path
    fallback = os.path.join(DISTILLED_DIR, name)
    if os.path.isfile(fallback):
        return fallback
    return None


def _rms_normalize(x):
    x = x.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return x / rms


def _magnitude_match(source, target):
    source = source.float()
    target = target.float()
    source_rms = torch.sqrt(source.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    target_rms = torch.sqrt(target.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    return source * (target_rms / source_rms)


class SemanticStudent(nn.Module):
    """上游原样: 5120→512→512→5120, SiLU, 全 bias。权重键名严格固定。"""
    def __init__(self, input_dim=5120, hidden_dim=512, output_dim=5120):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def _load_student(adapter_name, device):
    path = _full_adapter_path(adapter_name)
    if path is None:
        print(f"[H3-Bridge] adapter not found: {adapter_name} (expected in {DISTILLED_DIR})\n"
              f"[H3-Bridge] 未找到适配器: {adapter_name} (应在 {DISTILLED_DIR})")
        return None
    key = (os.path.abspath(path), str(device))
    cached = _STUDENT_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        weights = load_file(path, device="cpu")
    except Exception as e:
        print(f"[H3-Bridge] failed to load adapter: {e}\n[H3-Bridge] 适配器加载失败: {e}")
        return None

    required = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
    missing = [k for k in required if k not in weights]
    if missing:
        print(f"[H3-Bridge] invalid adapter, missing tensors: {missing}\n"
              f"[H3-Bridge] 适配器缺少张量: {missing}")
        return None
    if tuple(weights["fc1.weight"].shape) != (512, 5120):
        print(f"[H3-Bridge] unexpected fc1.weight shape: {tuple(weights['fc1.weight'].shape)}")
        return None
    if tuple(weights["fc2.weight"].shape) != (512, 512):
        print(f"[H3-Bridge] unexpected fc2.weight shape: {tuple(weights['fc2.weight'].shape)}")
        return None
    if tuple(weights["fc3.weight"].shape) != (5120, 512):
        print(f"[H3-Bridge] unexpected fc3.weight shape: {tuple(weights['fc3.weight'].shape)}")
        return None

    model = SemanticStudent()
    with torch.no_grad():
        model.fc1.weight.copy_(weights["fc1.weight"].float())
        model.fc1.bias.copy_(weights["fc1.bias"].float())
        model.fc2.weight.copy_(weights["fc2.weight"].float())
        model.fc2.bias.copy_(weights["fc2.bias"].float())
        model.fc3.weight.copy_(weights["fc3.weight"].float())
        model.fc3.bias.copy_(weights["fc3.bias"].float())
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    _STUDENT_CACHE[key] = model
    print(f"[H3-Bridge] Loaded adapter: {adapter_name}")
    return model


def _apply_distilled_bridge(conditioning, adapter_name, alpha, magnitude_match):
    """上游 _apply_distilled_bridge 的移植, 软失败版。返回新的 conditioning 列表。"""
    result = []

    for item in conditioning:
        if len(item) != 2:
            print(f"[H3-Bridge] unexpected CONDITIONING structure (len={len(item)}), skip\n"
                  f"[H3-Bridge] CONDITIONING 结构异常 (长度={len(item)})，跳过")
            result.append(item)
            continue

        native = item[0]
        metadata = item[1]

        if native.ndim != 3 or native.shape[-1] != 5120:
            print(f"[H3-Bridge] expected [B,T,5120], got {tuple(native.shape)}, skip\n"
                  f"[H3-Bridge] 期望 [B,T,5120], 实际 {tuple(native.shape)}，跳过")
            result.append(item)
            continue

        student = _load_student(adapter_name, native.device)
        if student is None:
            result.append(item)
            continue

        h = native.float()
        x = _rms_normalize(h)

        with torch.inference_mode():
            projected = student(x)

        if magnitude_match == "per_token":
            projected = _magnitude_match(projected, h)
        elif magnitude_match == "global":
            source_rms = torch.sqrt(projected.pow(2).mean() + 1e-8)
            target_rms = torch.sqrt(h.pow(2).mean() + 1e-8)
            projected = projected * (target_rms / source_rms)
        elif magnitude_match == "none":
            pass
        else:
            projected = _magnitude_match(projected, h)

        hybrid = h + float(alpha) * (projected - h)
        hybrid = hybrid.to(dtype=native.dtype)

        new_metadata = dict(metadata)
        new_metadata["sensenova_h3_distilled"] = True
        new_metadata["sensenova_h3_distilled_alpha"] = float(alpha)
        new_metadata["sensenova_h3_distilled_mode"] = magnitude_match
        new_metadata["sensenova_h3_distilled_adapter"] = adapter_name

        result.append([hybrid, new_metadata])

    return result


def apply_semantic_bridge(positive, adapter_name, alpha=0.10, magnitude_match="per_token"):
    """h3_sampler 的对外入口。adapter_name='none' 或 alpha=0 时直接返回。"""
    if not adapter_name or adapter_name == "none" or adapter_name.endswith("NO_DISTILLED_ADAPTER_FOUND.safetensors"):
        return positive
    if alpha is None or float(alpha) <= 0:
        return positive
    return _apply_distilled_bridge(positive, adapter_name, float(alpha), magnitude_match)


def clear_bridge_cache():
    """上游 SenseNovaH3ClearDistilledCache 节点的等价实现。"""
    _STUDENT_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
