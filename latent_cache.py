import os
import threading
import queue
import torch
import hashlib
import numpy as np
from comfy.nested_tensor import NestedTensor

def compute_input_hash(input_latent):
    """
    计算输入 Latent 的指纹哈希（仅取视频部分的前1024个元素）。
    用于检测一采结果是否变化，从而决定二采缓存是否失效。
    """
    if input_latent is None:
        return None
    # 尝试提取视频 latent（5维）
    v_lat = None
    if hasattr(input_latent, "is_nested") and input_latent.is_nested:
        tensors = list(input_latent.unbind())
        for t in tensors:
            if t.dim() == 5:
                v_lat = t
                break
    elif isinstance(input_latent, dict):
        samples = input_latent.get("samples")
        if samples is not None:
            if hasattr(samples, "is_nested") and samples.is_nested:
                for t in samples.unbind():
                    if t.dim() == 5:
                        v_lat = t
                        break
            elif isinstance(samples, torch.Tensor):
                v_lat = samples
    else:
        v_lat = input_latent

    if v_lat is None:
        return None
    flat = v_lat.flatten()
    if flat.numel() == 0:
        return None
    # 取前1024个字节计算MD5
    sample = flat[:1024].cpu().numpy().tobytes()
    return hashlib.md5(sample).hexdigest()


def normalize_cache_dir(raw_dir):
    """将路径统一为系统标准格式，支持正/反斜杠"""
    if not raw_dir:
        return ""
    # 替换反斜杠为正斜杠，然后规范化
    normalized = os.path.normpath(raw_dir.replace('\\', '/'))
    return normalized

# ------------------ 内部工具函数 ------------------
def _unwrap_nested(tensor):
    """将 NestedTensor 解包为普通张量列表，便于存储"""
    if isinstance(tensor, NestedTensor):
        return [t.cpu() for t in tensor.unbind()]
    if isinstance(tensor, (list, tuple)):
        return [t.cpu() for t in tensor]
    return tensor.cpu()

def _wrap_nested(tensors):
    """将张量列表重新打包为 NestedTensor"""
    if isinstance(tensors, (list, tuple)) and len(tensors) > 0:
        if len(tensors) == 1 and not isinstance(tensors[0], (list, tuple)):
            return tensors[0]
        try:
            return NestedTensor(tuple(tensors))
        except:
            # 如果打包失败，返回原始列表（降级处理）
            return tensors
    return tensors

def _compute_input_hash(input_latent):
    """计算输入 Latent 的指纹哈希（仅取视频部分的前1024个元素）"""
    if input_latent is None:
        return None
    # 尝试取出视频 latent
    if hasattr(input_latent, "is_nested") and input_latent.is_nested:
        tensors = list(input_latent.unbind())
        v_lat = None
        for t in tensors:
            if t.dim() == 5:  # 视频 latent
                v_lat = t
                break
        if v_lat is None:
            return None
    elif isinstance(input_latent, dict):
        # 可能是 dict
        return None  # 简单处理，忽略
    else:
        v_lat = input_latent

    # 取前1024个元素（或全部，如果小于1024）
    flat = v_lat.flatten()
    if flat.numel() == 0:
        return None
    sample = flat[:1024].cpu().numpy().tobytes()
    return hashlib.md5(sample).hexdigest()

# ------------------ 缓存文件操作 ------------------
def get_cache_path(cache_dir, seg_idx):
    """生成段缓存文件路径"""
    if not cache_dir:
        return None
    normalized_dir = normalize_cache_dir(cache_dir)
    os.makedirs(normalized_dir, exist_ok=True)
    return os.path.join(normalized_dir, f"seg_{seg_idx:04d}.pt")

def save_segment_latent_sync(cache_dir, seg_idx, samples, x0, metadata=None):
    """同步保存一段 latent（供异步线程调用）"""
    if not cache_dir:
        return
    path = get_cache_path(cache_dir, seg_idx)
    if not path:
        return

    # 提取 samples 内部张量
    if isinstance(samples, dict):
        samples_tensor = samples.get("samples")
    else:
        samples_tensor = samples

    # 解包并移至 CPU
    samples_unwrapped = _unwrap_nested(samples_tensor)
    x0_unwrapped = _unwrap_nested(x0) if x0 is not None else None

    data = {
        "samples": samples_unwrapped,
        "x0": x0_unwrapped,
        "metadata": metadata or {},
    }
    # 原子保存（先写临时文件，再重命名，防止写入中断导致半文件）
    temp_path = path + ".tmp"
    try:
        torch.save(data, temp_path)
        os.replace(temp_path, path)  # 原子替换
    except Exception as e:
        print(f"[H3-Cache] 保存段 {seg_idx} 失败: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def load_segment_latent(cache_dir, seg_idx, current_metadata=None):
    """
    加载一段缓存，若提供了 current_metadata 则校验关键参数。
    返回 (samples_dict, x0) 或 (None, None)
    """
    if not cache_dir:
        return None, None
    path = get_cache_path(cache_dir, seg_idx)
    if not os.path.exists(path):
        return None, None

    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[H3-Cache] 加载段 {seg_idx} 失败: {e}")
        return None, None

    saved_meta = data.get("metadata", {})

    # 参数校验（如果提供了 current_metadata）
    if current_metadata:
        sensitive_keys = ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "video_context_denoise", "sigmas_hash", "input_hash"]
        mismatch = False
        for k in sensitive_keys:
            if saved_meta.get(k) != current_metadata.get(k):
                mismatch = True
                break
        if mismatch:
            print(f"[H3-Cache] 段 {seg_idx} 参数或输入指纹变更，删除旧缓存")
            try:
                os.remove(path)
            except:
                pass
            return None, None

    # 重建张量
    samples_unwrapped = data["samples"]
    x0_unwrapped = data.get("x0")

    samples_tensor = _wrap_nested(samples_unwrapped)
    x0_tensor = _wrap_nested(x0_unwrapped) if x0_unwrapped is not None else None

    samples_dict = {"samples": samples_tensor}
    # 如果有 noise_mask，可以重建，但缓存中通常不存，忽略
    return samples_dict, x0_tensor

# ------------------ 异步保存队列 ------------------
_save_queue = queue.Queue()
_save_thread_started = False

def _save_worker():
    """后台线程工作函数"""
    while True:
        try:
            cache_dir, seg_idx, samples, x0, metadata = _save_queue.get(timeout=1)
            if cache_dir is None:  # 结束信号
                break
            save_segment_latent_sync(cache_dir, seg_idx, samples, x0, metadata)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[H3-Cache] 异步保存线程异常: {e}")

def start_save_thread():
    """启动后台保存线程（只启动一次）"""
    global _save_thread_started
    if not _save_thread_started:
        thread = threading.Thread(target=_save_worker, daemon=True)
        thread.start()
        _save_thread_started = True

def save_segment_latent_async(cache_dir, seg_idx, samples, x0, metadata=None):
    """异步保存一段 latent（将任务放入队列）"""
    if not cache_dir:
        return
    start_save_thread()
    _save_queue.put((cache_dir, seg_idx, samples, x0, metadata))

def flush_save_queue():
    """等待所有保存任务完成（可选，在程序退出前调用）"""
    _save_queue.join()