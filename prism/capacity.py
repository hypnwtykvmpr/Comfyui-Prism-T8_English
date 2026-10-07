"""Conservative, live GPU admission checks; never evict another model.

These checks reduce OOM risk, not reserve memory against unrelated processes.
Working-space estimates are deliberately conservative and are not benchmark claims.
"""
from contextvars import ContextVar
import math
import ctypes
import sys

GIB = 1024 ** 3
WORKSPACE = ContextVar("prism_gpu_workspace", default=0)


def commit_headroom():
    """Read the machine commit limit, not merely unused physical RAM."""
    if sys.platform == "win32":
        from ctypes import wintypes
        class PerformanceInfo(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
                "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
                "KernelNonpaged", "PageSize")] + [(name, wintypes.DWORD) for name in (
                "HandleCount", "ProcessCount", "ThreadCount")]
        info = PerformanceInfo()
        info.cb = ctypes.sizeof(info)
        api = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
        api.argtypes = [ctypes.POINTER(PerformanceInfo), wintypes.DWORD]
        api.restype = wintypes.BOOL
        if not api(ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return max(0, (info.CommitLimit - info.CommitTotal) * info.PageSize)
    if sys.platform.startswith("linux"):
        from pathlib import Path
        values = {line.split(':')[0]: int(line.split()[1])*1024
                  for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
        return max(0, min(values['MemAvailable'], values['CommitLimit']-values['Committed_AS']))
    raise OSError("No supported commit measurement on this platform")


def require_commit(needed):
    if type(needed) is not int or needed < 0:
        raise ValueError("Invalid Prism commit estimate")
    try:
        free = commit_headroom()
    except Exception as error:
        raise MemoryError("Unable to check system commit; Prism model load refused") from error
    if type(free) is not int or free < 0:
        raise MemoryError("Invalid system commit measurement; Prism model load refused")
    reserve = 16 * GIB
    if free < needed + reserve:
        raise MemoryError(
            f"Insufficient system commit headroom: {free/GIB:.2f} GiB available; "
            f"Prism needs an estimated {needed/GIB:.2f} GiB plus {reserve/GIB:.0f} GiB reserve. "
            "Model load refused before allocation; other applications are left untouched."
        )


def require_headroom(free, total, weights, workspace):
    values = (free, total, weights, workspace)
    if any(type(v) is not int or v < 0 for v in values) or total <= 0 or free > total:
        raise MemoryError("Cannot establish valid GPU memory headroom; Prism load refused")
    reserve = max(4 * GIB, math.ceil(total / 10))
    required = weights + workspace + reserve
    if free < required:
        raise MemoryError(
            f"Insufficient GPU memory: {free/GIB:.2f} GiB free; need approximately "
            f"{weights/GIB:.2f} GiB for transfer, {workspace/GIB:.2f} GiB working space "
            f"and {reserve/GIB:.2f} GiB reserve. Prism refused the load. "
            "Wait for other GPU work, use block offload, or reduce frames/resolution."
        )


def workspace_estimate(settings):
    width, height, frames = (settings[k] for k in ("width", "height", "num_frames"))
    if any(type(v) is not int or v <= 0 for v in (width, height, frames)):
        raise ValueError("Positive integer dimensions and frame count required")
    tokens = math.ceil(width/16) * math.ceil(height/16) * (math.ceil((frames-1)/4)+1)
    # Model width 5120, BF16, sixteen intermediate equivalents; plus decode/output
    # buffers. Actual peak depends on kernels/tiling and must be measured separately.
    return 2 * GIB + tokens * 5120 * 2 * 16 + width * height * frames * 3 * 4 * 3


def check_transfer(module, device):
    import torch
    device = torch.device(device)
    if device.type != "cuda":
        return
    require_commit(0)
    index = device.index if device.index is not None else torch.cuda.current_device()
    needed, largest_int8, seen = 0, 0, set()
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        if tensor.is_meta:
            raise MemoryError("Prism cannot transfer an unmaterialized model")
        if tensor.device.type == "cuda" and tensor.device.index == index:
            continue
        needed += tensor.numel() * tensor.element_size()
        if tensor.dtype == torch.int8:
            # Portable dequant can overlap FP32 weight, scaled FP32 result and
            # BF16 cast. Do not budget only the final BF16 tensor.
            largest_int8 = max(largest_int8, tensor.numel() * 10)
    try:
        free, total = torch.cuda.mem_get_info(index)
    except Exception as error:
        raise MemoryError("GPU memory query failed; Prism load refused") from error
    require_headroom(int(free), int(total), needed + largest_int8, WORKSPACE.get())


def preflight(parts, settings, device):
    sizes = [part.path.stat().st_size for part in parts.values()]
    if not sizes:
        raise ValueError("No Prism components supplied")
    # Count the full weights even when immutable CPU mappings can avoid private
    # copies. Include output/decode buffers and conversion/allocator contingency.
    output = settings['width'] * settings['height'] * settings['num_frames'] * 3 * 4
    require_commit(math.ceil(sum(sizes)*1.25) + 3*output + 2*GIB)
    import torch
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Prism requires a CUDA GPU")
    # Block offload has one encoder/block active at a time. The largest full
    # component is a conservative bound for the weights in that phase.
    transfer = max(sizes) if settings["offload"] == "block" else sum(sizes)
    free, total = torch.cuda.mem_get_info(device)
    require_headroom(int(free), int(total), transfer, workspace_estimate(settings))


def guarded_to(module, device, **kwargs):
    """Explicit guarded model transfer for standalone diagnostic scripts."""
    check_transfer(module, device)
    return module.to(device, **kwargs)


def check_conversion(weight, device, rows):
    """Bound the streaming quantizer's output and simultaneous chunk buffers."""
    import torch
    chunk_bytes = min(rows, weight.shape[0]) * weight.shape[1] * 4 * 16
    require_commit(weight.numel() + weight.shape[0] * 4 + chunk_bytes)
    target = torch.device(device)
    if target.type == "cuda":
        try:
            free, total = torch.cuda.mem_get_info(target)
        except Exception as error:
            raise MemoryError("GPU query failed; Prism conversion refused") from error
        require_headroom(int(free), int(total), 0, chunk_bytes)
