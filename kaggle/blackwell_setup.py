"""Blackwell (RTX PRO 6000) compatibility fixes for Kaggle Notebooks.

Why this file exists: the Nemotron-H hybrid Mamba+attention arch uses Triton
kernels (rmsnorm_fn + Mamba fast-path) that crash on Blackwell SMs. Every
public training notebook that lands above 0.80 LB applies the same two
workarounds. They're load-bearing and undocumented anywhere else.

Import order matters:
    1. `set_cuda_alloc_env()` BEFORE `import torch`.
    2. Load the model.
    3. `patch_for_blackwell()` AFTER the model is loaded (otherwise the
       modeling_nemotron_h modules haven't been imported and the patch
       no-ops silently).

Sources verified 2026-05-18:
- notebooks/kienngx__nvidia-nemotron-training-cot-labels/...ipynb (cells 5 + 11)
- notebooks/_host_metric/nvidia-nemotron-metric.ipynb (ptxas env var only)
"""

from __future__ import annotations

import os
import sys


def set_cuda_alloc_env() -> None:
    """Reduce CUDA memory fragmentation. MUST be called before `import torch`."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def patch_rmsnorm_with_pure_pytorch() -> int:
    """Replace any module-level `rmsnorm_fn` with a pure-PyTorch implementation.

    The Triton-backed rmsnorm in modeling_nemotron_h crashes on Blackwell. The
    pure-PyTorch fallback is slower but numerically equivalent (the upcast/dtype
    path matches the original). Returns the number of modules patched.
    """
    import torch
    import torch.nn.functional as F

    def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                         group_size=None, norm_before_gate=True, upcast=True):
        dtype = x.dtype
        if upcast:
            x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        out = x_normed * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None:
            out = out * F.silu(z.float())
        return out.to(dtype)

    patched = 0
    for name, mod in list(sys.modules.items()):
        if hasattr(mod, "rmsnorm_fn"):
            mod.rmsnorm_fn = _pure_rmsnorm_fn
            patched += 1
    return patched


def disable_nemotron_fast_path() -> int:
    """Force `is_fast_path_available = False` in every loaded modeling_nemotron_h module.

    The Mamba fast-path uses Triton kernels that segfault on Blackwell. Must be
    called AFTER the model is loaded (otherwise the modeling_nemotron_h modules
    haven't been imported yet). Returns the number of modules patched.
    """
    patched = 0
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name:
            mod.is_fast_path_available = False
            patched += 1
    return patched


def set_blackwell_ptxas_env() -> None:
    """Tell Triton where to find the Blackwell ptxas binary.

    Assumes the `ptxas-blackwell` binary exists at /tmp/ptxas-blackwell. The
    standard approach is to attach the public `metric/nvidia_metric_utility_script`
    kernel as a data source and untar it into /tmp before calling this.
    """
    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/tmp/ptxas-blackwell")
    try:
        import triton.backends.nvidia.compiler as nv_compiler  # type: ignore
        nv_compiler.get_ptxas_version = lambda arch: "12.0"
    except ImportError:
        pass


def patch_for_blackwell() -> dict:
    """Apply all post-model-load Blackwell patches. Returns a summary dict."""
    return {
        "rmsnorm_patched": patch_rmsnorm_with_pure_pytorch(),
        "fast_path_disabled": disable_nemotron_fast_path(),
        "ptxas_blackwell_path": os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
    }
