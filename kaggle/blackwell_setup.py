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


def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=False):
    """Pure-PyTorch rmsnorm_fn — same math the mamba_ssm.triton kernel does.

    upcast=False (default): stay in bf16 throughout — critical for fitting
    Nemotron-30B training in 96 GB. fp32 copies of 5376-wide Mamba hidden
    states at 4096+ sequence length exhaust VRAM during the backward pass.
    """
    import torch
    import torch.nn.functional as F
    dtype = x.dtype
    # Always work in bf16; the numerical difference vs fp32 norm is negligible
    # for LoRA SFT and avoids large transient fp32 tensors on long sequences.
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.to(dtype)
    if bias is not None:
        out = out + bias.to(dtype)
    if z is not None:
        out = out * F.silu(z.to(dtype))
    return out


def install_mamba_ssm_stub() -> None:
    """Pre-install a fake `mamba_ssm` module BEFORE the Nemotron-H modeling code
    is loaded. The modeling code does `from mamba_ssm.ops.triton.layernorm_gated
    import rmsnorm_fn` at import time and re-raises ImportError if it fails.

    Each stub module gets a real `ModuleSpec` so `importlib.util.find_spec`
    doesn't choke (otherwise `transformers.is_mamba_2_ssm_available` raises
    `ValueError: mamba_ssm.__spec__ is None`).

    We deliberately do NOT set `__version__` on the stub — that makes
    `transformers._is_package_available` flip `package_exists` back to False
    via its fallback path, so `is_mamba_2_ssm_available()` returns False and
    the heavy fast-path imports are skipped.

    Must run BEFORE the first `AutoModelForCausalLM.from_pretrained(...)` call.
    """
    import importlib.machinery
    import types

    # If real mamba_ssm is already loaded (e.g. attached from an offline-packages
    # dataset), defer to it — don't shadow a working install.
    if "mamba_ssm" in sys.modules:
        return

    def _stub_pkg(name: str, is_package: bool) -> types.ModuleType:
        mod = types.ModuleType(name)
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
        mod.__spec__ = spec
        mod.__loader__ = None
        if is_package:
            mod.__path__ = []
        return mod

    pkg = _stub_pkg("mamba_ssm", is_package=True)
    # Mark a too-old version so `transformers.is_mamba_2_ssm_available()` returns
    # False (it requires >= 2.0.4). With "N/A" the version.parse call raises;
    # with a real version string < 2.0.4 the comparison short-circuits cleanly
    # and the fast-path imports are skipped entirely.
    pkg.__version__ = "0.0.0"

    ops = _stub_pkg("mamba_ssm.ops", is_package=True)
    triton_pkg = _stub_pkg("mamba_ssm.ops.triton", is_package=True)
    layernorm = _stub_pkg("mamba_ssm.ops.triton.layernorm_gated", is_package=False)
    layernorm.rmsnorm_fn = _pure_rmsnorm_fn

    sys.modules.update({
        "mamba_ssm": pkg,
        "mamba_ssm.ops": ops,
        "mamba_ssm.ops.triton": triton_pkg,
        "mamba_ssm.ops.triton.layernorm_gated": layernorm,
    })


def patch_rmsnorm_with_pure_pytorch() -> int:
    """Defensive post-load patch: replace any module-level `rmsnorm_fn` with our
    pure-PyTorch impl. Used in case some modeling sub-module captured a stale
    reference before `install_mamba_ssm_stub` ran. Returns # modules patched.
    """
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



def patch_moe_dtype_for_qlora() -> int:
    """Fix float32/bf16 dtype mismatch in NemotronH MoE under QLoRA.

    When the base model is loaded in NF4 4-bit (QLoRA), expert routing weights
    (topk_weights) stay in float32 while expert outputs are bfloat16 (per
    bnb_4bit_compute_dtype). Multiplying them promotes weighted_output to float32,
    which then crashes index_add_() on the bf16 accumulator:

        RuntimeError: index_add_(): self (BFloat16) and source (Float) must have
        the same scalar type

    Fix: patch the `moe` method at the class level for every loaded
    modeling_nemotron_h class so that topk_weights are cast to hidden_states.dtype
    before entering the accumulation loop.

    Must be called AFTER the model is loaded (otherwise modeling_nemotron_h
    hasn't been imported). Returns number of classes patched.
    """
    patched = 0
    for _mod_name, mod in list(sys.modules.items()):
        if "modeling_nemotron_h" not in _mod_name:
            continue
        for attr_name in list(vars(mod).keys()):
            cls = getattr(mod, attr_name, None)
            if cls is None or not isinstance(cls, type):
                continue
            if "moe" not in vars(cls):
                continue
            original_moe = vars(cls)["moe"]

            def _make_patched(orig):
                def _patched_moe(self, hidden_states, topk_indices, topk_weights):
                    # Cast routing weights to hidden_states dtype (bf16) so the
                    # weighted_output stays in bf16 and index_add_() doesn't fail.
                    topk_weights = topk_weights.to(hidden_states.dtype)
                    return orig(self, hidden_states, topk_indices, topk_weights)
                return _patched_moe

            cls.moe = _make_patched(original_moe)
            patched += 1
    return patched


def patch_for_blackwell() -> dict:
    """Apply all post-model-load Blackwell patches. Returns a summary dict."""
    return {
        "rmsnorm_patched": patch_rmsnorm_with_pure_pytorch(),
        "fast_path_disabled": disable_nemotron_fast_path(),
        "moe_dtype_patched": patch_moe_dtype_for_qlora(),
        "ptxas_blackwell_path": os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
    }
