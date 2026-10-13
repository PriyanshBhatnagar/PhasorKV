"""Build CUDA sources to cubins with nvcc and launch them on torch's current stream (cuda-python
driver API), so the kernels run inside CUDA graphs like any torch op."""
import ctypes
import hashlib
import os
import subprocess

import torch
from cuda.bindings import driver as cu

NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
CACHE = os.path.join(os.path.dirname(__file__), "csrc", "build")


def _ok(r):
    err = r[0] if isinstance(r, tuple) else r
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}")
    return r[1] if isinstance(r, tuple) and len(r) == 2 else None


class CubinModule:
    def __init__(self, src, arch="sm_120a", flags=("-O3", "-std=c++17")):
        os.makedirs(CACHE, exist_ok=True)
        text = open(src).read()
        key = hashlib.sha1((text + arch + " ".join(flags)).encode()).hexdigest()[:12]
        out = os.path.join(CACHE, f"{os.path.basename(src)}.{key}.cubin")
        if not os.path.exists(out):
            subprocess.run([NVCC, "-cubin", f"-arch={arch}", *flags, "-o", out, src], check=True)
        torch.zeros(1, device="cuda")                     # make torch's primary context current
        self.mod = _ok(cu.cuModuleLoad(out.encode()))
        self.fns = {}

    def launch(self, name, grid, block, args, smem=0):
        """args: tensors (device pointer), ints (int32), ('i64', v), floats (fp32)."""
        f = self.fns.get(name)
        if f is None:
            f = self.fns[name] = _ok(cu.cuModuleGetFunction(self.mod, name.encode()))
        vals = []
        for a in args:
            if isinstance(a, torch.Tensor):
                vals.append(ctypes.c_uint64(a.data_ptr()))
            elif isinstance(a, tuple):
                vals.append(ctypes.c_int64(a[1]))
            elif isinstance(a, float):
                vals.append(ctypes.c_float(a))
            else:
                vals.append(ctypes.c_int32(a))
        ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        _ok(cu.cuLaunchKernel(f, *grid, *block, smem, torch.cuda.current_stream().cuda_stream,
                              ctypes.addressof(ptrs), 0))
