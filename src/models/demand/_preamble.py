"""Make torch importable regardless of import order.

On Windows with pandas 3.x + Prophet's cmdstanpy in the same process, torch's
`_load_dll_libraries` fails with WinError 1114 ("DLL initialization routine
failed") when torch is imported AFTER pandas. The root cause is that something
pandas imports puts the process CRT/threading state into a configuration that
torch's DllMain refuses.

Workaround: pre-load every DLL in torch/lib using LoadLibraryExW *before*
torch's normal init runs, in dependency order. Once they're in the process,
torch's own loader is a no-op.

This module is import-order-tolerant: importing it at any point in the program
fixes the issue, even after pandas has been loaded — because torch hasn't been
fully imported yet (the failing `import torch` raises before completing).
"""
from __future__ import annotations

import os
import sys


def _preload_torch_dlls() -> bool:
    if sys.platform != "win32":
        return True
    try:
        import torch  # noqa: F401 -- if it already loaded, we're done
        return True
    except OSError:
        pass
    except Exception:
        return False

    # Find torch's lib directory
    venv_site = next(
        (p for p in sys.path
         if p.endswith("site-packages") and os.path.isdir(os.path.join(p, "torch", "lib"))),
        None,
    )
    if not venv_site:
        return False
    torch_lib = os.path.join(venv_site, "torch", "lib")
    if not os.path.isdir(torch_lib):
        return False

    import ctypes
    os.add_dll_directory(torch_lib)
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    kernel32.LoadLibraryExW.restype = ctypes.c_void_p

    # Dependency order: OpenMP runtime → deps → c10 (core) → cpu → torch → python
    order = [
        "libiomp5md.dll",
        "torch_global_deps.dll",
        "c10.dll",
        "torch_cpu.dll",
        "torch.dll",
        "torch_python.dll",
        "shm.dll",
    ]
    for name in order:
        full = os.path.join(torch_lib, name)
        if not os.path.exists(full):
            continue
        kernel32.LoadLibraryExW(full, None, 0x00001100)

    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


_TORCH_OK = _preload_torch_dlls()
