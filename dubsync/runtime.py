"""Runtime setup for optional NVIDIA libraries installed inside the venv."""

from __future__ import annotations

import os
from pathlib import Path
import sys


_DLL_DIRECTORY_HANDLES: list[object] = []
_REGISTERED_DIRECTORIES: set[Path] = set()


def configure_nvidia_dll_paths() -> list[Path]:
    """Make NVIDIA pip-wheel DLL directories discoverable on Windows.

    NVIDIA's CUDA runtime wheels install their DLLs under the virtual
    environment, not a globally installed CUDA Toolkit. The handles are kept
    alive because ``os.add_dll_directory`` removes a directory when its handle
    is garbage-collected.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return []
    root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    directories = [path / "bin" for path in root.iterdir() if (path / "bin").is_dir()] if root.is_dir() else []
    print("NVIDIA DLL directories configured:")
    for directory in directories:
        # print(f"  {directory}")
        if directory not in _REGISTERED_DIRECTORIES:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
            _REGISTERED_DIRECTORIES.add(directory)
    # CTranslate2 dynamically loads CUDA libraries itself. In addition to the
    # Python DLL search registration above, it needs these directories on PATH.
    current = os.environ.get("PATH", "")
    existing = {item.casefold() for item in current.split(os.pathsep)}
    missing = [str(directory) for directory in directories if str(directory).casefold() not in existing]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, current])
    return directories
