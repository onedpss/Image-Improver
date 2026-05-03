"""Real-ESRGAN ncnn-vulkan: find executable, run, optional download, lite mode."""

from __future__ import annotations

import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
VENDOR_DIR = ROOT / "vendor"

_NCNN_ZIP_URLS = {
    "Windows": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip",
    "Linux": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
    "Darwin": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip",
}

DEFAULT_MODEL = "realesrgan-x4plus"

_DEMO_MEDIA_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
_EXTRA_DOC_NAMES = frozenset({"readme_ubuntu.md"})

def _strip_ncnn_bundle_extras(bundle_root: Path) -> None:
    """Remove demo media and extra readme from the unpacked Real-ESRGAN ncnn archive."""
    if not bundle_root.is_dir():
        return
    for p in bundle_root.rglob("*"):
        if not p.is_file():
            continue
        name_l = p.name.lower()
        if name_l in _EXTRA_DOC_NAMES:
            p.unlink(missing_ok=True)
            continue
        if name_l.endswith(".mp4"):
            p.unlink(missing_ok=True)
            continue
        if p.stem.lower().startswith("input") and p.suffix.lower() in _DEMO_MEDIA_SUFFIXES:
            p.unlink(missing_ok=True)


def _ensure_binary_executable(path: Path) -> None:
    """Zip from GitHub often drops +x on Linux; without it execve returns Permission denied."""
    if sys.platform == "win32":
        return
    target = path.resolve()
    if not target.is_file():
        return
    mode = target.stat().st_mode
    if mode & stat.S_IXUSR:
        return
    target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _exe_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("realesrgan-ncnn-vulkan.exe",)
    return ("realesrgan-ncnn-vulkan",)


def find_ncnn_executable(search_root: Path | None = None) -> Path | None:
    """Find realesrgan-ncnn-vulkan in vendor/ and in PATH."""
    names = _exe_names()
    root = search_root if search_root is not None else VENDOR_DIR
    if root.is_dir():
        for name in names:
            for p in root.rglob(name):
                if p.is_file():
                    return p
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def ncnn_bundle_root(exe: Path) -> Path:
    """The directory where ncnn expects the models/ folder (next to exe)."""
    return exe.parent


def run_ncnn_upscale(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    exe: Path | None = None,
    timeout: int = 600,
) -> None:
    """
    Run realesrgan-ncnn-vulkan. cwd = directory with exe, so that -m models works by default.
    """
    binary = exe or find_ncnn_executable()
    if binary is None:
        raise FileNotFoundError(
            "realesrgan-ncnn-vulkan not found. Unzip the archive to the vendor/ "
            "or click «Download NCNN» in the application. "
            "См. https://github.com/xinntao/Real-ESRGAN/releases"
        )
    binary = binary.resolve()
    _ensure_binary_executable(binary)
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cwd = str(ncnn_bundle_root(binary))
    cmd = [
        str(binary),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-n",
        model_name,
    ]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"код {proc.returncode}"
        raise RuntimeError(f"NCNN finished with an error: {err}")


def lite_upscale(
    input_path: Path,
    output_path: Path,
    *,
    scale: int = 4,
) -> None:
    """Without AI: increase scale× and light sharpness."""
    if scale < 2:
        scale = 2
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    img = img.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3)
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def download_ncnn_bundle(
    *,
    vendor_dir: Path | None = None,
    progress_hook: Callable[[int, int, int], object] | None = None,
) -> Path:
    """
    Download and unpack the official zip to vendor/. Returns the path to the executable file.
    progress_hook: optional callback(blocks, block_size, total_size) for urllib.
    """
    system = platform.system()
    url = _NCNN_ZIP_URLS.get(system)
    if not url:
        raise OSError(f"Auto-download NCNN for {system!r} not configured; download the archive manually.")

    dest_dir = vendor_dir if vendor_dir is not None else VENDOR_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "_realesrgan-ncnn-vulkan-download.zip"
    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    finally:
        if zip_path.is_file():
            zip_path.unlink(missing_ok=True)

    exe = find_ncnn_executable(dest_dir)
    if exe is None:
        raise RuntimeError(
            "Archive unpacked, but the executable file not found. "
            "Check the contents of the vendor/ folder."
        )
    _strip_ncnn_bundle_extras(ncnn_bundle_root(exe))
    _ensure_binary_executable(exe)
    return exe


def ncnn_available() -> bool:
    return find_ncnn_executable() is not None


def prepare_ncnn_input(src: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """
    Returns the path for NCNN and a temporary directory (if converted).
    If the second element is not None, call .cleanup() after processing.
    """
    suffix = src.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return src.resolve(), None
    td = tempfile.TemporaryDirectory()
    tmp = Path(td.name) / "_input_for_ncnn.png"
    Image.open(src).convert("RGB").save(tmp)
    return tmp, td