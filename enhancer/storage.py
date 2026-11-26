"""Non-destructive, atomic publication shared by all original-format exporters."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unicodedata

from .errors import EnhancerError
from .formats import EXTENSIONS, encode_original
from .formats.common import Profile


def default_data_directory() -> Path:
    override = os.environ.get("IMAGE_ENHANCER_DATA_DIR", "")
    if override and Path(override).is_absolute():
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        if not base.is_absolute():
            base = Path.home() / "AppData" / "Local"
        return base / "ImageEnhancer"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ImageEnhancer"
    base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    if not base.is_absolute():
        base = Path.home() / ".local" / "share"
    return base / "image-enhancer"


def output_paths(directory: Path, names) -> tuple[Path, ...]:
    """Flat outputs with stable numbered copies, including case-insensitive collisions."""
    def identity(name):
        return unicodedata.normalize("NFC", name).casefold()
    occupied = {identity(path.name) for path in directory.iterdir()}
    counters = {}
    result = []
    for name in names:
        if not name or Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name:
            raise EnhancerError("Unsafe output filename. No request was sent.")
        path = Path(name)
        key, candidate = identity(name), name
        number = counters.get(key, 2)
        while identity(candidate) in occupied:
            candidate = f"{path.stem} ({number}){path.suffix}"
            number += 1
        counters[key] = number
        occupied.add(identity(candidate))
        result.append(directory / candidate)
    return tuple(result)


def check_free_space(output: Path, output_bytes: int, recovery: Path, recovery_bytes: int) -> None:
    """Budget both drives, or their combined requirement when they share a device."""
    if output.stat().st_dev == recovery.stat().st_dev:
        insufficient = shutil.disk_usage(output).free < output_bytes + recovery_bytes
    else:
        insufficient = (shutil.disk_usage(output).free < output_bytes or
                        shutil.disk_usage(recovery).free < recovery_bytes)
    if insufficient:
        raise EnhancerError("The output or recovery drive has insufficient free space. Choose fewer files or another output drive. No request was sent.")


def paths_match(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    try:
        return first.samefile(second)
    except OSError:
        return False


def write_atomic(data: bytes, destination: Path | str, source_path: Path | None = None, *, overwrite: bool = False) -> Path:
    destination = Path(destination).expanduser().absolute()
    if source_path is not None and paths_match(destination, source_path):
        raise EnhancerError("The original source cannot be overwritten. Choose a different filename or output folder.")
    if destination.is_symlink():
        raise EnhancerError("Saving through a symbolic link is not allowed.")
    if destination.exists() and not overwrite:
        raise EnhancerError("That output already exists. Choose another name or confirm replacement.")
    if not destination.parent.is_dir():
        raise EnhancerError("The destination folder does not exist.")
    fd, name = tempfile.mkstemp(prefix=".image-enhancer-", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if source_path is not None and paths_match(destination, source_path):
            raise EnhancerError("The source cannot be overwritten.")
        if destination.is_symlink():
            raise EnhancerError("The destination became a symbolic link; saving was stopped.")
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise EnhancerError("An output with that name already exists; it was not overwritten.") from exc
            except OSError:
                if os.name != "nt":
                    raise
                os.rename(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_original_atomic(image, profile: Profile, destination: Path | str, source_path: Path | None = None, *, overwrite: bool = False) -> Path:
    destination = Path(destination)
    if EXTENSIONS.get(destination.suffix.lower()) != profile.format:
        raise EnhancerError(f"Original-format export must use {profile.extension}. Use Export PNG for an intermediate image.")
    if source_path is not None and paths_match(destination, source_path):
        raise EnhancerError("The original source cannot be overwritten.")
    # validation and encoding finish before an existing output is touched
    return write_atomic(encode_original(image, profile), destination, source_path, overwrite=overwrite)
