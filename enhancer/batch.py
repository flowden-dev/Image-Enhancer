"""Bounded, sequential jobs with preflight, cancellation and no implicit retries."""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import tempfile
from typing import Callable, Iterable, Mapping
import uuid

from .annotations import ImageAnnotation
from .api import Result, redact, run_job, private_recovery_directory
from .core import EnhancerError, Plan, Settings, Texture, is_up_to_date, load_texture, make_canvas, plan_settings, prepare_recovery, needs_api, save_image_atomic
from .formats import EXTENSIONS, encoded_size, output_name, output_profile, preflight
from .formats.common import MAX_OUTPUT_FILE_BYTES, open_output_image
from .formats import png as png_codec
from .storage import write_atomic, output_paths, check_free_space, save_original_atomic

MAX_BATCH_FILES = 10000
MAX_SCAN_ENTRIES = 200000


def is_link(path: Path) -> bool:
    # Windows directory junctions are reparse points, but are not always symlinks
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


@dataclass(frozen=True)
class BatchItem:
    index: int
    path: Path
    relative_path: Path
    sha256: str = ""
    plan: Plan | None = None
    description: str = ""
    error: str = ""
    pixel_sha256: str = ""
    annotation: ImageAnnotation | None = None
    output_format: str = "original"
    output_size: tuple[int, int] | None = None

    def needs_request(self, settings: Settings) -> bool:
        return bool(not self.error and self.plan and needs_api(settings.for_image(self.annotation), self.plan))

    @property
    def reuse_key(self) -> str:
        pixels = self.pixel_sha256 or self.sha256
        return self.annotation.reuse_key(pixels) if self.annotation else pixels


@dataclass(frozen=True)
class SavedExport:
    path: Path
    source_sha256: str
    settings: Settings
    plan: Plan
    size: tuple[int, int]


@dataclass(frozen=True)
class BatchEvent:
    index: int
    state: str
    message: str = ""
    output: Path | None = None
    dimensions: str = ""
    source_dimensions: str = ""
    saved: SavedExport | None = None


@dataclass(frozen=True)
class BatchSummary:
    completed: int
    failed: int
    cancelled: bool
    directory: Path
    last_result: Result | None = None
    last_texture: Texture | None = None
    warning: str = ""
    recovery_directory: Path | None = None


@dataclass(frozen=True)
class CachedOutput:
    native: Path
    png: Path
    raw: Path | None
    native_hash: str
    png_hash: str
    index: int
    source_sha256: str


def _pixel_digest(texture: Texture) -> str:
    # Decoders supply RGBA in working-image coordinates. Hidden RGB and data-channel semantics matter.
    image = texture.image
    digest = hashlib.sha256(struct.pack(">II?", image.width, image.height, texture.profile.data_texture))
    for top in range(0, image.height, 128):
        digest.update(image.crop((0, top, image.width, min(top + 128, image.height))).tobytes())
    return digest.hexdigest()


def request_count(items, settings: Settings) -> int:
    valid = [item for item in items if item.needs_request(settings)]
    return len({item.reuse_key for item in valid}) if settings.reuse_identical else len(valid)


def _verified_cache_bytes(path: Path, digest: str) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(MAX_OUTPUT_FILE_BYTES + 1)
    if len(data) > MAX_OUTPUT_FILE_BYTES or hashlib.sha256(data).hexdigest() != digest:
        raise EnhancerError("A saved result used for identical-image reuse changed. No extra API request was made.")
    return data


def restore_cached(entry: CachedOutput, texture: Texture, plan: Plan, settings: Settings, destination: Path) -> Result:
    png = _verified_cache_bytes(entry.png, entry.png_hash)
    info = png_codec.inspect(png, output=True)
    if (info.width, info.height) != plan.target_size:
        raise EnhancerError("The reused image has unexpected dimensions. No extra API request was made.")
    with open_output_image(png, plan.target_size) as opened:
        opened.load()
        image = opened.convert("RGBA")
        image.info.clear()
    del png
    if entry.source_sha256 == texture.sha256:
        # The original job verified this native export. Verify its hash immediately before copying.
        native = _verified_cache_bytes(entry.native, entry.native_hash)
        write_atomic(native, destination, texture.path)
    else:
        # Pixel-identical sources may have different TGA headers, palettes, profiles or formats.
        save_original_atomic(image, output_profile(texture.profile, settings.output_format), destination, texture.path)
    return Result(image, texture.path, settings, plan, entry.raw, destination, "", {}, (),
                  texture.profile, destination, None)


def collect_files(inputs: Iterable[Path | str], existing: Iterable[Path] = (), *,
                  output_directory: Path | None = None) -> tuple[list[Path], list[str]]:
    """Walk folders deterministically, without following links or loading pixels."""
    files: list[Path] = []
    notes: list[str] = []
    existing = tuple(existing)
    seen = {path.resolve() for path in existing}
    scanned = 0
    selected = tuple(Path(p).expanduser().absolute() for p in inputs)
    destination = output_directory.resolve() if output_directory is not None else None
    selected_paths = set(selected)
    pending = list(reversed(selected))
    while pending:
        path = pending.pop()
        scanned += 1
        if scanned > MAX_SCAN_ENTRIES:
            notes.append(f"Stopped scanning at {MAX_SCAN_ENTRIES} entries. Add smaller folders.")
            break
        try:
            if is_link(path):
                continue
            if path.is_dir():
                if destination is not None and path not in selected_paths and path.resolve() == destination:
                    continue
                # bound allocation as well as traversal; no whole-tree list in memory
                children = []
                with os.scandir(path) as entries:
                    for entry in entries:
                        if len(children) + scanned + len(pending) >= MAX_SCAN_ENTRIES:
                            notes.append("Folder scan limit reached. Add smaller folders.")
                            break
                        if not entry.is_symlink():
                            children.append(Path(entry.path))
                pending.extend(sorted(children, key=lambda p: str(p).casefold(), reverse=True))
                continue
            if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                continue
            path = path.resolve(strict=True)
            if path in seen:
                continue
            if len(files) + len(existing) >= MAX_BATCH_FILES:
                notes.append(f"The queue is limited to {MAX_BATCH_FILES} files.")
                break
            seen.add(path)
            files.append(path)
        except OSError as exc:
            notes.append(f"Cannot read {path.name}: {exc}")
    return files, notes


def prepare_batch(paths: tuple[Path, ...], settings: Settings, stop: threading.Event,
                  emit: Callable[[BatchEvent], None] = lambda event: None, *,
                  annotations: Mapping[Path, ImageAnnotation] | None = None) -> tuple[BatchItem, ...]:
    settings.validate()
    if settings.annotation is not None:
        raise EnhancerError("Assign annotations to individual queue files.")
    annotations = dict(annotations or {}) if settings.task == "enhance" else {}
    if settings.task == "translate" and (settings.translation_scope == "regions" or settings.protected_regions):
        raise EnhancerError("Selected text areas belong to one image. Use Translate Image for one source, or choose All text without areas for a batch.")
    if not 1 <= len(paths) <= MAX_BATCH_FILES:
        raise EnhancerError(f"Choose between 1 and {MAX_BATCH_FILES} files.")
    try:
        base = Path(os.path.commonpath([str(p.parent) for p in paths]))
    except ValueError:
        base = None
    items = []
    pixel_digests: dict[str, str] = {}
    checked_profiles: set[tuple[str, str]] = set()
    checked_canvases: set[tuple[str, Plan]] = set()
    for index, path in enumerate(paths):
        if stop.is_set():
            break
        relative = path.relative_to(base) if base is not None else Path(f"{index + 1:04d}") / path.name
        item = BatchItem(index, path, relative, annotation=annotations.get(path), output_format=settings.output_format)
        emit(BatchEvent(index, "Checking"))
        try:
            texture = load_texture(path)
            try:
                current = settings.for_image(item.annotation)
                plan = plan_settings(texture.image.size, current)
                if item.annotation:
                    item.annotation.check_source(texture.sha256)
                unchanged = is_up_to_date(current, plan)
                profile_key = (texture.sha256, "nearest" if unchanged else current.method)
                if profile_key not in checked_profiles:
                    preflight(texture.profile, texture.image, method=profile_key[1], output_format=current.output_format)
                    checked_profiles.add(profile_key)
                pixel_digest = texture.sha256
                if current.reuse_identical and not unchanged:
                    if texture.sha256 not in pixel_digests:
                        pixel_digests[texture.sha256] = _pixel_digest(texture)
                    pixel_digest = pixel_digests[texture.sha256]
                canvas_key = (pixel_digest, plan)
                if needs_api(current, plan) and canvas_key not in checked_canvases:
                    make_canvas(texture.image, plan)
                    checked_canvases.add(canvas_key)
                profile = output_profile(texture.profile, current.output_format)
                size = encoded_size(profile, plan.target_size)
                item = replace(item, sha256=texture.sha256, pixel_sha256=pixel_digest, output_size=size,
                               plan=plan, description=texture.profile.description)
            finally:
                texture.image.close()
            emit(BatchEvent(index, "Ready", item.description,
                            dimensions=f"{size[0]} x {size[1]}",
                            source_dimensions=f"{plan.source_size[0]} x {plan.source_size[1]}"))
        except Exception as exc:
            item = replace(item, error=str(exc) or type(exc).__name__)
            emit(BatchEvent(index, "Invalid", item.error))
        items.append(item)
    return tuple(items)


def run_batch(items: tuple[BatchItem, ...], settings: Settings, key: str, directory: Path,
              stop: threading.Event, emit: Callable[[BatchEvent], None] = lambda event: None,
              *, runner: Callable = run_job) -> BatchSummary:
    """Run only preflighted items. Any processing/export failure stops further spending."""
    settings.validate()
    if settings.annotation is not None or (settings.task != "enhance" and any(i.annotation for i in items)):
        raise EnhancerError("Assign artwork annotations per file and preflight this queue again.")
    if settings.task == "translate" and (settings.translation_scope == "regions" or settings.protected_regions):
        raise EnhancerError("Selected text areas belong to one image. Use Translate Image for one source, or choose All text without areas for a batch.")
    valid = tuple(item for item in items if not item.error and item.plan is not None)
    if not valid:
        raise EnhancerError("No valid files remain in the queue.")
    if any(item.output_format != settings.output_format or
           plan_settings(item.plan.source_size, settings.for_image(item.annotation)) != item.plan for item in valid):
        raise EnhancerError("The output settings changed after preflight. Check the queue again before processing.")
    directory = prepare_recovery(directory)
    work_root = prepare_recovery(private_recovery_directory())
    output_bytes = sum(i.plan.target_size[0] * i.plan.target_size[1] * 6 for i in valid)
    recovery_sizes = (i.plan.target_size[0] * i.plan.target_size[1] * 12 +
                      i.plan.canvas_size[0] * i.plan.canvas_size[1] * 8 for i in valid)
    recovery_bytes = (sum(recovery_sizes) if settings.keep_intermediates else max(recovery_sizes)) + 8 * 1024 * 1024
    duplicate_counts = Counter(item.reuse_key for item in valid)
    reusable = {item.reuse_key: item for item in valid if settings.reuse_identical and
                not is_up_to_date(settings.for_image(item.annotation), item.plan) and duplicate_counts[item.reuse_key] > 1}
    recovery_bytes += sum(item.plan.target_size[0] * item.plan.target_size[1] * 5 for item in reusable.values())
    check_free_space(directory, output_bytes, work_root, recovery_bytes)
    for item in valid:
        relative = item.relative_path
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise EnhancerError("Unsafe batch output path. No request was sent.")
    targets = output_paths(directory, (output_name(item.path, settings.output_format) for item in valid))
    destinations = {item.index: target for item, target in zip(valid, targets)}
    reserved = []
    try:
        for target in targets:
            with target.open("xb"):
                reserved.append(target)
    finally:
        for target in reserved:
            target.unlink(missing_ok=True)
    root = Path(tempfile.mkdtemp(prefix="batch-", dir=work_root))
    recovery = root
    records = [{"index": i.index, "source_name": redact(i.path.name, key), "source_sha256": i.sha256,
                "output": redact(destinations[i.index].name, key), "target": i.output_size,
                "method": settings.for_image(i.annotation).method, "annotated": bool(i.annotation),
                "status": "pending"} for i in valid]
    manifest = root / "batch.json"
    completed = failed = 0
    planned_requests = request_count(valid, settings)
    cache: dict[str, CachedOutput] = {}
    last_result = last_texture = None
    warning = ""

    def save_manifest():
        # never persist key, prompt, error response bodies or absolute source paths
        body = {"method": settings.method, "quality": settings.quality, "output_format": settings.output_format,
                "target_edge": settings.target_edge, "scale": settings.scale if settings.target_edge is None else None,
                "source_protection": settings.source_protection if settings.method == "openai" and settings.task == "enhance" else None,
                "task": settings.task, "reuse_identical": settings.reuse_identical,
                "planned_api_requests": planned_requests,
                "source_language": redact(settings.source_language, key) if settings.task == "translate" else None,
                "target_language": redact(settings.target_language, key) if settings.task == "translate" else None,
                "items": records}
        write_atomic(json.dumps(body, indent=2).encode("utf-8"), manifest, overwrite=True)

    try:
        save_manifest()
    except Exception:
        shutil.rmtree(root)
        raise
    reuse_folder = tempfile.TemporaryDirectory(prefix=".reuse-", dir=root)
    try:
        for position, item in enumerate(valid):
            if stop.is_set():
                break
            # all preceding results are safely on disk; release full-size pixels between jobs
            if last_result is not None:
                last_result.image.close()
                last_texture.image.close()
                last_result = last_texture = None
            texture = None
            result = None
            try:
                emit(BatchEvent(item.index, "Processing"))
                texture = load_texture(item.path)
                if texture.sha256 != item.sha256:
                    raise EnhancerError("Source changed after preflight. Re-add this file to review it before processing.")
                if stop.is_set():
                    texture.image.close()
                    emit(BatchEvent(item.index, "Ready", "Stopped before processing"))
                    break
                records[position]["status"] = "processing"
                save_manifest()
                group = item.reuse_key
                destination = destinations[item.index]
                current = settings.for_image(item.annotation)
                job_key = key if needs_api(current, item.plan) else ""
                cached = cache.get(group)
                if cached is not None:
                    result = restore_cached(cached, texture, item.plan, current, destination)
                    records[position]["reused_from_index"] = cached.index
                else:
                    result = (runner(texture, current, job_key, directory, native_destination=destinations[item.index], work_directory=recovery)
                              if runner is run_job else runner(texture, current, job_key, recovery))
                last_result, last_texture = result, texture
                if result.original_path is None:
                    raise EnhancerError("Native export failed. Processing stopped; inspect the saved recovery PNG or export the retained result manually.")
                if result.original_path != destination:
                    write_atomic(result.original_path.read_bytes(), destination, texture.path)
                if cached is None and group in reusable and not result.warnings:
                    cache_png = Path(reuse_folder.name) / (group + "_enhanced.png")
                    save_image_atomic(result.image, cache_png)
                    with destination.open("rb") as native, cache_png.open("rb") as png:
                        cache[group] = CachedOutput(destination, cache_png, result.raw_path,
                                                    hashlib.file_digest(native, "sha256").hexdigest(),
                                                    hashlib.file_digest(png, "sha256").hexdigest(), item.index, item.sha256)
                if not settings.keep_intermediates and not result.warnings:
                    # remove only job-owned staging files after native output is safely published
                    for path in (result.original_path if result.original_path != destination else None, result.metadata_path):
                        if path and path.resolve().is_relative_to(recovery.resolve()):
                            path.unlink(missing_ok=True)
                            if path.parent != recovery:
                                try:
                                    path.parent.rmdir()
                                except OSError:
                                    pass
                duplicate_counts[group] -= 1
                if duplicate_counts[group] == 0 and group in cache:
                    cache.pop(group).png.unlink(missing_ok=True)
                result = replace(result, original_path=destination, saved_path=destination,
                                 png_path=None if cached else result.png_path,
                                 metadata_path=None if not settings.keep_intermediates else result.metadata_path)
                last_result = result
                completed += 1
                records[position]["status"] = "done"
                records[position]["usage"] = result.usage
                records[position]["request_id"] = result.request_id
                records[position]["notices"] = result.notices
                saved = SavedExport(destination, texture.sha256, current, item.plan, item.output_size or item.plan.target_size)
                emit(BatchEvent(item.index, "Done", "Reused identical image; no API request." if cached else "\n".join(result.notices + result.warnings),
                                destination, saved=saved))
                try:
                    save_manifest()
                except Exception:
                    warning = "The image was saved, but the batch summary could not be updated. Processing stopped; inspect the output folder."
                    break
                # recovery failures warrant attention before spending on more images
                if result.warnings:
                    warning = "Processing stopped after an output warning. Review the last result before continuing."
                    break
            except Exception as exc:
                failed += 1
                warning = redact(str(exc) or type(exc).__name__, key)
                records[position]["status"] = "failed"
                emit(BatchEvent(item.index, "Failed", warning,
                                result.original_path or result.png_path if result is not None else None))
                if texture is not None and result is None:
                    texture.image.close()
                try:
                    save_manifest()
                except Exception:
                    warning += " The batch summary could not be updated; inspect the output folders."
                break
    finally:
        try:
            reuse_folder.cleanup()
        except OSError:
            warning = warning or "Outputs were saved, but temporary reuse files could not be removed. Check the output folder permissions."
        if not warning and not failed and not settings.keep_intermediates:
            try:
                shutil.rmtree(root)
            except OSError:
                warning = "Outputs were saved, but private recovery files could not be removed. Check the recovery folder permissions."
    retained = root if root.exists() else None
    return BatchSummary(completed, failed, stop.is_set(), directory, last_result, last_texture, warning, retained)
