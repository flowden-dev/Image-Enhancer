"""One explicit paid request, no retries, and recovery before postprocessing."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import hashlib
import json
import logging
from pathlib import Path
import re
import time
import tempfile
import uuid
import warnings

import httpx
from PIL import Image

from . import __version__
from .core import (
    EnhancerError, MODEL, MAX_SOURCE_PIXELS, MAX_UPLOAD_BYTES, Plan, Settings, Texture,
    build_prompt, finish_image, make_canvas, plan_image, prepare_recovery,
    save_image_atomic, plan_local, plan_settings, needs_api, is_up_to_date,
)

from .annotations import finish_annotated_area
from .fidelity import protect_source
from .super_resolution import upscale_artwork
from .translation import finish_translation, make_area_mask, make_translation_mask
from .formats import EXTENSIONS, encoded_size, output_name, output_profile, preflight
from .formats import png as png_codec
from .formats.common import Profile, resize_artwork, restore_hidden_rgb
from .storage import save_original_atomic, write_atomic, default_data_directory, output_paths, check_free_space

ENDPOINT = "https://api.openai.com/v1/images/edits"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 40 * 1024 * 1024


@dataclass(frozen=True)
class ApiImage:
    image: Image.Image
    request_id: str
    usage: dict[str, int]


@dataclass(frozen=True)
class Result:
    image: Image.Image
    source_path: Path
    settings: Settings
    plan: Plan
    raw_path: Path | None
    saved_path: Path | None
    request_id: str
    usage: dict[str, int]
    warnings: tuple[str, ...]
    profile: Profile | None = None
    original_path: Path | None = None
    png_path: Path | None = None
    metadata_path: Path | None = None
    notices: tuple[str, ...] = ()


def redact(message: str, key: str = "") -> str:
    if key:
        message = message.replace(key, "[API key removed]")
    message = re.sub(r"\bsk-[A-Za-z0-9_\-]{4,}", "[API key removed]", message)
    return "".join(c for c in message if c in "\n\t" or c.isprintable())[:2000]


def validate_key(key: str) -> str:
    key = key.strip()
    if not key or len(key) > 512 or not key.isascii() or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in key):
        raise EnhancerError("Paste a valid API key without internal spaces or line breaks.")
    return key


def _api_error(status: int, body: bytes, request_id: str, key: str) -> EnhancerError:
    hints = {
        400: "The API rejected these settings or this image. Check the details below.",
        401: "The API key was not accepted. Check that you copied the complete key.",
        403: "This API project may lack model access or required organization verification.",
        404: "The model or endpoint is unavailable for this project.",
        413: "The API rejected the upload size. Try a smaller source.",
        429: "Rate limit or insufficient API credit. Check API usage and billing before retrying.",
    }
    hint = hints.get(status, "The API request failed. Check the service and your API usage before retrying.")
    detail = ""
    try:
        payload = json.loads(body)
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            value = payload["error"].get("message")
            if isinstance(value, str):
                detail = "\n\n" + redact(value, key)
    except (ValueError, UnicodeError, RecursionError):
        pass
    request_info = f"\nRequest ID: {request_id}" if request_id else ""
    return EnhancerError(f"HTTP {status}: {hint}{detail}{request_info}\n\nNo automatic retry was made.")


def decode_response(body: bytes, request_id: str = "") -> ApiImage:
    if len(body) > MAX_RESPONSE_BYTES:
        raise EnhancerError("API response exceeded the 64 MiB safety limit.")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EnhancerError("The API did not return valid JSON. No automatic retry was made.") from exc
    if not isinstance(payload, dict):
        raise EnhancerError("Unexpected API response structure.")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise EnhancerError("Expected exactly one image in the API response.")
    encoded = data[0].get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise EnhancerError("The API returned no base64 image. URLs are not downloaded automatically.")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise EnhancerError("Returned image exceeds the 40 MiB safety limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EnhancerError("The API returned invalid base64 image data.") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise EnhancerError("Returned image exceeds the 40 MiB safety limit.")
    try:
        png_codec.inspect(raw)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.format != "PNG" or getattr(opened, "n_frames", 1) != 1:
                    raise EnhancerError("The API did not return a single PNG image as requested.")
                if opened.width * opened.height > MAX_SOURCE_PIXELS or max(opened.size) > 16384:
                    raise EnhancerError("Returned image dimensions exceed the safety limit.")
                opened.load()
                image = opened.convert("RGBA")
                image.info.clear()
    except EnhancerError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise EnhancerError("The returned PNG is corrupt or cannot be decoded.") from exc
    usage: dict[str, int] = {}
    raw_usage = payload.get("usage")
    if isinstance(raw_usage, dict):
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = raw_usage.get(name)
            if type(value) is int and value >= 0:
                usage[name] = value
    return ApiImage(image, request_id, usage)


def request_image(key: str, png: bytes, plan: Plan, settings: Settings, has_alpha: bool,
                  *, transport: httpx.BaseTransport | None = None) -> ApiImage:
    key = validate_key(key)
    settings.validate()
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) > MAX_UPLOAD_BYTES:
        raise EnhancerError("The prepared upload must be a PNG under the 40 MiB safety limit.")
    fields = {
        "model": MODEL,
        "prompt": build_prompt(plan, settings),
        "size": plan.size_string,
        "quality": settings.quality,
        "complexity": "high",
        "n": "1",
        "output_format": "png",
        "background": "transparent" if has_alpha else "opaque",
    }
    files = {"image[]": ("source.png", png, "image/png")}
    area = settings.annotation.canvas_area(plan) if settings.annotation else None
    mask = make_area_mask(plan.canvas_size, (area,)) if area else make_translation_mask(plan, settings)
    if mask is not None:
        if len(mask) >= 4 * 1024 * 1024:
            raise EnhancerError("The edit mask exceeds the upload size limit.")
        files["mask"] = ("text-areas.png", mask, "image/png")
    # no SDK retry defaults, environment base URLs, redirects or ambient HTTP proxies
    timeout = httpx.Timeout(connect=20.0, read=900.0, write=120.0, pool=20.0)
    # request bodies/headers must never be sent to debug logs by this application
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False,
                          transport=transport) as client:
            with client.stream("POST", ENDPOINT,
                               headers={"Authorization": f"Bearer {key}", "Accept-Encoding": "identity",
                                        "User-Agent": f"Image-Enhancer/{__version__}"},
                               data=fields, files=files) as response:
                request_id = redact(response.headers.get("x-request-id", ""), key)[:200]
                # reject compressed HTTP before httpx can allocate an unbounded decoded chunk
                if response.headers.get("content-encoding", "identity").strip().lower() not in ("", "identity"):
                    raise EnhancerError("Unexpected compressed API response. No automatic retry was made; check usage before retrying.")
                chunks = bytearray()
                limit = MAX_RESPONSE_BYTES if response.status_code == 200 else 256 * 1024
                for chunk in response.iter_bytes(chunk_size=65536):
                    if len(chunks) + len(chunk) > limit:
                        if response.status_code != 200:
                            raise _api_error(response.status_code, b"", request_id, key)
                        raise EnhancerError("API response exceeded the safety limit. No automatic retry was made.")
                    chunks.extend(chunk)
                if response.status_code != 200:
                    raise _api_error(response.status_code, bytes(chunks), request_id, key)
                return decode_response(bytes(chunks), request_id)
    except httpx.TimeoutException as exc:
        raise EnhancerError("The API request timed out. It may still have been processed and charged. "
                            "Check your API usage before trying again; this tool does not retry automatically.") from exc
    except httpx.HTTPError as exc:
        raise EnhancerError("The API connection failed or was interrupted. Check your connection and API usage "
                            "before trying again. No automatic retry was made.") from exc


def default_recovery_directory() -> Path:
    # legacy name retained for callers; this is the user-facing output folder
    return default_data_directory() / "outputs"


def private_recovery_directory() -> Path:
    return default_data_directory() / "recovery"


def run_job(texture: Texture, settings: Settings, key: str, recovery_directory: Path,
            *, transport: httpx.BaseTransport | None = None, native_destination: Path | None = None,
            work_directory: Path | None = None) -> Result:
    started = time.monotonic()
    settings.validate()
    if settings.annotation:
        settings.annotation.check_source(texture.sha256)
    plan = plan_settings(texture.image.size, settings)
    unchanged = is_up_to_date(settings, plan)
    use_api = needs_api(settings, plan)
    key = validate_key(key) if use_api else ""
    if texture.profile is None:
        raise EnhancerError("The source has no compatibility profile. Reopen the original texture before processing.")
    # exercise every native encoder before a paid request, including reverse ICC conversion
    profile = output_profile(texture.profile, settings.output_format)
    copy_source = unchanged and profile is texture.profile
    preflight(texture.profile, texture.image, method="nearest" if unchanged else settings.method, output_format=settings.output_format)
    output_directory = prepare_recovery(recovery_directory)
    work_root = prepare_recovery(work_directory or private_recovery_directory())
    owned_work = work_directory is None
    work = Path(tempfile.mkdtemp(prefix="image-", dir=work_root)) if owned_work else work_root
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
        base = work / stamp
        check_free_space(output_directory, plan.target_size[0] * plan.target_size[1] * 6,
                         work, plan.target_size[0] * plan.target_size[1] * 12 +
                         plan.canvas_size[0] * plan.canvas_size[1] * 8 + 8 * 1024 * 1024)
        native_destination = (output_paths(output_directory, (output_name(texture.path, settings.output_format),))[0] if native_destination is None
                              else Path(native_destination))
        if EXTENSIONS.get(native_destination.suffix.lower()) != profile.format:
            raise EnhancerError("The destination extension does not match the selected output format.")
        # probe without ever deleting an existing destination
        created = False
        try:
            with native_destination.open("xb"):
                created = True
            native_destination.unlink()
        except OSError as exc:
            if created:
                native_destination.unlink(missing_ok=True)
            raise EnhancerError("Cannot create the native output filename. Check the output path length and permissions; no request was sent.") from exc
        raw_path = saved_path = original_path = png_path = None
        notes: list[str] = []
        notices: list[str] = []
        request_id = ""
        usage: dict[str, int] = {}
        if use_api:
            detail = settings.task == "enhance" and (settings.source_protection == "detail" or settings.annotation is not None)
            guide = None
            if detail:
                width = min(plan.target_size[0], plan.crop[2] - plan.crop[0])
                height = min(plan.target_size[1], plan.crop[3] - plan.crop[1])
                if width >= texture.image.width and height >= texture.image.height and (width, height) != texture.image.size:
                    guide = upscale_artwork(texture.image, (width, height))
            upload, has_alpha = make_canvas(texture.image, plan, artwork=guide)
            api_result = request_image(key, upload, plan, settings, has_alpha, transport=transport)
            request_id, usage = api_result.request_id, api_result.usage
            try:
                raw_path = save_image_atomic(api_result.image, base.with_name(base.name + "_api-canvas.png"))
            except (OSError, EnhancerError, MemoryError) as exc:
                notes.append("Automatic canvas recovery failed: " + redact(str(exc), key))
            try:
                if settings.task == "translate":
                    result = finish_translation(api_result.image, texture.image, plan, settings)
                elif settings.annotation and settings.annotation.area:
                    result = finish_annotated_area(api_result.image, texture.image, plan, settings.annotation.area, guide)
                    result = restore_hidden_rgb(result, texture.image)
                else:
                    result = finish_image(api_result.image, texture.image, plan, settings.preserve_alpha,
                                          detail=detail, guide=guide)
                    if not settings.annotation:
                        result = protect_source(result, texture.image, settings.source_protection)
                    if settings.preserve_alpha:
                        result = restore_hidden_rgb(result, texture.image)
            except (EnhancerError, OSError, ValueError, MemoryError) as exc:
                recovery_note = (f"\n\nThe unmodified API canvas is saved at:\n{raw_path}" if raw_path else
                                 "\n\nThe API canvas could not be saved: " + "; ".join(notes))
                raise EnhancerError((str(exc) or "Image postprocessing failed.") + recovery_note) from exc
            if plan.resampled_output:
                notices.append("The requested dimensions exceed the API canvas. The edited canvas was enlarged locally to the exact target; this is not native full-resolution AI generation.")
        elif unchanged:
            result = texture.image.copy()
            notices.append("Already at or above the requested size; copied without an API request." if copy_source else
                           "Already at the requested size. Converted locally to " + profile.format + ".")
        elif settings.method == "neural":
            result = upscale_artwork(texture.image, plan.target_size)
        else:
            method = Image.Resampling.LANCZOS if settings.method == "lanczos" else Image.Resampling.NEAREST
            # channel/data textures are resized per channel, not as opacity-weighted artwork
            if texture.profile.data_texture:
                result = Image.merge("RGBA", tuple(channel.resize(plan.target_size, method) for channel in texture.image.split()))
            else:
                result = resize_artwork(texture.image, plan.target_size, method)
            result.info.clear()
        if result.size != plan.target_size:
            raise EnhancerError(f"Produced {result.width} x {result.height} pixels, expected "
                                f"{plan.target_size[0]} x {plan.target_size[1]}. No native file was written."
                                + (f" API canvas saved at {raw_path}." if raw_path else ""))
        if use_api or settings.keep_intermediates:
            try:
                png_path = save_image_atomic(result, base.with_name(base.name + "_enhanced.png"))
            except (OSError, EnhancerError, MemoryError) as exc:
                notes.append("Automatic PNG recovery failed. Keep this window open and export manually. " + redact(str(exc), key))
        try:
            if copy_source:
                data = texture.path.read_bytes()
                if hashlib.sha256(data).hexdigest() != texture.sha256:
                    raise EnhancerError("Source changed after loading; reopen it before processing.")
                original_path = write_atomic(data, native_destination, texture.path)
            else:
                original_path = save_original_atomic(result, profile, native_destination, texture.path)
        except Exception as exc:
            # keep the paid result available even when an encoder, disk or verification fails
            label = "Original-format" if settings.output_format == "original" else profile.format
            notes.append(f"{label} export failed. The result remains available in memory and in the recovery PNG when saved. " + redact(str(exc), key))
            if png_path is None and not use_api and not settings.keep_intermediates:
                try:
                    png_path = save_image_atomic(result, base.with_name(base.name + "_enhanced.png"))
                except (OSError, EnhancerError, MemoryError) as recovery_error:
                    notes.append("Recovery PNG could not be saved: " + redact(str(recovery_error), key))
        if original_path and not notes and not settings.keep_intermediates:
            for intermediate in (raw_path, png_path):
                if intermediate:
                    try:
                        intermediate.unlink()
                    except OSError as exc:
                        notes.append("Temporary recovery file could not be removed: " + redact(str(exc), key))
            raw_path = raw_path if raw_path and raw_path.exists() else None
            png_path = png_path if png_path and png_path.exists() else None
        saved_path = original_path or png_path
        metadata = {
            "app_version": __version__, "source_name": redact(texture.path.name, key),
            "source_sha256": texture.sha256, "model": MODEL if use_api else None,
            "method": ("copy" if copy_source else "convert") if unchanged else settings.method, "scale": settings.scale if settings.target_edge is None else None,
            "target_longest_edge": settings.target_edge,
            "quality": settings.quality if use_api else None, "complexity": "high" if use_api else None,
            "preserve_alpha": settings.preserve_alpha if settings.task == "enhance" else "automatic-text-areas",
            "source_protection": settings.source_protection if use_api and settings.task == "enhance" and not settings.annotation else None,
            "annotated": settings.annotation is not None,
            "task": settings.task,
            "translation_scope": settings.translation_scope if settings.task == "translate" else None,
            "text_area_count": len(settings.text_regions) if settings.task == "translate" else 0,
            "protected_area_count": len(settings.protected_regions) if settings.task == "translate" else 0,
            "request_source_crop": plan.source_crop,
            "source_language": redact(settings.source_language, key) if settings.task == "translate" else None,
            "target_language": redact(settings.target_language, key) if settings.task == "translate" else None,
            "processing_seconds": round(time.monotonic() - started, 3),
            "source_size": plan.source_size, "canvas_size": plan.canvas_size if use_api else None,
            "crop": plan.crop if use_api else None, "output_size": encoded_size(profile, plan.target_size),
            "output_format": profile.format,
            "local_final_enlargement": use_api and plan.resampled_output,
            "original_format": texture.profile.description,
            "request_id": request_id, "usage": usage,
            "raw_saved": raw_path is not None, "png_saved": png_path is not None,
            "result_saved": saved_path is not None, "original_saved": original_path is not None,
        }
        metadata_path = None
        if settings.keep_intermediates or notes:
            try:
                metadata_path = write_atomic(json.dumps(metadata, indent=2).encode("utf-8"), base.with_suffix(".json"))
            except (OSError, EnhancerError, MemoryError) as exc:
                notes.append("Metadata could not be saved: " + redact(str(exc), key))
        return Result(result, texture.path, settings, plan, raw_path, saved_path,
                      request_id, usage, tuple(notes), texture.profile, original_path, png_path, metadata_path, tuple(notices))
    finally:
        if owned_work:
            try:
                work.rmdir()
            except OSError:
                pass
