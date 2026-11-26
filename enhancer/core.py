"""Validated texture loading, canvas planning and non-destructive output."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import struct
import tempfile
from dataclasses import dataclass, field, replace
from decimal import Decimal

from PIL import Image

from .annotations import ImageAnnotation
from .errors import EnhancerError
from .formats.common import MAX_EDGE as MAX_OUTPUT_EDGE, MAX_OUTPUT_PIXELS, Profile
from .formats import OUTPUT_SUFFIXES, decode_source
from .storage import write_atomic, paths_match

MODEL = "gpt-image-2.6-sunburst"
QUALITIES = ("low", "medium", "high", "xhigh", "max")
MAX_EDGE = 3840
MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
MAX_SOURCE_PIXELS = 16_777_216
MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_INSTRUCTION = 4000
SUPPORTED_FORMATS = {"DDS", "TGA", "PNG", "JPEG"}
METHODS = ("openai", "neural", "lanczos", "nearest")


@dataclass(frozen=True)
class Texture:
    path: Path
    image: Image.Image
    format: str
    sha256: str
    notes: tuple[str, ...] = ()
    profile: Profile | None = None


@dataclass(frozen=True)
class Settings:
    scale: int | float = 2
    quality: str = "max"
    instruction: str = ""
    preserve_alpha: bool = True
    method: str = "openai"
    target_edge: int | None = None
    source_protection: str = "detail"
    task: str = "enhance"
    source_language: str = "Auto-detect"
    target_language: str = "German"
    translation_scope: str = "regions"
    text_regions: tuple[tuple[int, int, int, int], ...] = ()
    replacements: str = ""
    protected_terms: str = ""
    keep_names: bool = True
    protected_regions: tuple[tuple[int, int, int, int], ...] = ()
    optimize_translation: bool = True
    reuse_identical: bool = True
    keep_intermediates: bool = False
    skip_up_to_date: bool = True
    annotation: ImageAnnotation | None = field(default=None, repr=False)
    output_format: str = "original"

    def for_image(self, annotation: ImageAnnotation | None) -> Settings:
        if annotation is None or self.task != "enhance":
            return self
        if self.method == "openai":
            return replace(self, annotation=annotation)
        # Local methods hide the global AI controls. Only this row opts into a model request.
        return replace(self, annotation=annotation, method="openai", instruction="",
                       preserve_alpha=True, source_protection="detail")

    def validate(self) -> None:
        if not isinstance(self.output_format, str) or (self.output_format != "original" and self.output_format not in OUTPUT_SUFFIXES):
            raise EnhancerError("Choose an available output format.")
        if self.annotation is not None:
            if not isinstance(self.annotation, ImageAnnotation) or self.task != "enhance" or self.method != "openai":
                raise EnhancerError("Per-image annotations require OpenAI artwork enhancement.")
            self.annotation.validate()
        if self.method not in METHODS:
            raise EnhancerError("Choose an available upscaling method.")
        if type(self.scale) not in (int, float) or not 1 <= self.scale <= MAX_OUTPUT_EDGE:
            raise EnhancerError(f"Enter a scale factor between 1 and {MAX_OUTPUT_EDGE}, such as 1.5, 10 or 12.")
        if self.target_edge is not None and (type(self.target_edge) is not int or not 1 <= self.target_edge <= MAX_OUTPUT_EDGE):
            raise EnhancerError(f"The target longest edge must be between 1 and {MAX_OUTPUT_EDGE} pixels.")
        if self.quality not in QUALITIES:
            raise EnhancerError("Choose a supported quality setting.")
        if not isinstance(self.instruction, str) or len(self.instruction) > MAX_INSTRUCTION:
            raise EnhancerError(f"Additional instructions must be at most {MAX_INSTRUCTION} characters.")
        if any(ord(c) < 32 and c not in "\n\t" for c in self.instruction):
            raise EnhancerError("Additional instructions contain unsupported control characters.")
        try:
            self.instruction.encode("utf-8")
        except UnicodeError as exc:
            raise EnhancerError("Additional instructions contain invalid Unicode characters.") from exc
        if self.source_protection not in ("detail", "strict", "balanced", "off"):
            raise EnhancerError("Choose Detail recovery, Strong, Moderate or Off for source protection.")
        if self.task == "enhance" and self.method == "openai" and self.instruction.strip() and self.source_protection != "off":
            raise EnhancerError("Additional edits change the artwork. Set Source protection to Off to use an additional edit, or clear that field to preserve the original.")
        if self.task not in ("enhance", "translate"):
            raise EnhancerError("Choose Enhance artwork or Translate text.")
        if self.task == "translate":
            from .translation import validate_translation
            validate_translation(self)
        if type(self.keep_intermediates) is not bool or type(self.skip_up_to_date) is not bool:
            raise EnhancerError("Storage and skip options must be enabled or disabled.")
        if type(self.reuse_identical) is not bool:
            raise EnhancerError("Identical-image reuse must be enabled or disabled.")
        if type(self.preserve_alpha) is not bool:
            raise EnhancerError("The alpha option must be enabled or disabled.")


@dataclass(frozen=True)
class Plan:
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    canvas_size: tuple[int, int]
    crop: tuple[int, int, int, int]

    source_crop: tuple[int, int, int, int] | None = None

    @property
    def content_target(self) -> tuple[int, int, int, int]:
        if self.source_crop is None:
            return (0, 0, *self.target_size)
        from .translation import map_box
        return map_box(self.source_crop, self.source_size, self.target_size)

    @property
    def size_string(self) -> str:
        return f"{self.canvas_size[0]}x{self.canvas_size[1]}"

    @property
    def resampled_output(self) -> bool:
        x, y, right, bottom = self.content_target
        return self.crop[2] - self.crop[0] < right - x or self.crop[3] - self.crop[1] < bottom - y

    @property
    def experimental(self) -> bool:
        return self.canvas_size[0] * self.canvas_size[1] > 2560 * 1440


def _align16(value: int) -> int:
    return ((value + 15) // 16) * 16


def target_dimensions(source_size: tuple[int, int], scale: int | float = 2, target_edge: int | None = None) -> tuple[int, int]:
    Settings(scale=scale, target_edge=target_edge).validate()
    if (len(source_size) != 2 or any(type(v) is not int or v <= 0 for v in source_size)):
        raise EnhancerError("Image dimensions must be positive integers.")
    if target_edge is None:
        # Use the decimal factor so that half pixels round up even for values like 1.005.
        numerator, denominator = Decimal(str(scale)).as_integer_ratio()
        target = tuple((dimension * numerator + denominator // 2) // denominator for dimension in source_size)
    else:
        # use the requested edge; round only the other dimension to the nearest pixel
        longest = max(source_size)
        edge = max(longest, target_edge)
        target = tuple(max(1, (dimension * edge + longest // 2) // longest) for dimension in source_size)
    if max(target) > MAX_OUTPUT_EDGE or target[0] * target[1] > MAX_OUTPUT_PIXELS:
        raise EnhancerError(f"Requested {target[0]:,} × {target[1]:,} pixels. Choose a smaller size to fit "
                            f"within {MAX_OUTPUT_EDGE:,} pixels per edge and {MAX_OUTPUT_PIXELS:,} pixels total.")
    return target


def plan_image(source_size: tuple[int, int], scale: int | float = 2, *, target_edge: int | None = None) -> Plan:
    target = target_dimensions(source_size, scale, target_edge)
    return _plan_target(source_size, target)


def _plan_target(source_size: tuple[int, int], target: tuple[int, int]) -> Plan:
    content_target = target
    # fit oversized artwork into a legal API canvas without changing the final requested size
    # include API alignment before deciding whether the requested canvas fits
    padded_w, padded_h = _align16(target[0]), _align16(target[1])
    padded_w = max(padded_w, _align16(math.ceil(padded_h / 3)))
    padded_h = max(padded_h, _align16(math.ceil(padded_w / 3)))
    ratio = min(1.0, MAX_EDGE / padded_w, MAX_EDGE / padded_h,
                math.sqrt(MAX_PIXELS / (padded_w * padded_h)))
    if ratio < 1.0:
        # leave rounding room for alignment to multiples of 16
        ratio *= 0.99
        content_target = tuple(max(1, round(dimension * ratio)) for dimension in target)
    # integer enlargement keeps small artwork proportional on the minimum API canvas
    multiple = max(1, math.ceil(1024 / max(content_target)))
    content_w, content_h = (v * multiple for v in content_target)
    canvas_w, canvas_h = _align16(content_w), _align16(content_h)
    canvas_w = max(canvas_w, _align16(math.ceil(canvas_h / 3)))
    canvas_h = max(canvas_h, _align16(math.ceil(canvas_w / 3)))
    if canvas_w * canvas_h < MIN_PIXELS:
        grow = math.sqrt(MIN_PIXELS / (canvas_w * canvas_h))
        canvas_w = _align16(math.ceil(canvas_w * grow))
        canvas_h = _align16(math.ceil(canvas_h * grow))
    # rounding can slightly violate the aspect limit, so enforce it again
    canvas_w = max(canvas_w, _align16(math.ceil(canvas_h / 3)))
    canvas_h = max(canvas_h, _align16(math.ceil(canvas_w / 3)))
    if (max(canvas_w, canvas_h) > MAX_EDGE or
            not MIN_PIXELS <= canvas_w * canvas_h <= MAX_PIXELS):
        raise EnhancerError("The padded API canvas exceeds the supported size. Try 2x or a smaller image.")
    x, y = (canvas_w - content_w) // 2, (canvas_h - content_h) // 2
    return Plan(source_size, target, (canvas_w, canvas_h), (x, y, x + content_w, y + content_h))


def load_texture(path: Path | str) -> Texture:
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise EnhancerError("Choose a regular image file.")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise EnhancerError("Input file is larger than the 128 MiB safety limit.")
    with path.open("rb") as handle:
        data = handle.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise EnhancerError("Input file is larger than the 128 MiB safety limit.")
    try:
        image, profile = decode_source(data, path.suffix)
        if image.width * image.height > MAX_SOURCE_PIXELS:
            raise EnhancerError("Input exceeds the source pixel safety limit.")
    except EnhancerError:
        raise
    except (OSError, ValueError, SyntaxError, NotImplementedError, struct.error,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise EnhancerError("Cannot decode this image safely. It may be corrupt or use an unsupported format variant.") from exc
    return Texture(path, image, profile.format, hashlib.sha256(data).hexdigest(), profile.notes, profile)


def has_transparency(image: Image.Image) -> bool:
    return image.convert("RGBA").getchannel("A").getextrema()[0] < 255


def make_canvas(source: Image.Image, plan: Plan, *, artwork: Image.Image | None = None) -> tuple[bytes, bool]:
    if source.size != plan.source_size:
        raise EnhancerError("The source no longer matches the prepared request.")
    rgba = (source.crop(plan.source_crop) if plan.source_crop else source).convert("RGBA")
    alpha_range = rgba.getchannel("A").getextrema()
    if alpha_range[1] == 0:
        raise EnhancerError("This image is completely transparent; there is no visible artwork to enhance.")
    alpha = alpha_range[0] < 255
    x0, y0, x1, y1 = plan.crop
    guide = rgba if artwork is None else artwork
    scaled = guide.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", plan.canvas_size, (0, 0, 0, 0) if alpha else (24, 24, 24, 255))
    canvas.paste(scaled, (x0, y0))
    output = io.BytesIO()
    canvas.save(output, format="PNG")
    data = output.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        raise EnhancerError("Converted PNG exceeds the 40 MiB upload safety limit.")
    return data, alpha


def build_prompt(plan: Plan, settings: Settings) -> str:
    settings.validate()
    if settings.task == "translate":
        from .translation import translation_prompt
        return translation_prompt(plan, settings)
    if settings.annotation:
        prompt = settings.annotation.prompt(plan)
        if settings.instruction.strip():
            prompt += (" Additional artwork instruction (the per-image annotation takes priority): "
                       + settings.instruction.strip())
        return prompt
    x0, y0, x1, y1 = plan.crop
    if settings.source_protection == "strict":
        prompt = (
            "Perform conservative restoration of this raster image. "
            "The supplied pixels are the authority, not a reference for a new illustration. "
            "The image is already enlarged to the working size; make only minimal corrections "
            "to visible sampling artifacts. Do not invent detail that the source does not resolve. "
            "Preserve every silhouette, contour, line weight, facial feature, pose, garment, "
            "pattern, logo, glyph, kerning, baseline, tiny copyright line and object position. "
            "Do not re-typeset, reinterpret, beautify or redraw the artwork. "
            "Preserve the original brightness, contrast, saturation, palette and shading. "
            "Faint, washed-out or low-contrast artwork is intentional: keep it equally faint. "
            "Do not darken outlines, recover faded background figures, add hatching, "
            "texture, highlights, shadows, gradients or sharpened halos. "
            "If a detail is ambiguous, keep its existing shape and softness. "
        )
    else:
        prompt = (
            "Restore the low-resolution artwork for a high-resolution display. Remove interpolation blur. "
            "Reconstruct clean antialiased contours, sharp lettering, fine strokes and distinct borders "
            "where the source supports them. Keep the exact spelling, glyph layout, silhouette and alignment. "
            "Retain the palette, flat fills, intentional soft gradients, glow and translucency. "
            "Do not add texture, change the design or introduce halos. "
        )
    prompt += (
        f"Return the full {plan.size_string} canvas unchanged in layout. "
        f"The artwork rectangle is x={x0}, y={y0}, width={x1-x0}, height={y1-y0}. "
        "Keep this exact rectangle; do not crop, zoom, shift or draw in the padding. "
        "The padding is not part of the source image. "
    )
    if settings.instruction.strip():
        prompt += ("The following explicit edit is the only exception to the preservation rules; "
                   "apply it while keeping everything else unchanged: " + settings.instruction.strip())
    return prompt


def source_coverage(alpha: Image.Image, size: tuple[int, int], detail: bool) -> Image.Image:
    if detail:
        from .super_resolution import upscale_artwork
        return upscale_artwork(alpha, size)
    return alpha.resize(size, Image.Resampling.BILINEAR)


def finish_image(raw: Image.Image, source: Image.Image, plan: Plan, preserve_alpha: bool,
                 *, detail: bool = False, guide: Image.Image | None = None) -> Image.Image:
    if raw.size != plan.canvas_size:
        raise EnhancerError(f"API returned {raw.width} x {raw.height}; expected {plan.size_string}. "
                            "The image was not cropped because that could damage alignment.")
    if source.size != plan.source_size:
        raise EnhancerError("The source no longer matches the result.")
    content = raw.convert("RGBA").crop(plan.crop)
    if detail and plan.resampled_output and all(a <= b for a, b in zip(content.size, plan.target_size)):
        from .super_resolution import upscale_artwork
        result = upscale_artwork(content, plan.target_size)
    else:
        result = content.resize(plan.target_size, Image.Resampling.LANCZOS)
    original_alpha = source.convert("RGBA").getchannel("A")
    if preserve_alpha:
        # coverage is resampled, not regenerated; this cannot preserve layout changed by AI
        if guide is not None and guide.size == plan.target_size:
            result.putalpha(guide.getchannel("A"))
        else:
            result.putalpha(source_coverage(original_alpha, plan.target_size, detail))
    if original_alpha.getextrema() == (255, 255):
        if not preserve_alpha and result.getchannel("A").getextrema()[0] != 255:
            raise EnhancerError("The model returned transparency for an opaque input. Enable Keep original alpha or inspect the recovered canvas.")
        result = result.convert("RGB")
    result.info.clear()
    return result


def save_image_atomic(image: Image.Image, destination: Path | str,
                      source_path: Path | None = None, *, overwrite: bool = False) -> Path:
    """Save an intermediate PNG or TGA, not an original-format compatibility export."""
    destination = Path(destination).expanduser().absolute()
    if destination.suffix.lower() not in (".png", ".tga"):
        raise EnhancerError("Intermediate output must be PNG or TGA. Use Save Copy for native export.")
    stream = io.BytesIO()
    image.save(stream, format="PNG" if destination.suffix.lower() == ".png" else "TGA")
    return write_atomic(stream.getvalue(), destination, source_path, overwrite=overwrite)


def prepare_recovery(directory: Path) -> Path:
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=directory):
        pass
    return directory


def plan_local(source_size: tuple[int, int], scale: int | float = 2, *, target_edge: int | None = None) -> Plan:
    target = target_dimensions(source_size, scale, target_edge)
    return Plan(source_size, target, target, (0, 0, *target))


def plan_settings(source_size: tuple[int, int], settings: Settings) -> Plan:
    settings.validate()
    from .translation import validate_regions
    validate_regions(settings, source_size)
    if settings.annotation:
        settings.annotation.check_bounds(source_size)
    plan = (plan_image if settings.method == "openai" else plan_local)(source_size, settings.scale, target_edge=settings.target_edge)
    if settings.task == "translate" and settings.translation_scope == "regions" and settings.optimize_translation:
        from .translation import optimize_plan
        plan = optimize_plan(plan, settings)
    return plan


def is_up_to_date(settings: Settings, plan: Plan) -> bool:
    return (settings.skip_up_to_date and settings.task == "enhance" and settings.annotation is None and settings.target_edge is not None
            and plan.target_size == plan.source_size)


def needs_api(settings: Settings, plan: Plan) -> bool:
    return settings.method == "openai" and not is_up_to_date(settings, plan)
