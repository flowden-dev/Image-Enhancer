"""Bounded binary parsing and palette/color helpers."""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageCms

from ..errors import EnhancerError

MAX_PIXELS = 16_777_216
# Allow a 12× export of a 1024-square texture while keeping input limits separate.
MAX_OUTPUT_PIXELS = 12_288 ** 2
MAX_EDGE = 16384
# Includes a 32-bit DDS with a complete mip chain or an uncompressed RGBA PNG.
MAX_OUTPUT_FILE_BYTES = MAX_OUTPUT_PIXELS * 6


def check_size(width: int, height: int, *, output: bool = False) -> None:
    limit = MAX_OUTPUT_PIXELS if output else MAX_PIXELS
    if width < 1 or height < 1 or max(width, height) > MAX_EDGE or width * height > limit:
        raise EnhancerError(f"Image must fit within {MAX_EDGE:,} pixels per edge and {limit:,} pixels total.")


def open_output_image(data: bytes, size: tuple[int, int]) -> Image.Image:
    """Open a planned export. Check its header against the plan before decoding pixels."""
    check_size(*size, output=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        image = Image.open(io.BytesIO(data))
    if image.size != size:
        image.close()
        raise EnhancerError("Exported image dimensions do not match the requested scale.")
    return image


@dataclass(frozen=True)
class Profile:
    format: str
    mode: str
    size: tuple[int, int]
    details: Any = None
    icc: bytes = b""
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def extension(self) -> str:
        return {"DDS": ".dds", "TGA": ".tga", "PNG": ".png", "JPEG": ".jpg",
                "BMP": ".bmp", "WEBP": ".webp"}[self.format]

    @property
    def data_texture(self) -> bool:
        return bool(getattr(self.details, "data_texture", False))

    @property
    def description(self) -> str:
        if self.details is not None:
            return self.details.description
        if self.format == "JPEG":
            progressive = "progressive" if self.metadata.get("progressive") else "sequential"
            return f"JPEG {self.mode}, {progressive}, original quantization and sampling"
        return f"{self.format} {self.mode}"


def rgba_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def unpremultiply(image: Image.Image) -> Image.Image:
    result = Image.new("RGBA", image.size)
    for top in range(0, image.height, 64):
        data = rgba_array(image.crop((0, top, image.width, min(top + 64, image.height)))).astype(np.uint32)
        alpha = data[:, :, 3:4]
        data[:, :, :3] = np.where(alpha == 0, 0, np.minimum(255, (data[:, :, :3] * 255 + alpha // 2) // np.maximum(alpha, 1)))
        result.paste(Image.fromarray(data.astype(np.uint8)), (0, top))
    return result


def premultiply(image: Image.Image) -> Image.Image:
    result = Image.new("RGBA", image.size)
    for top in range(0, image.height, 64):
        data = rgba_array(image.crop((0, top, image.width, min(top + 64, image.height)))).astype(np.uint16)
        data[:, :, :3] = (data[:, :, :3] * data[:, :, 3:4] + 127) // 255
        result.paste(Image.fromarray(data.astype(np.uint8)), (0, top))
    return result


def palette_indices(image: Image.Image, palette: np.ndarray) -> np.ndarray:
    """Map to the existing palette; do not replace alpha entries or palette indices."""
    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 4 or not 1 <= len(palette) <= 256:
        raise EnhancerError("Only palettes containing 1 to 256 RGBA entries are supported.")
    pixels = rgba_array(image)
    # Pillow's C quantizer is fast for opaque RGB palettes
    if np.all(palette[:, 3] == 255) and np.all(pixels[:, :, 3] == 255):
        reference = Image.new("P", (1, 1))
        rgb = palette[:, :3].tobytes()
        reference.putpalette(rgb + rgb[-3:] * (256 - len(palette)))
        quantized = image.convert("RGB").quantize(palette=reference, dither=Image.Dither.NONE)
        return np.asarray(quantized, dtype=np.uint8).clip(0, len(palette) - 1)
    # include alpha in distance; an RGB-only quantizer can turn transparent icons opaque
    flat = pixels.reshape(-1, 4)
    output = np.empty(len(flat), dtype=np.uint8)
    colors = palette.astype(np.int32)
    # bound unique-sort/inverse memory for large textures; preserve the same RGBA distance
    for offset in range(0, len(flat), 65536):
        unique, inverse = np.unique(flat[offset:offset + 65536], axis=0, return_inverse=True)
        nearest = np.empty(len(unique), dtype=np.uint8)
        for start in range(0, len(unique), 1024):
            delta = unique[start:start + 1024, None, :].astype(np.int32) - colors[None, :, :]
            square = delta * delta
            square[:, :, 3] *= 4
            nearest[start:start + 1024] = np.argmin(square.sum(axis=2), axis=1)
        output[offset:offset + len(inverse)] = nearest[inverse]
    return output.reshape(image.height, image.width)


def to_srgb(image: Image.Image, profile: Profile) -> Image.Image:
    if not profile.icc:
        return image.convert("RGBA")
    try:
        original = ImageCms.ImageCmsProfile(io.BytesIO(profile.icc))
        mode = "L" if profile.mode == "LA" else (profile.mode if profile.mode in ("RGB", "CMYK", "L") else "RGB")
        source = image.convert(mode)
        converted = ImageCms.profileToProfile(source, original, ImageCms.createProfile("sRGB"), outputMode="RGB")
        converted.putalpha(image.convert("RGBA").getchannel("A"))
        return converted
    except (OSError, ValueError, ImageCms.PyCMSError) as exc:
        raise EnhancerError("The embedded ICC color profile cannot be converted safely. Convert the source explicitly to sRGB first.") from exc


def from_srgb(image: Image.Image, profile: Profile) -> Image.Image:
    if not profile.icc:
        return image.copy()
    try:
        original = ImageCms.ImageCmsProfile(io.BytesIO(profile.icc))
        mode = "L" if profile.mode == "LA" else (profile.mode if profile.mode in ("RGB", "CMYK", "L") else "RGB")
        result = ImageCms.profileToProfile(image.convert("RGB"), ImageCms.createProfile("sRGB"), original, outputMode=mode)
        if mode == "RGB" and "A" in image.getbands():
            result.putalpha(image.getchannel("A"))
        elif mode == "L" and profile.format != "JPEG" and "A" in image.getbands():
            result.putalpha(image.getchannel("A"))
        return result
    except (OSError, ValueError, ImageCms.PyCMSError) as exc:
        raise EnhancerError("The original ICC profile could not be restored. The sRGB recovery PNG is retained.") from exc


def restore_hidden_rgb(result: Image.Image, source: Image.Image) -> Image.Image:
    """Retain stored color under zero alpha without changing visible compositing."""
    if result.mode != "RGBA":
        return result
    alpha = result.getchannel("A")
    if alpha.getextrema()[0] == 0:
        reference = source.convert("RGB").resize(result.size, Image.Resampling.LANCZOS)
        hidden = alpha.point(lambda value: 255 if value == 0 else 0)
        result.paste(reference, (0, 0), hidden)
        result.putalpha(alpha)
    result.info.clear()
    return result


def resize_artwork(source: Image.Image, size: tuple[int, int], method=Image.Resampling.LANCZOS) -> Image.Image:
    result = source.resize(size, method)
    return restore_hidden_rgb(result, source) if method != Image.Resampling.NEAREST else result
