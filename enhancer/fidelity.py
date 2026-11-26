"""Bound generative color changes using the original artwork."""
from PIL import Image, ImageFilter
import numpy as np

from .errors import EnhancerError
from .formats.common import resize_artwork


def _protect_detail(result: Image.Image, source: Image.Image) -> Image.Image:
    """Preserve local colors while allowing sharp contours to move within a source texel."""
    rgb = source.convert("RGB")
    clear = source.convert("RGBA").getchannel("A").point(lambda a: 255 if a == 0 else 0)
    low = rgb.copy()
    low.paste((255, 255, 255), mask=clear)
    low = low.filter(ImageFilter.MinFilter(3))
    high = rgb.copy()
    high.paste((0, 0, 0), mask=clear)
    high = high.filter(ImageFilter.MaxFilter(3))
    # A wholly invisible neighborhood contributes no color constraint.
    empty = clear.filter(ImageFilter.MinFilter(3))
    low.paste((0, 0, 0), mask=empty)
    high.paste((255, 255, 255), mask=empty)
    output = result.copy()
    for top in range(0, result.height, 128):
        bottom = min(top + 128, result.height)
        box = (0, top * source.height / result.height, source.width, bottom * source.height / result.height)
        size = (result.width, bottom - top)
        lower = np.asarray(low.resize(size, Image.Resampling.BILINEAR, box), dtype=np.int16)
        upper = np.asarray(high.resize(size, Image.Resampling.BILINEAR, box), dtype=np.int16)
        region = (0, top, result.width, bottom)
        edited = np.array(result.crop(region))
        colors = edited[:, :, :3]
        colors[:] = np.clip(colors, lower - 12, upper + 12)
        output.paste(Image.fromarray(edited), region)
    output.info.clear()
    return output


def protect_source(result: Image.Image, source: Image.Image, protection: str) -> Image.Image:
    """Limit RGB deltas before export; this is not geometric registration or super-resolution.

    Alpha remains controlled by the existing alpha option. Native lossy encoding
    can introduce further differences after this bound is applied.
    """
    if protection == "off":
        return result
    if protection not in ("detail", "strict", "balanced"):
        raise EnhancerError("Unknown source protection setting.")
    if result.mode not in ("RGB", "RGBA"):
        raise EnhancerError("Source protection requires an RGB or RGBA result.")
    if protection == "detail":
        return _protect_detail(result, source)
    limit = 12 if protection == "strict" else 32
    reference = resize_artwork(source.convert("RGBA"), result.size)
    protected = result.copy()
    # bounded strips avoid several full-size float arrays for 8K textures
    for top in range(0, result.height, 128):
        box = (0, top, result.width, min(result.height, top + 128))
        base = np.array(reference.crop(box).convert("RGB"), dtype=np.int16)
        edited = np.array(result.crop(box))
        rgb = edited[:, :, :3].astype(np.int16)
        edited[:, :, :3] = np.clip(rgb, np.maximum(base - limit, 0),
                                  np.minimum(base + limit, 255)).astype(np.uint8)
        protected.paste(Image.fromarray(edited), box)
    reference.close()
    protected.info.clear()
    return protected
