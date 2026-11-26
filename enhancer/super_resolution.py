"""Tiled CPU inference for the bundled Anime4K x2 artwork model."""
from functools import lru_cache
from importlib.resources import files
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from .errors import EnhancerError
from .formats.common import restore_hidden_rgb

_TILE = 160
_LAYERS = 7
# Transparent-color extension, network receptive field and final resampling footprint.
_HALO = 2 * _LAYERS + 1


@lru_cache(maxsize=1)
def _weights():
    asset = files("enhancer").joinpath("models", "anime4k_x2_m.npz")
    try:
        with asset.open("rb") as stream, np.load(stream, allow_pickle=False) as weights:
            layers = [(weights[f"k{i}"], weights[f"b{i}"]) for i in range(_LAYERS)]
            return layers, weights["head"], weights["bias"]
    except (OSError, BadZipFile, ValueError, KeyError) as exc:
        raise EnhancerError("The bundled neural model could not be read. Re-extract the complete app.") from exc


def _predict_2x(pixels):
    """Infer four subpixel luminance corrections per input pixel. Values are in [0,1]."""
    height, width, channels = pixels.shape
    layers, head, bias = _weights()
    residual = np.broadcast_to(bias, (height, width, 4)).copy()
    features = pixels
    for index, (kernel, layer_bias) in enumerate(layers):
        if index == 0 and channels == 1:
            # A grayscale alpha mask is equivalent to three identical input channels.
            kernel = kernel.sum(axis=2, keepdims=True)
        padded = np.pad(features, ((1, 1), (1, 1), (0, 0)), mode="edge")
        convolved = np.broadcast_to(layer_bias, (height, width, 4)).copy()
        for y in range(3):
            for x in range(3):
                convolved += padded[y:y + height, x:x + width] @ kernel[y, x]
        features = np.concatenate((np.maximum(convolved, 0), np.maximum(-convolved, 0)), axis=2)
        residual += features @ head[index]

    # Pixel shuffle, with the bilinear skip connection used by the trained model.
    # Bound each channel to its contributing texels to keep flat fills and avoid ringing.
    padded = np.pad(pixels, ((1, 1), (1, 1), (0, 0)), mode="edge")
    result = np.empty((height * 2, width * 2, channels), dtype=np.uint8)
    for iy, y in enumerate((0, 2)):
        for ix, x in enumerate((0, 2)):
            vertical = padded[y:y + height, 1:1 + width]
            horizontal = padded[1:1 + height, x:x + width]
            diagonal = padded[y:y + height, x:x + width]
            base = (9 * pixels + 3 * (vertical + horizontal) + diagonal) * (1 / 16)
            low = np.minimum(np.minimum(pixels, vertical), np.minimum(horizontal, diagonal))
            high = np.maximum(np.maximum(pixels, vertical), np.maximum(horizontal, diagonal))
            value = base + residual[:, :, 2 * iy + ix, None]
            result[iy::2, ix::2] = np.rint(np.clip(value, low, high) * 255).astype(np.uint8)
    return result


def _extend_visible_color(rgb, alpha):
    """Give the CNN visible neighbors' color on clear texels regardless of hidden RGB."""
    coverage = alpha[:, :, None]
    weighted = np.concatenate((rgb * coverage, coverage), axis=2)
    radius = _LAYERS
    padded = np.pad(weighted, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    # Float64 integral sums keep the result independent of the tile origin even at low coverage.
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0))).cumsum(axis=0, dtype=np.float64).cumsum(axis=1)
    span = 2 * radius + 1
    sums = integral[span:, span:] - integral[:-span, span:] - integral[span:, :-span] + integral[:-span, :-span]
    clear = alpha == 0
    rgb[clear] = (sums[:, :, :3] / np.maximum(sums[:, :, 3:], 1e-10))[clear]


def _double_tile(tile):
    size = (tile.width * 2, tile.height * 2)
    if tile.mode == "L":
        lo, hi = tile.getextrema()
        if lo == hi:
            return Image.new("L", size, lo)
        alpha = np.asarray(tile, dtype=np.float32)[:, :, None] / 255
        return Image.fromarray(_predict_2x(alpha)[:, :, 0])

    alpha = tile.getchannel("A")
    lo, hi = alpha.getextrema()
    if hi == 0:
        return Image.new("RGBA", size)
    colors = tile.convert("RGB")
    if all(low == high for low, high in colors.getextrema()):
        result = Image.new("RGB", size, colors.getpixel((0, 0)))
    else:
        rgb = np.asarray(colors, dtype=np.float32) / 255
        if lo == 0:
            _extend_visible_color(rgb, np.asarray(alpha, dtype=np.float32) / 255)
        result = Image.fromarray(_predict_2x(rgb))
    result.putalpha(_double_tile(alpha) if lo != hi else lo)
    return result


def _area_taps(start, stop, doubled_length, target_length, origin):
    # Each output pixel covers between one and two predicted pixels, intersecting at most three.
    left = np.arange(start, stop, dtype=np.int64)[:, None] * doubled_length
    index = left // target_length + np.arange(3)
    weight = np.maximum(0, np.minimum(left + doubled_length, (index + 1) * target_length)
                        - np.maximum(left, index * target_length))
    return np.minimum(index, doubled_length - 1) - 2 * origin, weight


def _fractional_tile(tile, dest, region, source_size, target):
    # Use global integer phases on every tile, including at very large or odd dimensions.
    width, height = (2 * n for n in source_size)
    x, wx = _area_taps(dest[0], dest[2], width, target[0], region[0])
    y, wy = _area_taps(dest[1], dest[3], height, target[1], region[1])
    samples = np.asarray(tile, dtype=np.int64)
    if tile.mode == "RGBA":
        samples[:, :, :3] *= samples[:, :, 3:]
    else:
        samples = samples[:, :, None]
    rows = sum(samples[:, x[:, k]] * wx[None, :, k, None] for k in range(3))
    total = sum(rows[y[:, k]] * wy[:, k, None, None] for k in range(3))
    area = width * height
    value = (total + area // 2) // area
    if tile.mode == "RGBA":
        alpha = total[:, :, 3:]
        value[:, :, :3] = (total[:, :, :3] + alpha // 2) // np.maximum(alpha, 1)
    else:
        value = value[:, :, 0]
    return Image.fromarray(value.astype(np.uint8))


def _upscale_pass(source, target):
    output = Image.new(source.mode, target)
    for top in range(0, source.height, _TILE):
        bottom = min(top + _TILE, source.height)
        for left in range(0, source.width, _TILE):
            right = min(left + _TILE, source.width)
            region = (max(left - _HALO, 0), max(top - _HALO, 0),
                      min(right + _HALO, source.width), min(bottom + _HALO, source.height))
            doubled = _double_tile(source.crop(region))
            # Map tile boundaries in global coordinates so fractional factors have no seams.
            dest = (left * target[0] // source.width, top * target[1] // source.height,
                    right * target[0] // source.width, bottom * target[1] // source.height)
            if target == (2 * source.width, 2 * source.height):
                box = (2 * (left - region[0]), 2 * (top - region[1]),
                       2 * (right - region[0]), 2 * (bottom - region[1]))
                output.paste(doubled.crop(box), dest)
                continue
            piece = _fractional_tile(doubled, dest, region, source.size, target)
            output.paste(piece, dest)
    return output


def upscale_artwork(source: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Reconstruct up to 2× per pass, returning exactly the planned size (RGBA or coverage)."""
    if source.size == target:
        return source.copy()
    if source.mode not in ("L", "RGBA"):
        raise ValueError("Artwork reconstruction requires RGBA artwork or an L coverage mask.")
    if target[0] < source.width or target[1] < source.height:
        raise ValueError("Artwork reconstruction requires an enlargement.")
    extrema = source.getextrema()
    bands = (extrema,) if source.mode == "L" else extrema
    if all(low == high for low, high in bands):
        return Image.new(source.mode, target, source.getpixel((0, 0)))
    current = source
    while current.size != target:
        size = (min(current.width * 2, target[0]), min(current.height * 2, target[1]))
        current = _upscale_pass(current, size)
    return restore_hidden_rgb(current, source) if source.mode == "RGBA" else current
