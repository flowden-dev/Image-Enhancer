"""TGA round-trip codec retaining layout, palette, RLE and pixel depth."""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..errors import EnhancerError
from .common import check_size, palette_indices, premultiply, unpremultiply, rgba_array

FOOTER = b"TRUEVISION-XFILE.\0"


@dataclass(frozen=True)
class TGA:
    header: bytes
    identifier: bytes
    palette_bytes: bytes
    palette_rgba: bytes
    extension: bytes
    color_table: bytes
    footer: bool

    @property
    def width(self) -> int:
        return struct.unpack_from("<H", self.header, 12)[0]

    @property
    def height(self) -> int:
        return struct.unpack_from("<H", self.header, 14)[0]

    @property
    def bits(self) -> int:
        return self.header[16]

    @property
    def image_type(self) -> int:
        return self.header[2]

    @property
    def indexed(self) -> bool:
        return self.image_type in (1, 9)

    @property
    def rle(self) -> bool:
        return self.image_type in (9, 10, 11)

    @property
    def premultiplied(self) -> bool:
        return bool(self.extension and self.extension[494] == 4)

    @property
    def data_texture(self) -> bool:
        # TGA 2 types 1/2 store undefined attributes, not opacity
        return bool(self.extension and self.extension[494] in (1, 2))

    @property
    def description(self) -> str:
        return f"TGA {self.bits}-bit {'indexed' if self.indexed else ('gray' if self.image_type in (3, 11) else 'color')}, {'RLE' if self.rle else 'uncompressed'}, original origin/alpha layout"


def _decode_colors(raw: bytes, bits: int, attributes: int) -> np.ndarray:
    bytecount = (bits + 7) // 8
    src = np.frombuffer(raw, dtype=np.uint8).reshape(-1, bytecount)
    out = np.empty((len(src), 4), dtype=np.uint8)
    out[:, 3] = 255
    if bits in (15, 16):
        packed = src[:, 0].astype(np.uint16) | (src[:, 1].astype(np.uint16) << 8)
        for i, shift in enumerate((10, 5, 0)):
            out[:, i] = (((packed >> shift) & 31) * 255 + 15) // 31
        if attributes:
            out[:, 3] = ((packed >> 15) & 1) * 255
    else:
        out[:, :3] = src[:, [2, 1, 0]]
        if bits == 32 and attributes:
            out[:, 3] = src[:, 3]
    return out


def inspect(data: bytes, *, output: bool = False) -> TGA:
    if len(data) < 18:
        raise EnhancerError("Truncated TGA header.")
    header = data[:18]
    id_length, mapped, image_type, first, count, palette_bits, x, y, width, height, bits, descriptor = struct.unpack("<BBBHHBHHHHBB", header)
    check_size(width, height, output=output)
    if image_type not in (1, 2, 3, 9, 10, 11) or mapped not in (0, 1):
        raise EnhancerError("Only standard single-image TGA types 1/2/3/9/10/11 are supported.")
    if descriptor >> 6 == 3:
        raise EnhancerError("TGA uses reserved interleaving flags.")
    attributes = descriptor & 15
    indexed = image_type in (1, 9)
    if indexed:
        if mapped != 1 or not 1 <= count <= 256 or bits not in (8, 16) or first + count > 1 << bits or palette_bits not in (15, 16, 24, 32):
            raise EnhancerError("TGA palette/index layout cannot be retained safely.")
    elif mapped or count:
        raise EnhancerError("TGA has an unexpected palette attached to a non-indexed image.")
    elif image_type in (2, 10):
        if bits not in (15, 16, 24, 32) or attributes not in (0, 1, 8) or (attributes == 8 and bits != 32) or (attributes == 1 and bits != 16):
            raise EnhancerError("Unsupported TGA color/alpha bit layout.")
    elif bits not in (8, 16) or attributes not in (0, 8) or attributes and bits != 16:
        raise EnhancerError("Unsupported TGA grayscale/alpha bit layout.")
    palette_length = count * ((palette_bits + 7) // 8) if mapped else 0
    start = 18 + id_length + palette_length
    if start >= len(data):
        raise EnhancerError("TGA identifier/palette or pixels are truncated.")
    identifier = data[18:18 + id_length]
    palette_bytes = data[18 + id_length:start]
    palette = b""
    if mapped:
        palette = _decode_colors(palette_bytes, palette_bits, 8 if palette_bits == 32 else (1 if palette_bits == 16 else 0)).tobytes()
    footer = len(data) >= 26 and data[-18:] == FOOTER
    extension = color_table = b""
    if footer:
        ext_offset, dev_offset = struct.unpack_from("<II", data, len(data) - 26)
        if dev_offset:
            raise EnhancerError("TGA contains a private developer directory; its offset-dependent data cannot be safely resized.")
        if ext_offset:
            if ext_offset < start or ext_offset + 495 > len(data) - 26 or struct.unpack_from("<H", data, ext_offset)[0] != 495:
                raise EnhancerError("TGA extension area is truncated or invalid.")
            extension = data[ext_offset:ext_offset + 495]
            if extension[494] > 4:
                raise EnhancerError("TGA has reserved alpha semantics.")
            if extension[494] == 0 and attributes:
                raise EnhancerError("TGA declares no alpha in its extension but has alpha bits in its descriptor.")
            color_offset, stamp_offset, scan_offset = struct.unpack_from("<III", extension, 482)
            if color_offset:
                if color_offset < start or color_offset + 2048 > len(data) - 26:
                    raise EnhancerError("TGA color correction table is truncated.")
                color_table = data[color_offset:color_offset + 2048]
            if stamp_offset and not start <= stamp_offset < len(data) - 26:
                raise EnhancerError("TGA thumbnail offset is outside the file.")
            if scan_offset and not start <= scan_offset <= len(data) - 26 - height * 4:
                raise EnhancerError("TGA scan-line table is outside the file.")
    if palette and extension and extension[494] == 0:
        colors = np.frombuffer(palette, dtype=np.uint8).reshape(-1, 4).copy()
        colors[:, 3] = 255
        palette = colors.tobytes()
    return TGA(header, identifier, palette_bytes, palette, extension, color_table, footer)


def _pixels(data: bytes, info: TGA) -> tuple[bytes, int]:
    start = 18 + len(info.identifier) + len(info.palette_bytes)
    limit = len(data) - (26 if info.footer else 0)
    if info.extension:
        ext_offset = struct.unpack_from("<I", data, len(data) - 26)[0]
        offsets = [ext_offset] + [x for x in struct.unpack_from("<III", info.extension, 482) if x]
        limit = min(limit, *offsets)
    count = info.width * info.height
    bytecount = (info.bits + 7) // 8
    if not info.rle:
        end = start + count * bytecount
        if end > limit:
            raise EnhancerError("TGA pixels are truncated.")
        return data[start:end], end
    result = bytearray(count * bytecount)
    written = 0
    while written < count:
        if start >= limit:
            raise EnhancerError("TGA RLE stream is truncated.")
        packet = data[start]
        start += 1
        run = (packet & 127) + 1
        if written + run > count:
            raise EnhancerError("TGA RLE packet exceeds the pixel count.")
        need = bytecount if packet & 128 else run * bytecount
        if start + need > limit:
            raise EnhancerError("TGA RLE packet is truncated.")
        value = data[start:start + need]
        result[written * bytecount:(written + run) * bytecount] = value * run if packet & 128 else value
        start += need
        written += run
    return bytes(result), start


def _rows(height: int, descriptor: int) -> np.ndarray:
    factor = 1 << (descriptor >> 6)
    return np.concatenate([np.arange(start, height, factor) for start in range(factor)])


def decode(data: bytes, info: TGA) -> Image.Image:
    raw, end = _pixels(data, info)
    descriptor = info.header[17]
    if info.indexed:
        indices = np.frombuffer(raw, dtype=np.uint8 if info.bits == 8 else "<u2").astype(np.int32)
        first = struct.unpack_from("<H", info.header, 3)[0]
        palette = np.frombuffer(info.palette_rgba, dtype=np.uint8).reshape(-1, 4)
        indices -= first
        if indices.min() < 0 or indices.max() >= len(palette):
            raise EnhancerError("TGA contains an out-of-range palette index.")
        pixels = palette[indices]
    elif info.image_type in (3, 11):
        src = np.frombuffer(raw, dtype=np.uint8).reshape(-1, info.bits // 8)
        pixels = np.empty((len(src), 4), dtype=np.uint8)
        pixels[:, :3] = src[:, :1]
        pixels[:, 3] = src[:, 1] if info.bits == 16 and (descriptor & 15 or info.data_texture) else 255
    else:
        pixels = _decode_colors(raw, info.bits, (descriptor & 15) or int(info.data_texture and info.bits in (16, 32)))
    rows = pixels.reshape(info.height, info.width, 4)
    ordered = np.empty_like(rows)
    ordered[_rows(info.height, descriptor)] = rows
    if not descriptor & 32:
        ordered = ordered[::-1]
    if descriptor & 16:
        ordered = ordered[:, ::-1]
    image = Image.fromarray(ordered)
    return unpremultiply(image) if info.premultiplied else image


def _rle_encode(raw: np.ndarray) -> bytes:
    """Emit row-bounded packets, including 128-pixel and final-pixel boundaries."""
    output = io.BytesIO()
    for row in raw:
        values = [pixel.tobytes() for pixel in row]
        i = 0
        while i < len(values):
            run = 1
            while i + run < len(values) and run < 128 and values[i + run] == values[i]:
                run += 1
            if run > 1:
                output.write(bytes((127 + run,)))
                output.write(values[i])
                i += run
                continue
            begin = i
            i += 1
            while i < len(values) and i - begin < 128:
                if i + 1 < len(values) and values[i] == values[i + 1]:
                    break
                i += 1
            output.write(bytes((i - begin - 1,)))
            output.write(b"".join(values[begin:i]))
    return output.getvalue()


def encode(image: Image.Image, info: TGA) -> bytes:
    check_size(*image.size, output=True)
    if max(image.size) > 65535:
        raise EnhancerError("TGA dimensions exceed its 16-bit header fields.")
    if info.premultiplied:
        image = premultiply(image)
    pixels = rgba_array(image)
    descriptor = info.header[17]
    if info.indexed:
        palette = np.frombuffer(info.palette_rgba, dtype=np.uint8).reshape(-1, 4)
        indices = palette_indices(image, palette).astype(np.uint16)
        indices += struct.unpack_from("<H", info.header, 3)[0]
        raw = indices.astype(np.uint8 if info.bits == 8 else "<u2").view(np.uint8).reshape(image.height, image.width, info.bits // 8)
    elif info.image_type in (3, 11):
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        raw = gray[:, :, None] if info.bits == 8 else np.dstack((gray, pixels[:, :, 3] if descriptor & 15 or info.data_texture else np.full(gray.shape, 255, dtype=np.uint8)))
    elif info.bits in (15, 16):
        p = pixels.astype(np.uint16)
        packed = (((p[:, :, 0] * 31 + 127) // 255) << 10) | (((p[:, :, 1] * 31 + 127) // 255) << 5) | ((p[:, :, 2] * 31 + 127) // 255)
        if descriptor & 15 or info.data_texture and info.bits == 16:
            packed |= (p[:, :, 3] >= 128).astype(np.uint16) << 15
        raw = packed.astype("<u2").view(np.uint8).reshape(image.height, image.width, 2)
    else:
        raw = pixels[:, :, [2, 1, 0, 3] if info.bits == 32 else [2, 1, 0]].copy()
        if info.bits == 32 and not descriptor & 15 and not info.data_texture:
            raw[:, :, 3] = 255
    if descriptor & 16:
        raw = raw[:, ::-1]
    if not descriptor & 32:
        raw = raw[::-1]
    raw = raw[_rows(image.height, descriptor)]
    header = bytearray(info.header)
    struct.pack_into("<HH", header, 12, *image.size)
    result = bytearray(header + info.identifier + info.palette_bytes)
    result.extend(_rle_encode(raw) if info.rle else raw.tobytes())
    extension_offset = 0
    if info.extension:
        extension = bytearray(info.extension)
        color_offset = len(result) if info.color_table else 0
        result.extend(info.color_table)
        # old thumbnails and scan-line offsets describe the previous dimensions
        struct.pack_into("<III", extension, 482, color_offset, 0, 0)
        extension_offset = len(result)
        result.extend(extension)
    if info.footer:
        result.extend(struct.pack("<II", extension_offset, 0) + FOOTER)
    output = bytes(result)
    verified = inspect(output, output=True)
    if verified.header[:12] != info.header[:12] or verified.header[16:] != info.header[16:] or verified.palette_bytes != info.palette_bytes:
        raise EnhancerError("TGA export changed the source's compatibility layout.")
    return output
