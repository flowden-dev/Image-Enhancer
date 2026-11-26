"""PNG writer retaining original color type, palette, depth and Adam7 layout."""
from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..errors import EnhancerError
from .common import check_size, palette_indices, rgba_array

SIGNATURE = b"\x89PNG\r\n\x1a\n"
PASSES = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
          (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


@dataclass(frozen=True)
class PNG:
    width: int
    height: int
    depth: int
    color: int
    interlace: int
    palette: bytes
    transparency: bytes
    chunks: tuple[tuple[bytes, bytes], ...]

    @property
    def description(self) -> str:
        names = {0: "grayscale", 2: "RGB", 3: "indexed", 4: "gray+alpha", 6: "RGBA"}
        return f"PNG {self.depth}-bit {names[self.color]}, {'Adam7' if self.interlace else 'non-interlaced'}"


def inspect(data: bytes, *, output: bool = False) -> PNG:
    if not data.startswith(SIGNATURE):
        raise EnhancerError("Invalid PNG signature.")
    position, seen_ihdr, ended, seen_idat = 8, False, False, False
    palette = transparency = b""
    saved = []
    unique = set()
    idat_ended = False
    width = height = depth = color = interlace = 0
    while position < len(data):
        if position + 12 > len(data):
            raise EnhancerError("PNG chunk header is truncated.")
        length = struct.unpack_from(">I", data, position)[0]
        kind = data[position + 4:position + 8]
        if any(not (65 <= c <= 90 or 97 <= c <= 122) for c in kind) or kind[2] & 32:
            raise EnhancerError("PNG contains an invalid chunk name or reserved bit.")
        if kind in (b"IHDR", b"PLTE", b"tRNS", b"iCCP", b"sRGB", b"gAMA", b"cHRM", b"pHYs", b"sBIT", b"bKGD", b"eXIf"):
            if kind in unique:
                raise EnhancerError("PNG contains a duplicate singleton chunk.")
            unique.add(kind)
        if seen_idat and kind != b"IDAT":
            idat_ended = True
        if kind == b"IDAT" and idat_ended:
            raise EnhancerError("PNG IDAT chunks are not contiguous.")
        end = position + 12 + length
        if end > len(data):
            raise EnhancerError("PNG chunk body is truncated.")
        payload = data[position + 8:position + 8 + length]
        checksum = struct.unpack_from(">I", data, position + 8 + length)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != checksum:
            raise EnhancerError("PNG chunk checksum failed.")
        if not seen_ihdr and kind != b"IHDR":
            raise EnhancerError("PNG does not begin with IHDR.")
        if kind == b"IHDR":
            if seen_ihdr or length != 13:
                raise EnhancerError("Invalid or duplicate PNG IHDR.")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            check_size(width, height, output=output)
            if compression or filtering or interlace not in (0, 1):
                raise EnhancerError("Unsupported PNG compression/filter/interlace method.")
            valid = {0: (1, 2, 4, 8), 2: (8,), 3: (1, 2, 4, 8), 4: (8,), 6: (8,)}
            if color not in valid or depth not in valid[color]:
                raise EnhancerError("Only 1/2/4/8-bit PNG color artwork is supported. 16-bit PNG is rejected, not silently reduced to 8-bit.")
            seen_ihdr = True
        elif kind == b"PLTE":
            if palette or seen_idat or not length or length % 3 or length > 768:
                raise EnhancerError("Invalid PNG palette.")
            if color in (0, 4):
                raise EnhancerError("Grayscale PNG cannot contain a palette.")
            palette = payload
        elif kind == b"tRNS":
            if transparency or seen_idat:
                raise EnhancerError("Invalid duplicate or misplaced PNG transparency.")
            transparency = payload
        elif kind == b"IDAT":
            seen_idat = True
        elif kind == b"IEND":
            if length or not seen_idat or end != len(data):
                raise EnhancerError("Invalid PNG end marker or trailing data.")
            ended = True
            break
        elif kind in (b"acTL", b"fcTL", b"fdAT"):
            raise EnhancerError("Animated PNG is unsupported. Export a single frame before importing it.")
        elif kind in (b"cICP", b"mDCV", b"cLLI"):
            raise EnhancerError("PNG HDR/video color metadata has no validated round-trip transform here; it must not be silently treated as sRGB.")
        elif not kind[0] & 32:
            raise EnhancerError("PNG has an unknown critical chunk; compatibility cannot be guaranteed.")
        elif kind in (b"iCCP", b"sRGB", b"gAMA", b"cHRM", b"pHYs", b"bKGD") or kind[3] & 32:
            if kind not in (b"eXIf", b"sBIT"):
                saved.append((kind, payload))
        position = end
    if not ended:
        raise EnhancerError("PNG is missing its end marker.")
    if color == 3 and (not palette or len(palette) // 3 > 1 << depth or len(transparency) > len(palette) // 3):
        raise EnhancerError("PNG palette exceeds its original bit depth.")
    if transparency and (color in (4, 6) or color == 0 and len(transparency) != 2 or color == 2 and len(transparency) != 6):
        raise EnhancerError("Invalid PNG color-key transparency.")
    return PNG(width, height, depth, color, interlace, palette, transparency, tuple(saved))


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _samples(image: Image.Image, info: PNG) -> np.ndarray:
    rgba = rgba_array(image)
    if info.color == 3:
        count = len(info.palette) // 3
        palette = np.full((count, 4), 255, dtype=np.uint8)
        palette[:, :3] = np.frombuffer(info.palette, dtype=np.uint8).reshape(count, 3)
        if info.transparency:
            palette[:len(info.transparency), 3] = np.frombuffer(info.transparency, dtype=np.uint8)
        return palette_indices(image, palette)[:, :, None]
    if info.color in (0, 4):
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        maximum = (1 << info.depth) - 1
        gray = ((gray.astype(np.uint16) * maximum + 127) // 255).astype(np.uint8)
        if info.transparency:
            key = struct.unpack(">H", info.transparency)[0]
            if key > maximum:
                raise EnhancerError("PNG grayscale transparency key exceeds the bit depth.")
            transparent = rgba[:, :, 3] < 128
            collision = (gray == key) & ~transparent
            gray = gray.copy()
            gray[transparent] = key
            gray[collision] = key - 1 if key else 1
        return np.dstack((gray, rgba[:, :, 3])) if info.color == 4 else gray[:, :, None]
    if info.color == 6:
        return rgba
    rgb = rgba[:, :, :3].copy()
    if info.transparency:
        key = np.asarray(struct.unpack(">3H", info.transparency))
        if np.any(key > 255):
            raise EnhancerError("PNG RGB transparency key exceeds the bit depth.")
        transparent = rgba[:, :, 3] < 128
        collision = np.all(rgb == key, axis=2) & ~transparent
        rgb[transparent] = key
        rgb[collision, 0] = int(key[0]) - 1 if key[0] else 1
    return rgb


def encode(image: Image.Image, info: PNG) -> bytes:
    check_size(*image.size, output=True)
    samples = _samples(image, info)
    result = io.BytesIO()
    result.write(SIGNATURE)
    result.write(chunk(b"IHDR", struct.pack(">IIBBBBB", image.width, image.height, info.depth, info.color, 0, 0, info.interlace)))
    # color information precedes the palette; dependent bKGD follows it
    for kind, payload in info.chunks:
        if kind != b"bKGD":
            result.write(chunk(kind, payload))
    if info.palette:
        result.write(chunk(b"PLTE", info.palette))
    if info.transparency:
        result.write(chunk(b"tRNS", info.transparency))
    for kind, payload in info.chunks:
        if kind == b"bKGD":
            result.write(chunk(kind, payload))
    compressor = zlib.compressobj(6)
    pending = bytearray()
    for x, y, step_x, step_y in PASSES if info.interlace else ((0, 0, 1, 1),):
        sub = samples[y::step_y, x::step_x]
        if not sub.shape[0] or not sub.shape[1]:
            continue
        for row in sub:
            if info.depth == 8:
                raw = row.tobytes()
            else:
                flat = row[:, 0]
                per_byte = 8 // info.depth
                packed = np.zeros((len(flat) + per_byte - 1) // per_byte, dtype=np.uint8)
                for offset in range(per_byte):
                    value = flat[offset::per_byte]
                    packed[:len(value)] |= value << (8 - info.depth * (offset + 1))
                raw = packed.tobytes()
            pending.extend(compressor.compress(b"\0" + raw))
            if len(pending) >= 1_048_576:
                result.write(chunk(b"IDAT", bytes(pending)))
                pending.clear()
    pending.extend(compressor.flush())
    result.write(chunk(b"IDAT", bytes(pending)))
    result.write(chunk(b"IEND", b""))
    output = result.getvalue()
    verified = inspect(output, output=True)
    if (verified.depth, verified.color, verified.interlace, verified.palette, verified.transparency) != (info.depth, info.color, info.interlace, info.palette, info.transparency):
        raise EnhancerError("PNG export did not retain its original compatibility layout.")
    return output
