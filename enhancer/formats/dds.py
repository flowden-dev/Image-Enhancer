"""DDS 2D texture codec: retain original pixel descriptors, headers and mip policy.

Pillow supplies the BC1/BC2/BC3 block codec. Container headers, byte masks,
row pitch and mip chains are explicitly implemented and checked here.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..errors import EnhancerError
from .common import check_size, open_output_image, premultiply, unpremultiply, rgba_array, resize_artwork, restore_hidden_rgb

BLOCKS = {b"DXT1": ("DXT1", 8), b"DXT2": ("DXT3", 16), b"DXT3": ("DXT3", 16),
          b"DXT4": ("DXT5", 16), b"DXT5": ("DXT5", 16), b"ATI1": ("BC4", 8),
          b"BC4U": ("BC4", 8), b"ATI2": ("BC5", 16), b"BC5U": ("BC5", 16)}
DXGI_BLOCKS = {70: ("DXT1", 8), 71: ("DXT1", 8), 72: ("DXT1", 8),
               73: ("DXT3", 16), 74: ("DXT3", 16), 75: ("DXT3", 16),
               76: ("DXT5", 16), 77: ("DXT5", 16), 78: ("DXT5", 16),
               79: ("BC4", 8), 80: ("BC4", 8), 82: ("BC5", 16), 83: ("BC5", 16)}
DXGI_RAW = {
    27: (32, (255, 65280, 16711680, 4278190080), "rgb"),
    28: (32, (255, 65280, 16711680, 4278190080), "rgb"),
    29: (32, (255, 65280, 16711680, 4278190080), "rgb"),
    87: (32, (16711680, 65280, 255, 4278190080), "rgb"),
    90: (32, (16711680, 65280, 255, 4278190080), "rgb"),
    91: (32, (16711680, 65280, 255, 4278190080), "rgb"),
    88: (32, (16711680, 65280, 255, 0), "rgb"),
    92: (32, (16711680, 65280, 255, 0), "rgb"),
    93: (32, (16711680, 65280, 255, 0), "rgb"),
    85: (16, (0xF800, 0x07E0, 0x001F, 0), "rgb"),
    86: (16, (0x7C00, 0x03E0, 0x001F, 0x8000), "rgb"),
    115: (16, (0x0F00, 0x00F0, 0x000F, 0xF000), "rgb"),
    61: (8, (255, 0, 0, 0), "luminance"),
    65: (8, (0, 0, 0, 255), "alpha"),
    49: (16, (255, 65280, 0, 0), "rg"),
}


@dataclass(frozen=True)
class DDS:
    header: bytes
    width: int
    height: int
    mipmaps: int
    codec: str
    block_bytes: int
    bits: int
    masks: tuple[int, int, int, int]
    kind: str
    row_alignment: int
    premultiplied: bool
    srgb: bool
    opaque_alpha: bool = False

    @property
    def description(self) -> str:
        name = self.header[84:88].decode("ascii", "replace").rstrip("\x00") if self.block_bytes else f"{self.bits}-bit {self.kind}"
        if self.header[84:88] == b"DX10":
            name = f"DXGI {struct.unpack_from('<I', self.header, 128)[0]} ({self.codec or name})"
        return f"DDS {name}, {'DX10' if len(self.header) == 148 else 'legacy'} header, {self.mipmaps} mip level(s)" + (", premultiplied alpha" if self.premultiplied else "")

    @property
    def data_texture(self) -> bool:
        return self.codec in ("BC4", "BC5") or self.kind in ("alpha", "rg")

    def output_mipmaps(self, size: tuple[int, int]) -> int:
        # no mips stays no mips; complete chains stay complete; partial chains retain count
        if self.mipmaps <= 1:
            return 1
        if self.mipmaps == max(self.width, self.height).bit_length():
            return max(size).bit_length()
        return min(self.mipmaps, max(size).bit_length())

    def level_size(self, width: int, height: int) -> int:
        if self.block_bytes:
            return ((width + 3) // 4) * ((height + 3) // 4) * self.block_bytes
        pitch = (width * self.bits + 7) // 8
        pitch = (pitch + self.row_alignment - 1) // self.row_alignment * self.row_alignment
        return pitch * height


def _mask_check(masks: tuple[int, ...], bits: int, kind: str) -> None:
    used = 0
    for index, mask in enumerate(masks):
        if not mask:
            continue
        if mask >= 1 << bits:
            raise EnhancerError("DDS channel mask extends outside the pixel.")
        shift = (mask & -mask).bit_length() - 1
        value = mask >> shift
        if value & (value + 1) or value.bit_length() > 8:
            raise EnhancerError("DDS requires contiguous channels of at most 8 bits; HDR/high-precision data is not color artwork.")
        if used & mask and not (kind == "luminance" and index in (1, 2)):
            raise EnhancerError("DDS has overlapping channel masks.")
        used |= mask


def inspect(data: bytes, *, output: bool = False) -> DDS:
    if len(data) < 128 or data[:4] != b"DDS " or struct.unpack_from("<I", data, 4)[0] != 124 or struct.unpack_from("<I", data, 76)[0] != 32:
        raise EnhancerError("Invalid or truncated DDS header.")
    flags, height, width, pitch, depth, levels = struct.unpack_from("<6I", data, 8)
    check_size(width, height, output=output)
    caps2 = struct.unpack_from("<I", data, 112)[0]
    if caps2 & (0xFE00 | 0x200000) or depth > 1:
        raise EnhancerError("DDS cubemaps and volume textures cannot be flattened safely; only single 2D textures are supported.")
    levels = max(1, levels)
    if levels > max(width, height).bit_length():
        raise EnhancerError("DDS mip count exceeds its dimensions.")
    pflags, fourcc_int, bits, *masks_list = struct.unpack_from("<7I", data, 80)
    # bump, YUV, palette and contradictory flags must not masquerade as color artwork
    allowed_flags = 0x1 | 0x2 | 0x4 | 0x40 | 0x20000
    base_flags = pflags & (0x2 | 0x4 | 0x40 | 0x20000)
    if pflags & ~allowed_flags or not base_flags or base_flags & (base_flags - 1):
        raise EnhancerError("DDS uses unsupported or contradictory pixel format flags.")
    fourcc = struct.pack("<I", fourcc_int)
    header_size, codec, block_bytes, kind = 128, "", 0, ""
    masks = tuple(masks_list)
    premul = fourcc in (b"DXT2", b"DXT4")
    srgb = opaque = False
    if pflags & 4:
        if pflags & (0x40 | 0x20000 | 0x20):
            raise EnhancerError("DDS has contradictory pixel format flags.")
        if fourcc == b"DX10":
            if len(data) < 148:
                raise EnhancerError("Truncated DX10 DDS header.")
            dxgi, dimension, misc, array_size, alpha = struct.unpack_from("<5I", data, 128)
            if dimension != 3 or misc & 4 or array_size != 1:
                raise EnhancerError("DDS arrays, cubemaps and volumes are not single 2D textures.")
            if (alpha & 7) > 3 or alpha & ~7:
                raise EnhancerError("DDS custom/reserved alpha semantics cannot be safely processed.")
            header_size, premul, opaque = 148, (alpha & 7) == 2, (alpha & 7) == 3
            srgb = dxgi in (29, 72, 75, 78, 91, 93)
            if dxgi in DXGI_BLOCKS:
                codec, block_bytes = DXGI_BLOCKS[dxgi]
            elif dxgi in DXGI_RAW:
                bits, masks, kind = DXGI_RAW[dxgi]
            else:
                raise EnhancerError(f"DDS DXGI format {dxgi} has no validated round-trip encoder here. BC6/BC7, signed, HDR and data formats are not silently converted.")
        elif fourcc in BLOCKS:
            codec, block_bytes = BLOCKS[fourcc]
        else:
            raise EnhancerError(f"DDS FourCC {fourcc!r} has no validated round-trip encoder.")
    elif pflags & 0x40:
        kind = "rgb"
    elif pflags & 0x20000:
        kind = "luminance"
    elif pflags & 2:
        kind = "alpha"
    else:
        raise EnhancerError("DDS palette or unknown pixel layout is not supported for safe export.")
    alignment = 1
    if not block_bytes:
        if bits not in (8, 16, 24, 32):
            raise EnhancerError("Only 8/16/24/32-bit uncompressed DDS color layouts are supported.")
        # alpha masks without alpha flags are unused X bits in legacy DDS
        if header_size == 128 and not (pflags & (1 | 2)):
            masks = masks[:3] + (0,)
        _mask_check(masks, bits, kind)
        if kind == "rgb" and not all(masks[:3]) or kind == "luminance" and not masks[0] or kind == "alpha" and not masks[3]:
            raise EnhancerError("DDS is missing a required channel mask.")
        tight_pitch = width * (bits // 8)
        dword_pitch = (tight_pitch + 3) & ~3
        if flags & 8 and pitch not in (0, tight_pitch, dword_pitch):
            raise EnhancerError("DDS uses an unsupported row pitch; refusing to shift its pixels.")
        def expected(align: int) -> int:
            return sum((((max(1, width >> i) * bits // 8 + align - 1) // align) * align) * max(1, height >> i) for i in range(levels))
        available = len(data) - header_size
        if available == expected(4) and (available != expected(1) or pitch != tight_pitch):
            alignment = 4
        elif available != expected(1):
            raise EnhancerError("DDS pixel/mipmap data is truncated or its row layout is unsupported.")
    profile = DDS(data[:header_size], width, height, levels, codec, block_bytes, bits, masks, kind, alignment, premul, srgb, opaque)
    expected_size = header_size + sum(profile.level_size(max(1, width >> i), max(1, height >> i)) for i in range(levels))
    if len(data) != expected_size:
        raise EnhancerError("DDS pixel/mipmap data is truncated or has unexpected trailing surfaces/data.")
    return profile


def _bcn_decode(payload: bytes, width: int, height: int, codec: str) -> Image.Image:
    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<7I", header, 4, 124, 0x81007, height, width, len(payload), 0, 1)
    struct.pack_into("<II4sI", header, 76, 32, 4, codec.encode("ascii"), 0)
    struct.pack_into("<I", header, 108, 0x1000)
    with open_output_image(header + payload, (width, height)) as opened:
        opened.load()
        return opened.convert("RGBA")


def decode(data: bytes, info: DDS) -> Image.Image:
    payload = memoryview(data)[len(info.header):len(info.header) + info.level_size(info.width, info.height)]
    if info.codec in ("BC4", "BC5"):
        blocks = np.frombuffer(payload, dtype=np.uint8).reshape(-1, info.block_bytes)
        channels = []
        for offset in range(0, info.block_bytes, 8):
            expanded = np.zeros((len(blocks), 16), dtype=np.uint8)
            expanded[:, :8] = blocks[:, offset:offset + 8]
            channels.append(_bcn_decode(expanded.tobytes(), info.width, info.height, "DXT5").getchannel("A"))
        if len(channels) == 1:
            image = channels[0].convert("RGBA")
        else:
            image = Image.merge("RGB", (channels[0], channels[1], Image.new("L", (info.width, info.height), 0))).convert("RGBA")
    elif info.block_bytes:
        image = _bcn_decode(payload, info.width, info.height, info.codec)
    else:
        stride = len(payload) // info.height
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(info.height, stride)[:, :info.width * (info.bits // 8)]
        image = Image.new("RGBA", (info.width, info.height))
        for top in range(0, info.height, 64):
            band = raw[top:top + 64].reshape(-1, info.width, info.bits // 8)
            values = np.zeros(band.shape[:2], dtype=np.uint32)
            for byte in range(info.bits // 8):
                values |= band[:, :, byte].astype(np.uint32) << (8 * byte)
            pixels = np.zeros((*values.shape, 4), dtype=np.uint8)
            pixels[:, :, 3] = 255
            if info.kind == "alpha":
                pixels[:, :, :3] = 255
            for channel, mask in enumerate(info.masks):
                if not mask:
                    continue
                shift = (mask & -mask).bit_length() - 1
                maximum = mask >> shift
                pixels[:, :, channel] = (((values & mask) >> shift) * 255 + maximum // 2) // maximum
            if info.kind == "luminance":
                pixels[:, :, 1] = pixels[:, :, 2] = pixels[:, :, 0]
            image.paste(Image.fromarray(pixels), (0, top))
    if info.opaque_alpha:
        image.putalpha(255)
    return unpremultiply(image) if info.premultiplied else image


def _compress(image: Image.Image, codec: str) -> bytes:
    stream = io.BytesIO()
    image.convert("RGBA").save(stream, format="DDS", pixel_format=codec)
    # discard Pillow's container; its mip and pitch defaults are not appropriate here
    return stream.getvalue()[128:]


def _encode_level(image: Image.Image, info: DDS) -> bytes:
    if info.premultiplied:
        image = premultiply(image)
    if info.opaque_alpha:
        image = image.convert("RGBA")
        image.putalpha(255)
    if info.codec in ("BC4", "BC5"):
        channels = (image.getchannel("R"),) if info.codec == "BC4" else (image.getchannel("R"), image.getchannel("G"))
        blocks = []
        for channel in channels:
            carrier = Image.new("RGBA", image.size, (0, 0, 0, 255))
            carrier.putalpha(channel)
            blocks.append(np.frombuffer(_compress(carrier, "DXT5"), dtype=np.uint8).reshape(-1, 16)[:, :8])
        return np.concatenate(blocks, axis=1).tobytes()
    if info.block_bytes:
        return _compress(image, info.codec)
    pitch = info.level_size(image.width, 1)
    stream = io.BytesIO()
    for top in range(0, image.height, 64):
        band = image.crop((0, top, image.width, min(top + 64, image.height)))
        pixels = rgba_array(band)
        values = np.zeros((band.height, band.width), dtype=np.uint32)
        for channel, mask in enumerate(info.masks):
            if not mask or info.kind == "luminance" and channel in (1, 2):
                continue
            samples = np.asarray(band.convert("L")) if info.kind == "luminance" and channel == 0 else pixels[:, :, channel]
            shift = (mask & -mask).bit_length() - 1
            maximum = mask >> shift
            values |= ((samples.astype(np.uint32) * maximum + 127) // 255) << shift
        raw = np.zeros((band.height, pitch), dtype=np.uint8)
        target = raw[:, :band.width * (info.bits // 8)].reshape(band.height, band.width, info.bits // 8)
        for byte in range(info.bits // 8):
            target[:, :, byte] = (values >> (8 * byte)) & 255
        stream.write(raw)
    return stream.getvalue()


def _mip(image: Image.Image, size: tuple[int, int], srgb: bool) -> Image.Image:
    if not srgb:
        return resize_artwork(image, size)
    # Filter one float plane at a time. Four full-size float channels can exhaust RAM at 12K.
    unit = np.arange(256, dtype=np.float32) / 255.0
    linear = np.where(unit <= 0.04045, unit / 12.92, ((unit + 0.055) / 1.055) ** 2.4)
    alpha = image.getchannel("A").point(unit.tolist(), mode="F")
    alpha_small = np.asarray(alpha.resize(size, Image.Resampling.LANCZOS)).clip(0, 1)
    alpha.close()
    alpha_plane = Image.fromarray(np.rint(alpha_small * 255).astype(np.uint8))
    divisor = np.maximum(alpha_small, 1e-8)
    colors = []
    for channel in range(3):
        weighted = Image.new("F", image.size)
        for top in range(0, image.height, 64):
            band = rgba_array(image.crop((0, top, image.width, min(top + 64, image.height))))
            values = linear[band[:, :, channel]] * unit[band[:, :, 3]]
            weighted.paste(Image.fromarray(values), (0, top))
        color = (np.asarray(weighted.resize(size, Image.Resampling.LANCZOS)) / divisor).clip(0, 1)
        weighted.close()
        color = np.where(color <= 0.0031308, color * 12.92, 1.055 * color ** (1 / 2.4) - 0.055)
        colors.append(Image.fromarray(np.rint(color.clip(0, 1) * 255).astype(np.uint8)))
    return restore_hidden_rgb(Image.merge("RGBA", (*colors, alpha_plane)), image)


def encode(image: Image.Image, info: DDS) -> bytes:
    check_size(*image.size, output=True)
    width, height = image.size
    levels = info.output_mipmaps(image.size)
    header = bytearray(info.header)
    flags = struct.unpack_from("<I", header, 8)[0]
    flags = (flags | 0x1007) & ~(0x8 | 0x80000 | 0x20000 | 0x800000)
    flags |= 0x80000 if info.block_bytes else 8
    if levels > 1:
        flags |= 0x20000
    pitch = info.level_size(width, height) if info.block_bytes else info.level_size(width, 1)
    struct.pack_into("<6I", header, 8, flags, height, width, pitch, 0, levels if levels > 1 else struct.unpack_from("<I", info.header, 28)[0])
    caps = struct.unpack_from("<I", header, 108)[0]
    caps = (caps | 0x1000) & ~(0x8 | 0x400000)
    if levels > 1:
        caps |= 0x8 | 0x400000
    struct.pack_into("<I", header, 108, caps)
    result = io.BytesIO()
    result.write(header)
    top = image if image.mode == "RGBA" else image.convert("RGBA")
    for level in range(levels):
        size = (max(1, width >> level), max(1, height >> level))
        mip = top if level == 0 else _mip(top, size, info.srgb)
        encoded = _encode_level(mip, info)
        if len(encoded) != info.level_size(*size):
            raise EnhancerError("DDS encoder produced an unexpected mip length; output was not published.")
        result.write(encoded)
    output = result.getvalue()
    verified = inspect(output, output=True)
    if verified.header[76:108] != info.header[76:108] or verified.header[128:] != info.header[128:] or verified.mipmaps != levels:
        raise EnhancerError("DDS export did not retain its original format descriptor.")
    return output
