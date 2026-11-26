"""Format-aware loading and export; no extension-only conversions."""
from __future__ import annotations

import hashlib
import io
import struct
import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from PIL import Image, JpegImagePlugin

from ..errors import EnhancerError
from .common import MAX_OUTPUT_FILE_BYTES, Profile, check_size, from_srgb, open_output_image, to_srgb
from . import dds, tga, png

EXTENSIONS = {".dds": "DDS", ".tga": "TGA", ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
OUTPUT_SUFFIXES = {"PNG": ".png", "JPEG": ".jpg", "TGA": ".tga", "DDS": ".dds"}
_ORIENTATIONS = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                 4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}


def output_name(source: Path, choice: str) -> str:
    return source.name if choice == "original" else source.with_suffix(OUTPUT_SUFFIXES[choice]).name


@lru_cache(maxsize=4)
def _standard_profile(format_name: str) -> Profile:
    # Obtain a real, validated codec descriptor using the same reader as source files.
    seed = Image.new("RGB" if format_name == "JPEG" else "RGBA", (1, 1))
    stream = io.BytesIO()
    options = {"quality": 95, "subsampling": 0} if format_name == "JPEG" else {}
    seed.save(stream, format=format_name, **options)
    image, profile = decode_source(stream.getvalue(), OUTPUT_SUFFIXES[format_name])
    image.close()
    return profile


def output_profile(source: Profile, choice: str) -> Profile:
    if choice == "original" or choice == source.format:
        return source
    if choice not in OUTPUT_SUFFIXES:
        raise EnhancerError("Choose an available output format.")
    target = _standard_profile(choice)
    orientation = source.metadata.get("orientation", 1)
    if orientation in _ORIENTATIONS or (choice == "JPEG" and source.data_texture):
        return replace(target, metadata={**target.metadata, "bake_orientation": orientation,
                                         "data_alpha": source.data_texture})
    return target


def encoded_size(profile: Profile, working_size: tuple[int, int]) -> tuple[int, int]:
    return working_size[::-1] if profile.metadata.get("bake_orientation") in (5, 6, 7, 8) else working_size


def orient_for_export(image: Image.Image, profile: Profile, *, inverse=False) -> Image.Image:
    tag = profile.metadata.get("bake_orientation")
    if inverse and tag in (6, 8):
        tag = 14 - tag
    return image.transpose(_ORIENTATIONS[tag]) if tag in _ORIENTATIONS else image


def _jpeg_sof(data: bytes, *, with_offset: bool = False):
    if data[:2] != b"\xff\xd8":
        raise EnhancerError("Invalid JPEG signature.")
    position = 2
    while position < len(data):
        if data[position] != 255:
            raise EnhancerError("Invalid JPEG marker sequence.")
        while position < len(data) and data[position] == 255:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker == 0xDA:
            break
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            break
        length = struct.unpack_from(">H", data, position)[0]
        if length < 2 or position + length > len(data):
            break
        if marker in (0xC0, 0xC1, 0xC2):
            if length < 8 or data[position + 2] != 8:
                raise EnhancerError("Only 8-bit sequential/progressive JPEG is supported.")
            return (marker, position - 1) if with_offset else marker
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            raise EnhancerError("Arithmetic, lossless or nonstandard JPEG coding is not supported.")
        position += length
    raise EnhancerError("JPEG is missing a supported frame header.")


def _jpeg_metadata(opened: Image.Image, data: bytes) -> dict:
    sof = _jpeg_sof(data)
    tables = getattr(opened, "quantization", None)
    if not tables or any(len(values) != 64 for values in tables.values()):
        raise EnhancerError("JPEG has no valid quantization tables.")
    if opened.mode not in ("L", "RGB", "CMYK"):
        raise EnhancerError("Unsupported JPEG color mode.")
    sampling = JpegImagePlugin.get_sampling(opened)
    layers = getattr(opened, "layer", ())
    keep_rgb = opened.mode == "RGB" and tuple(layer[0] for layer in layers) == (82, 71, 66)
    if opened.mode == "RGB" and sampling not in (0, 1, 2):
        raise EnhancerError("JPEG uses unusual chroma sampling which cannot be retained by this encoder.")
    if keep_rgb and sampling != 0:
        raise EnhancerError("RGB JPEG with chroma subsampling cannot be retained safely.")
    if opened.mode == "CMYK" and any(layer[1:3] != (1, 1) for layer in layers):
        raise EnhancerError("CMYK JPEG with chroma subsampling cannot be retained safely.")
    if opened.info.get("adobe_transform") == 2:
        raise EnhancerError("YCCK JPEG cannot be re-encoded identically as CMYK. Convert explicitly to RGB first.")
    orientation = opened.getexif().get(274, 1)
    return {"qtables": {k: list(v) for k, v in tables.items()}, "sampling": sampling,
            "progressive": sof == 0xC2, "sof": sof, "keep_rgb": keep_rgb,
            "exif": opened.info.get("exif", b""), "orientation": orientation if type(orientation) is int else 1,
            "dpi": opened.info.get("dpi"),
            "comment": opened.info.get("comment", b"")}


def decode_source(data: bytes, extension: str, *, output_size: tuple[int, int] | None = None) -> tuple[Image.Image, Profile]:
    extension = extension.lower()
    expected = EXTENSIONS.get(extension)
    if expected is None:
        raise EnhancerError("Supported inputs for original-format export: DDS, TGA, PNG and JPG/JPEG.")
    if expected == "DDS":
        info = dds.inspect(data, output=output_size is not None)
        if output_size and (info.width, info.height) != output_size:
            raise EnhancerError("Saved DDS dimensions have changed.")
        image = dds.decode(data, info)
        notes = [info.description]
        if info.mipmaps > 1:
            notes.append("Mipmaps are regenerated from the edited top level; full chains stay full and partial chains retain their count.")
        if info.data_texture:
            notes.append("This is a channel/data texture. Use local scaling; generative editing is disabled to avoid inventing channel values.")
        return image, Profile("DDS", "RGBA", image.size, info, notes=tuple(notes))
    if expected == "TGA":
        info = tga.inspect(data, output=output_size is not None)
        if output_size and (info.width, info.height) != output_size:
            raise EnhancerError("Saved TGA dimensions have changed.")
        image = tga.decode(data, info)
        notes = [info.description]
        if info.indexed:
            notes.append("The original palette is retained; enhanced colors are mapped back to that palette.")
        if info.data_texture:
            notes.append("The TGA attribute channel is data, not transparency. Use local scaling to preserve the channels independently.")
        if info.extension:
            notes.append("TGA extension/color-correction data is retained; stale thumbnail and scan-line pointers are cleared.")
        return image, Profile("TGA", "RGBA", image.size, info, notes=tuple(notes))
    info = png.inspect(data, output=output_size is not None) if expected == "PNG" else None
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with (open_output_image(data, output_size) if output_size else Image.open(io.BytesIO(data))) as opened:
            if opened.format != expected:
                raise EnhancerError("The file extension does not match its actual image format. Rename it correctly first.")
            if getattr(opened, "n_frames", 1) != 1:
                raise EnhancerError("Animated or multi-frame images cannot be flattened safely.")
            check_size(*opened.size, output=output_size is not None)
            metadata = _jpeg_metadata(opened, data) if expected == "JPEG" else {}
            opened.load()
            notes = []
            if expected == "JPEG":
                notes.append("JPEG is re-encoded with the original color mode, sampling, quantization tables and progressive/sequential coding. Recompression is lossy.")
                if opened.getexif().get(274, 1) != 1:
                    notes.append("Stored pixel orientation and EXIF orientation are retained; pixels stay in their stored orientation.")
                if opened.mode == "CMYK" and not opened.info.get("icc_profile"):
                    notes.append("Unprofiled CMYK uses Pillow's RGB conversion; exact colorimetric equivalence is not promised.")
            elif info and info.color == 3:
                notes.append("Original PNG palette and transparency table are retained; enhanced colors are quantized to that palette.")
            if opened.info.get("icc_profile"):
                notes.append("The embedded ICC color profile is converted to sRGB for editing and restored on native export.")
            profile = Profile(expected, opened.mode, opened.size, info,
                              opened.info.get("icc_profile", b""), tuple(notes), metadata)
            # color management is explicit; retain stored pixel coordinates when EXIF orientation is present
            image = to_srgb(opened, profile)
            image.info.clear()
    # validate the reverse color transform before the user can spend API credit
    if profile.icc:
        from_srgb(image.resize((1, 1)), profile)
    return image, profile


def _saved_bytes(path: Path) -> bytes:
    # Our own exports may exceed the smaller input pixel and byte limits.
    if path.stat().st_size > MAX_OUTPUT_FILE_BYTES:
        raise EnhancerError("The saved file exceeds the output size limit.")
    with path.open("rb") as stream:
        data = stream.read(MAX_OUTPUT_FILE_BYTES + 1)
    if len(data) > MAX_OUTPUT_FILE_BYTES:
        raise EnhancerError("The saved file exceeds the output size limit.")
    return data


@dataclass(frozen=True)
class SavedImage:
    path: Path
    image: Image.Image
    digest: bytes

    def verified_bytes(self) -> bytes:
        data = _saved_bytes(self.path)
        if hashlib.sha256(data).digest() != self.digest:
            raise EnhancerError("The saved output changed. Reopen it before saving a copy.")
        return data


def read_saved_export(path: Path, size: tuple[int, int]) -> SavedImage:
    data = _saved_bytes(path)
    image, _ = decode_source(data, path.suffix, output_size=size)
    return SavedImage(path, image, hashlib.sha256(data).digest())


def _jpeg_encode(image: Image.Image, profile: Profile) -> bytes:
    info = profile.metadata
    options = {"qtables": info["qtables"], "progressive": info["progressive"], "optimize": False,
               "icc_profile": profile.icc}
    if profile.mode == "RGB":
        options["subsampling"] = info["sampling"]
        if info["keep_rgb"]:
            options["keep_rgb"] = True
    if info.get("dpi"):
        options["dpi"] = info["dpi"]
    if info.get("comment"):
        options["comment"] = info["comment"]
    if info.get("exif"):
        exif = Image.Exif()
        exif.load(info["exif"])
        for tag, value in ((256, image.width), (257, image.height)):
            if tag in exif:
                exif[tag] = value
        if 34665 in exif:
            sub = exif.get_ifd(34665)
            for tag, value in ((40962, image.width), (40963, image.height)):
                if tag in sub:
                    sub[tag] = value
        options["exif"] = exif.tobytes()
    buffer = io.BytesIO()
    image.convert(profile.mode).save(buffer, format="JPEG", **options)
    result = buffer.getvalue()
    if info["sof"] == 0xC1 and _jpeg_sof(result) == 0xC0:
        # SOF1 is extended sequential; an 8-bit SOF0 stream is valid with this marker
        _, marker_offset = _jpeg_sof(result, with_offset=True)
        result = result[:marker_offset] + b"\xc1" + result[marker_offset + 1:]
    return result


def encode_original(image: Image.Image, profile: Profile) -> bytes:
    image = orient_for_export(image, profile)
    check_size(*image.size, output=True)
    if profile.format == "JPEG" and "A" in image.getbands() and not profile.metadata.get("data_alpha"):
        alpha = image.getchannel("A")
        if alpha.getextrema()[0] < 255:
            matte = Image.new("RGB", image.size, "white")
            matte.paste(image, (0, 0), alpha)
            image = matte
    restored = from_srgb(image, profile) if profile.icc else image
    if profile.format == "DDS":
        data = dds.encode(restored, profile.details)
    elif profile.format == "TGA":
        data = tga.encode(restored, profile.details)
    elif profile.format == "PNG":
        data = png.encode(restored, profile.details)
    elif profile.format == "JPEG":
        data = _jpeg_encode(restored, profile)
    else:
        raise EnhancerError("This source format has no original-format exporter.")
    verify_export(data, profile, image.size)
    return data


def verify_export(data: bytes, original: Profile, target_size: tuple[int, int]) -> None:
    if original.format == "DDS":
        info = dds.inspect(data, output=True)
        actual_size = (info.width, info.height)
        # parse all mip lengths and decode the top level independently of the writer
        decoded = dds.decode(data, info)
        if info.header[76:108] != original.details.header[76:108] or info.header[128:] != original.details.header[128:]:
            raise EnhancerError("DDS output changed its format/header compatibility.")
        if info.mipmaps != original.details.output_mipmaps(target_size):
            raise EnhancerError("DDS output has an incorrect mip count.")
    elif original.format == "TGA":
        info = tga.inspect(data, output=True)
        decoded = tga.decode(data, info)
        actual_size = decoded.size
        if info.header[:12] != original.details.header[:12] or info.header[16:] != original.details.header[16:]:
            raise EnhancerError("TGA output changed its compatibility descriptor.")
    else:
        with open_output_image(data, target_size) as check:
            check.load()
            actual_size = check.size
            if check.format != original.format:
                raise EnhancerError("Exported image has the wrong format.")
            if original.format == "PNG":
                info = png.inspect(data, output=True)
                source = original.details
                if (info.depth, info.color, info.interlace, info.palette, info.transparency) != (source.depth, source.color, source.interlace, source.palette, source.transparency):
                    raise EnhancerError("PNG output changed its color/palette compatibility.")
            else:
                result = _jpeg_metadata(check, data)
                if check.mode != original.mode or any(result[k] != original.metadata[k] for k in ("qtables", "sampling", "progressive", "keep_rgb", "sof")):
                    raise EnhancerError("JPEG output changed its coding, sampling or quantization.")
    if actual_size != target_size:
        raise EnhancerError("Exported image dimensions do not match the requested scale.")


def preflight(profile: Profile, image: Image.Image, *, method: str = "openai", output_format: str = "original") -> None:
    if profile.data_texture and method == "neural":
        raise EnhancerError("Neural artwork upscaling cannot process channel/data textures. Choose Faithful resize or Pixel art.")
    if profile.data_texture and method == "openai":
        raise EnhancerError("Generative editing is disabled for BC4/BC5, alpha-only, RG and TGA attribute data textures. Choose a local scaling method to retain channel semantics.")
    if method == "openai" and profile.format == "PNG" and not profile.icc:
        chunks = dict(profile.details.chunks)
        # these tags describe sample values, not just decorative metadata
        if b"sRGB" not in chunks:
            gamma = chunks.get(b"gAMA")
            chroma = chunks.get(b"cHRM")
            srgb_chroma = struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)
            if gamma is not None and gamma != struct.pack(">I", 45455) or chroma is not None and chroma != srgb_chroma:
                raise EnhancerError("This PNG has custom color calibration without an ICC profile. Use local scaling; its original color metadata will be retained. AI editing requires a supported ICC or sRGB source.")
    if method == "openai" and profile.format == "TGA" and profile.details.extension:
        info = profile.details
        numerator, denominator = struct.unpack_from("<HH", info.extension, 478)
        if info.color_table or numerator and denominator:
            raise EnhancerError("This TGA uses explicit gamma or a color-correction table. Use local scaling to retain that calibration; generative editing cannot restore it safely.")
    # exercise the complete local encoder/decoder path before a paid request
    encode_original(image.resize((8, 8), Image.Resampling.NEAREST), output_profile(profile, output_format))
