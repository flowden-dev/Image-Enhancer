# Format support

**Retain original** uses the source file’s extension and supported encoding. If you explicitly select the same format, the app keeps its encoding and uses the standard lowercase extension. When the source and output formats differ, the app uses the conversion settings below.

| Output | Retained from a matching source | Conversion from another format |
| --- | --- | --- |
| JPG / JPEG | Color mode, supported chroma sampling, quantization tables, sequential or progressive coding, ICC and supported EXIF | RGB JPEG, quality 95, 4:4:4 chroma, `.jpg`. Transparent artwork is composited on white. |
| PNG | Supported color type and bit depth, palette, transparency, interlacing and color metadata | Lossless 8-bit RGBA |
| TGA | Pixel depth, raw or RLE storage, palette, orientation and supported extension data | Uncompressed 32-bit true color with alpha |
| DDS | Supported legacy or DX10 descriptor, channel masks, compression, alpha and mipmap policy | Uncompressed 32-bit RGBA, legacy header, one mip level |

JPG and JPEG are two extensions for the same format. Native JPEG retains its stored pixel orientation and EXIF tag. Conversion into another format applies the EXIF rotation or flip to the pixels. A quarter-turn swaps the final width and height. New formats use working color values and omit source-specific metadata. When saving a data texture as JPG, only its RGB channels are written.

JPEG and block-compressed DDS are lossy. Indexed or low-bit-depth originals map back to their palette or precision. The **Saved** preview shows the result of encoding.

## Supported layouts

- **JPEG** supports 8-bit grayscale, common RGB sampling, and supported CMYK. Arithmetic or lossless coding, unusual sampling and YCCK are rejected.
- **PNG** supports single-frame 1/2/4/8-bit layouts, palette transparency and Adam7. The encoder rejects APNG, 16-bit samples and unsupported HDR information.
- **TGA** supports the validated true-color, grayscale and indexed variants, including RLE. Private developer directories are unsupported.
- **DDS** supports single 2D textures with supported raw channel masks, DXT1–5, unsigned BC4/BC5 and supported DX10 BC1–5 layouts. Arrays, cubemaps, volumes, BC6/7 and signed or HDR variants are unsupported. Full mip chains regenerate to 1 × 1. Partial chains keep their level count where possible.

Use **Faithful resize** or **Pixel art** for channel data such as BC4/BC5 textures. For native output, supported ICC profiles are converted to sRGB for editing and restored on export. PNG custom calibration without ICC and TGA explicit gamma or color-correction tables require local processing.

Inputs are limited to 128 MiB and 16,777,216 pixels. Outputs support up to 16,384 pixels on an edge and 150,994,944 pixels total.
