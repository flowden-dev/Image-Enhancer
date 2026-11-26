from __future__ import annotations

import io
import os
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from enhancer.core import (
    EnhancerError, MAX_EDGE, MAX_PIXELS, MIN_PIXELS, Settings,
    build_prompt, finish_image, has_transparency, load_texture,
    make_canvas, paths_match, plan_image, prepare_recovery, save_image_atomic,
)


class PlanTests(unittest.TestCase):
    def check_plan(self, size, scale):
        plan = plan_image(size, scale)
        w, h = plan.canvas_size
        self.assertLessEqual(max(w, h), MAX_EDGE)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertLessEqual(max(w, h), 3 * min(w, h))
        self.assertLessEqual(w * h, MAX_PIXELS)
        self.assertGreaterEqual(w * h, MIN_PIXELS)
        x0, y0, x1, y1 = plan.crop
        self.assertTrue(0 <= x0 < x1 <= w)
        self.assertTrue(0 <= y0 < y1 <= h)
        self.assertLessEqual(abs((x1 - x0) * size[1] - (y1 - y0) * size[0]), sum(size) / 2)
        self.assertEqual(plan.target_size, (size[0] * scale, size[1] * scale))
        return plan

    def test_square(self):
        self.assertEqual(self.check_plan((256, 256), 2).target_size, (512, 512))

    def test_smallest(self):
        self.check_plan((1, 1), 2)

    def test_thin_horizontal(self):
        self.check_plan((1024, 1), 2)

    def test_thin_vertical(self):
        self.check_plan((1, 1024), 2)

    def test_odd_dimensions(self):
        self.check_plan((101, 37), 4)

    def test_exact_4k(self):
        self.assertEqual(self.check_plan((1920, 1080), 2).canvas_size, (3840, 2160))

    def test_large_flag(self):
        self.assertTrue(self.check_plan((1024, 1024), 2).experimental)
        self.assertFalse(self.check_plan((256, 256), 2).experimental)

    def test_large_edge_uses_explicit_final_resampling(self):
        plan = self.check_plan((1024, 1024), 4)
        self.assertEqual(plan.target_size, (4096, 4096))
        self.assertTrue(plan.resampled_output)

    def test_large_area_uses_explicit_final_resampling(self):
        self.assertTrue(self.check_plan((1800, 1800), 2).resampled_output)

    def test_padding_overflow_is_fitted_without_changing_target(self):
        self.assertEqual(self.check_plan((1919, 1081), 2).target_size, (3838, 2162))

    def test_invalid_dimensions(self):
        for size in ((0, 1), (-1, 100), (True, 1), (1.5, 2), (1,)):
            with self.subTest(size=size), self.assertRaises(EnhancerError):
                plan_image(size, 2)

    def test_invalid_settings(self):
        for settings in (Settings(scale=0), Settings(scale=True), Settings(scale=0.5), Settings(quality="ultra"),
                         Settings(instruction="x" * 4001), Settings(preserve_alpha="yes")):
            with self.subTest(settings=settings), self.assertRaises(EnhancerError):
                settings.validate()

    def test_seeded_dimension_sweep(self):
        rng = random.Random(77321)
        count = 0
        for _ in range(10000):
            size = (rng.randint(1, 2100), rng.randint(1, 2100))
            scale = rng.choice((2, 4))
            try:
                self.check_plan(size, scale)
                count += 1
            except EnhancerError:
                pass
        self.assertGreater(count, 3000)

    def test_prompt_includes_crop_and_edit_override(self):
        plan = plan_image((80, 20), 2)
        prompt = build_prompt(plan, Settings(instruction='Replace PLAY with SPIELEN', source_protection='off'))
        self.assertIn(plan.size_string, prompt)
        self.assertIn("only exception", prompt)
        self.assertIn("SPIELEN", prompt)


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="image test ü ")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name="source.png", mode="RGBA", color=(150, 80, 30, 128)):
        path = self.root / name
        Image.new(mode, (64, 32), color).save(path)
        return path

    def test_png_load(self):
        texture = load_texture(self.fixture())
        self.assertEqual(texture.image.getpixel((0, 0)), (150, 80, 30, 128))
        self.assertEqual(len(texture.sha256), 64)

    def test_tga_load(self):
        self.assertEqual(load_texture(self.fixture("texture.TGA")).format, "TGA")

    def test_dds_load(self):
        texture = load_texture(self.fixture("texture.dds"))
        self.assertEqual(texture.format, "DDS")
        self.assertEqual(texture.image.size, (64, 32))

    def test_dxt_formats(self):
        for compression in ("DXT1", "DXT3", "DXT5"):
            with self.subTest(compression=compression):
                path = self.root / f"{compression}.dds"
                Image.new("RGBA", (16, 16), (80, 160, 100, 255)).save(path, pixel_format=compression)
                self.assertEqual(load_texture(path).image.size, (16, 16))

    def test_jpeg_load(self):
        self.assertEqual(load_texture(self.fixture("texture.jpg", "RGB", (10, 20, 30))).image.mode, "RGBA")

    def test_bmp_not_advertised_as_supported_native_export(self):
        with self.assertRaises(EnhancerError):
            load_texture(self.fixture("texture.bmp", "RGB", (10, 20, 30)))

    def test_webp_not_advertised_as_supported_native_export(self):
        with self.assertRaises(EnhancerError):
            load_texture(self.fixture("texture.webp"))

    def test_exif_orientation(self):
        image = Image.new("RGB", (16, 32))
        exif = image.getexif()
        exif[274] = 6
        path = self.root / "rotated.jpg"
        image.save(path, exif=exif)
        texture = load_texture(path)
        self.assertEqual(texture.image.size, (16, 32))
        self.assertEqual(texture.image.info, {})

    def test_palette_transparency(self):
        image = Image.new("P", (16, 16), 0)
        image.putpalette([255, 0, 0] + [0, 0, 0] * 255)
        image.info["transparency"] = 0
        path = self.root / "palette.png"
        image.save(path)
        self.assertTrue(has_transparency(load_texture(path).image))

    def test_gray_alpha(self):
        path = self.root / "gray.png"
        Image.new("LA", (16, 16), (150, 64)).save(path)
        self.assertEqual(load_texture(path).image.getpixel((0, 0)), (150, 150, 150, 64))

    def test_unsupported_gif(self):
        with self.assertRaises(EnhancerError):
            load_texture(self.fixture("texture.gif"))

    def test_animated_png(self):
        path = self.root / "animated.png"
        Image.new("RGBA", (16, 16), "red").save(path, save_all=True, append_images=[Image.new("RGBA", (16, 16), "blue")])
        with self.assertRaises(EnhancerError):
            load_texture(path)

    def test_16bit_gray(self):
        path = self.root / "16bit.png"
        Image.new("I;16", (16, 16)).save(path)
        with self.assertRaises(EnhancerError):
            load_texture(path)

    def test_cmyk(self):
        path = self.root / "cmyk.jpg"
        Image.new("CMYK", (16, 16)).save(path)
        self.assertEqual(load_texture(path).profile.mode, "CMYK")

    def test_corrupt_file(self):
        path = self.root / "bad.png"
        path.write_bytes(b"not an image")
        with self.assertRaises(EnhancerError):
            load_texture(path)

    def test_truncated_png(self):
        path = self.fixture()
        path.write_bytes(path.read_bytes()[:55])
        with self.assertRaises(EnhancerError):
            load_texture(path)

    def test_source_byte_limit(self):
        path = self.fixture()
        with patch("enhancer.core.MAX_SOURCE_BYTES", 20), self.assertRaises(EnhancerError):
            load_texture(path)

    def test_source_pixel_limit(self):
        path = self.fixture()
        with patch("enhancer.core.MAX_SOURCE_PIXELS", 100), self.assertRaises(EnhancerError):
            load_texture(path)

    def test_missing_file(self):
        with self.assertRaises(OSError):
            load_texture(self.root / "no.png")

    def test_directory(self):
        with self.assertRaises(EnhancerError):
            load_texture(self.root)

    def test_canvas_exact_content(self):
        image = Image.new("RGBA", (101, 37), (25, 35, 45, 128))
        plan = plan_image(image.size, 2)
        png, alpha = make_canvas(image, plan)
        self.assertTrue(alpha)
        with Image.open(io.BytesIO(png)) as canvas:
            self.assertEqual(canvas.size, plan.canvas_size)
            self.assertEqual(canvas.getpixel(plan.crop[:2])[3], 128)
            if plan.crop[1] > 0:
                self.assertEqual(canvas.getpixel((0, 0))[3], 0)

    def test_fully_transparent_rejected(self):
        image = Image.new("RGBA", (16, 16))
        with self.assertRaises(EnhancerError):
            make_canvas(image, plan_image(image.size, 2))

    def test_source_size_mismatch(self):
        with self.assertRaises(EnhancerError):
            make_canvas(Image.new("RGB", (32, 32)), plan_image((64, 64), 2))

    def test_upload_byte_limit(self):
        image = Image.new("RGB", (16, 16))
        with patch("enhancer.core.MAX_UPLOAD_BYTES", 10), self.assertRaises(EnhancerError):
            make_canvas(image, plan_image(image.size, 2))

    def test_preserves_alpha_exactly_as_resampled(self):
        source = Image.new("RGBA", (8, 4), (100, 80, 20, 0))
        source.putalpha(Image.linear_gradient("L").resize(source.size))
        plan = plan_image(source.size, 2)
        raw = Image.new("RGBA", plan.canvas_size, (120, 70, 30, 255))
        result = finish_image(raw, source, plan, True)
        self.assertEqual(result.size, (16, 8))
        expected = source.getchannel("A").resize(plan.target_size, Image.Resampling.BILINEAR)
        self.assertEqual(result.getchannel("A").tobytes(), expected.tobytes())

    def test_model_alpha_when_disabled(self):
        source = Image.new("RGBA", (8, 8), (10, 20, 30, 128))
        plan = plan_image(source.size, 2)
        result = finish_image(Image.new("RGBA", plan.canvas_size, (0, 0, 0, 64)), source, plan, False)
        self.assertEqual(result.getchannel("A").getextrema(), (64, 64))

    def test_opaque_becomes_rgb(self):
        source = Image.new("RGB", (8, 8))
        plan = plan_image(source.size, 2)
        result = finish_image(Image.new("RGBA", plan.canvas_size, (20, 40, 60, 255)), source, plan, True)
        self.assertEqual(result.mode, "RGB")

    def test_wrong_result_size(self):
        with self.assertRaises(EnhancerError):
            finish_image(Image.new("RGB", (64, 64)), Image.new("RGB", (8, 8)), plan_image((8, 8), 2), True)

    def test_unexpected_transparent_result(self):
        source = Image.new("RGB", (8, 8))
        plan = plan_image(source.size, 2)
        with self.assertRaises(EnhancerError):
            finish_image(Image.new("RGBA", plan.canvas_size), source, plan, False)

    def test_atomic_png_and_tga(self):
        image = Image.new("RGBA", (16, 16), (30, 40, 50, 60))
        for extension in ("png", "tga"):
            with self.subTest(extension=extension):
                path = save_image_atomic(image, self.root / f"export.{extension}")
                with Image.open(path) as saved:
                    self.assertEqual(saved.convert("RGBA").tobytes(), image.tobytes())

    def test_source_overwrite_blocked(self):
        source = self.fixture()
        previous = source.read_bytes()
        with self.assertRaises(EnhancerError):
            save_image_atomic(Image.new("RGB", (4, 4)), source, source, overwrite=True)
        self.assertEqual(source.read_bytes(), previous)

    def test_hardlink_source_overwrite_blocked(self):
        source = self.fixture()
        alias = self.root / "alias.png"
        os.link(source, alias)
        self.assertTrue(paths_match(source, alias))
        with self.assertRaises(EnhancerError):
            save_image_atomic(Image.new("RGB", (4, 4)), alias, source, overwrite=True)

    def test_symlink_output_blocked(self):
        source = self.fixture()
        alias = self.root / "alias.png"
        try:
            alias.symlink_to(source)
        except OSError:
            self.skipTest("Symbolic links not available for this user")
        with self.assertRaises(EnhancerError):
            save_image_atomic(Image.new("RGB", (4, 4)), alias, overwrite=True)

    def test_output_overwrite_requires_consent(self):
        destination = self.fixture()
        with self.assertRaises(EnhancerError):
            save_image_atomic(Image.new("RGB", (4, 4)), destination)
        save_image_atomic(Image.new("RGB", (4, 4)), destination, overwrite=True)
        with Image.open(destination) as image:
            self.assertEqual(image.size, (4, 4))

    def test_encode_failure_leaves_existing_output_intact(self):
        destination = self.fixture()
        previous = destination.read_bytes()
        with patch.object(Image.Image, "save", side_effect=OSError("disk full")), self.assertRaises(OSError):
            save_image_atomic(Image.new("RGB", (4, 4)), destination, overwrite=True)
        self.assertEqual(destination.read_bytes(), previous)
        self.assertEqual(list(self.root.glob(".image-enhancer-*")), [])

    def test_rename_failure_cleans_temp(self):
        destination = self.fixture()
        with patch("enhancer.storage.os.replace", side_effect=PermissionError("locked")), self.assertRaises(OSError):
            save_image_atomic(Image.new("RGB", (4, 4)), destination, overwrite=True)
        self.assertEqual(list(self.root.glob(".image-enhancer-*")), [])

    def test_unsupported_output(self):
        with self.assertRaises(EnhancerError):
            save_image_atomic(Image.new("RGB", (4, 4)), self.root / "out.dds")

    def test_missing_output_directory(self):
        with self.assertRaises(EnhancerError):
            save_image_atomic(Image.new("RGB", (4, 4)), self.root / "missing" / "out.png")

    def test_recovery_directory(self):
        output = prepare_recovery(self.root / "new" / "outputs")
        self.assertTrue(output.is_dir())
        self.assertEqual(list(output.iterdir()), [])

    def test_recovery_directory_is_file(self):
        with self.assertRaises(OSError):
            prepare_recovery(self.fixture())


class DdsHeaderTests(unittest.TestCase):
    # v1.2 validates the complete mip payload, not only the first 128 bytes
    def header(self, **kwargs):
        from format_fixtures import dds_fixture
        return bytearray(dds_fixture(**kwargs))

    def inspect(self, data):
        from enhancer.formats.dds import inspect
        return inspect(data)

    def test_truncated_header(self):
        with self.assertRaises(EnhancerError): self.inspect(b"DDS ")

    def test_invalid_header_size(self):
        data = self.header(); struct.pack_into("<I", data, 4, 123)
        with self.assertRaises(EnhancerError): self.inspect(data)

    def test_cubemap(self):
        data = self.header(); struct.pack_into("<I", data, 112, 0x200)
        with self.assertRaises(EnhancerError): self.inspect(data)

    def test_volume(self):
        data = self.header(); struct.pack_into("<I", data, 24, 3)
        with self.assertRaises(EnhancerError): self.inspect(data)

    def test_mipmap_notice(self):
        self.assertIn("4 mip", self.inspect(self.header(levels=4)).description)

    def test_premultiplied_now_supported(self):
        for codec in ("DXT2", "DXT4"):
            self.assertTrue(self.inspect(self.header(codec=codec, premul=True)).premultiplied)

    def test_dx10_array_cube_volume_and_custom_alpha(self):
        for dimension, misc, array, alpha in ((3, 0, 2, 0), (3, 4, 1, 0), (4, 0, 1, 0), (3, 0, 1, 4)):
            header = self.header(dxgi=77)
            struct.pack_into("<5I", header, 128, 77, dimension, misc, array, alpha)
            with self.subTest(values=(dimension, misc, array, alpha)), self.assertRaises(EnhancerError): self.inspect(header)

    def test_dx10_single_texture_allowed(self):
        self.assertEqual(self.inspect(self.header(dxgi=77)).codec, "DXT5")

    def test_dx10_truncated(self):
        with self.assertRaises(EnhancerError): self.inspect(self.header(dxgi=77)[:147])


if __name__ == "__main__":
    unittest.main()
