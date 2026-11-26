"""Selected output formats through the real processing and batch entry points."""
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image, ImageOps, JpegImagePlugin

from enhancer.api import ApiImage, run_job
from enhancer.batch import collect_files, prepare_batch, request_count, run_batch
from enhancer.core import EnhancerError, Settings, load_texture
from enhancer.formats import EXTENSIONS, output_profile, read_saved_export
from format_fixtures import pattern


class OutputFormatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "output"
        self.stop = threading.Event()

    def test_all_input_and_output_formats_through_local_jobs(self):
        source = pattern((17, 11))
        base = Settings(method="nearest", scale=2)
        for extension in (".PNG", ".tgA", ".DDS", ".JPG", ".jpeg"):
            path = self.root / ("card" + extension)
            (source.convert("RGB") if extension.lower() in (".jpeg", ".jpg") else source).save(path)
            texture = load_texture(path)
            for choice in ("original", "PNG", "TGA", "DDS", "JPEG"):
                with self.subTest(source=extension, output=choice):
                    result = run_job(texture, replace(base, output_format=choice), "", self.out)
                    self.assertEqual(result.warnings, ())
                    saved = result.original_path
                    target = texture.format if choice == "original" else choice
                    self.assertEqual(EXTENSIONS[saved.suffix.lower()], target)
                    self.assertEqual(saved.suffix, extension if choice == "original" else output_profile(texture.profile, choice).extension)
                    with Image.open(saved) as file:
                        self.assertEqual((file.format, file.size), (target, (34, 22)))
                    actual = read_saved_export(saved, (34, 22)).image
                    if target != "JPEG":
                        self.assertEqual(actual.tobytes(), result.image.convert("RGBA").tobytes())

    def test_jpg_matte_sampling_alias_and_byte_exact_native_copy(self):
        src = self.root / "alpha.png"
        image = Image.new("RGBA", (40, 16), (255, 0, 50, 0))
        image.paste((0, 0, 255, 128), (20, 0, 40, 16))
        image.save(src)
        settings = Settings(output_format="JPEG", target_edge=10, method="openai")
        with patch("enhancer.api.request_image") as model:
            saved = run_job(load_texture(src), settings, "", self.out).original_path
            model.assert_not_called()
        with Image.open(saved) as jpg:
            self.assertEqual(jpg.mode, "RGB")
            self.assertEqual(JpegImagePlugin.get_sampling(jpg), 0)
            self.assertEqual(jpg.getpixel((6, 8)), (255, 255, 255))
            r, g, b = jpg.getpixel((32, 8))
            self.assertLessEqual(abs(r-127)+abs(g-127)+abs(b-255), 4)
        alias = self.root / "Photograph.JPEG"
        alias.write_bytes(saved.read_bytes())
        native = run_job(load_texture(alias), settings, "", self.out).original_path
        self.assertEqual(native.name, "Photograph.jpg")
        self.assertEqual(native.read_bytes(), alias.read_bytes())

    def test_jpeg_exif_rotation_and_mirror_are_baked_into_converted_pixels(self):
        source = self.root / "portrait.JPG"
        for tag in (2, 5, 6, 8):
            with self.subTest(tag=tag):
                exif = Image.Exif()
                exif[274] = tag
                pattern((13, 9), opaque=True).convert("RGB").save(source, exif=exif, quality=96, subsampling=0)
                settings = Settings(method="nearest", scale=1, output_format="TGA")
                items = prepare_batch((source,), settings, self.stop)
                with Image.open(source) as picture:
                    expected = ImageOps.exif_transpose(picture).convert("RGBA")
                self.assertEqual(items[0].output_size, expected.size)
                result = run_batch(items, settings, "", self.out, self.stop).last_result
                actual = read_saved_export(result.original_path, expected.size).image
                self.assertEqual(actual.tobytes(), expected.tobytes())
                native = run_job(load_texture(source), replace(settings, output_format="original"), "", self.out)
                with Image.open(native.original_path) as jpg:
                    self.assertEqual(jpg.getexif()[274], tag)
                    self.assertEqual(jpg.size, (13, 9))

    def test_paid_batch_reuses_artwork_and_allocates_names_for_the_final_extension(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        for name in ("button.png", "button.tga", "another.PNG"):
            pattern().save(inputs / name)
        settings = Settings(output_format="JPEG", scale=2)
        sources, _ = collect_files((inputs,))
        items = prepare_batch(tuple(sources), settings, self.stop)
        self.assertEqual(request_count(items, settings), 1)
        def response(key, upload, plan, *args, **kwargs):
            return ApiImage(Image.new("RGBA", plan.canvas_size, (170, 110, 35, 255)), "mock", {})
        self.out.mkdir()
        existing = self.out / "button.JPG"
        existing.write_bytes(b"older output")
        with patch("enhancer.api.request_image", side_effect=response) as model:
            summary = run_batch(items, settings, "offline", self.out, self.stop)
        self.assertEqual((summary.completed, summary.failed), (3, 0))
        model.assert_called_once()
        self.assertEqual(existing.read_bytes(), b"older output")
        names = ("button (2).jpg", "button (3).jpg", "another.jpg")
        for filename in names:
            with Image.open(self.out / filename) as file:
                self.assertEqual(file.format, "JPEG")
        self.assertEqual(len({(self.out / n).read_bytes() for n in names}), 1)
        with self.assertRaisesRegex(EnhancerError, "after preflight"):
            run_batch(items, replace(settings, output_format="PNG"), "", self.out, self.stop)

    def test_invalid_destination_and_stale_saved_preview(self):
        source = self.root / "original.png"
        pattern().save(source)
        settings = Settings(output_format="JPEG")
        with patch("enhancer.api.request_image") as api, self.assertRaisesRegex(EnhancerError, "extension"):
            run_job(load_texture(source), settings, "offline", self.out, native_destination=self.out / "wrong.png")
        api.assert_not_called()
        saved = run_job(load_texture(source), replace(settings, method="nearest"), "", self.out)
        with self.assertRaises(EnhancerError):
            read_saved_export(saved.original_path, (7, 9))
        view = read_saved_export(saved.original_path, saved.plan.target_size)
        self.assertEqual(view.verified_bytes(), saved.original_path.read_bytes())
        saved.original_path.write_bytes(b"replaced after opening")
        with self.assertRaisesRegex(EnhancerError, "changed"):
            view.verified_bytes()

    def test_saved_export_uses_output_pixel_budget(self):
        source = self.root / "large.png"
        Image.new("RGBA", (21, 17)).save(source)
        with patch("enhancer.formats.common.MAX_PIXELS", 100):
            with self.assertRaises(EnhancerError):
                load_texture(source)
            self.assertEqual(read_saved_export(source, (21, 17)).image.size, (21, 17))
