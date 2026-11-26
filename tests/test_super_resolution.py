"""Regression checks for learned reconstruction, transparency, tiling and export size."""
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from enhancer.api import ApiImage, run_job
from enhancer.batch import prepare_batch, run_batch
from enhancer.core import EnhancerError, Settings, build_prompt, load_texture, plan_settings
from enhancer.fidelity import protect_source
from enhancer.super_resolution import upscale_artwork
from format_fixtures import raw_dds


class ReconstructionTests(unittest.TestCase):
    def test_antialiased_round_button_is_closer_to_known_high_resolution_outline(self):
        target = (192, 144)
        master = Image.new("L", (768, 576))
        ImageDraw.Draw(master).ellipse((160, 64, 616, 520), fill=255)
        original = master.resize((48, 36), Image.Resampling.BOX)
        truth = np.asarray(master.resize(target, Image.Resampling.LANCZOS), dtype=np.int16)
        smooth = np.asarray(original.resize(target, Image.Resampling.LANCZOS), dtype=np.int16)
        actual = np.asarray(upscale_artwork(original, target), dtype=np.int16)
        self.assertLess(np.abs(actual - truth).mean(), np.abs(smooth - truth).mean() / 2)
        self.assertLess(abs((actual - truth).mean()), 1)

    def test_hidden_rgb_never_tints_visible_texels_on_repeated_passes(self):
        alpha = Image.new("L", (23, 17))
        ImageDraw.Draw(alpha).ellipse((3, 1, 19, 15), fill=200)
        outputs = []
        for hidden in ("magenta", "lime"):
            image = Image.new("RGBA", alpha.size, hidden)
            image.paste((20, 100, 250, 255), mask=alpha.point(lambda a: 255 if a else 0))
            image.putalpha(alpha)
            outputs.append(np.asarray(upscale_artwork(image, (184, 136))))
        a, b = outputs
        np.testing.assert_array_equal(a[:, :, 3], b[:, :, 3])
        visible = a[:, :, 3] > 0
        np.testing.assert_array_equal(a[visible], b[visible])
        self.assertTrue(np.all(a[visible, :3] == (20, 100, 250)))
        self.assertFalse(np.array_equal(a[0, 0], b[0, 0]))

    def test_tiled_integer_and_fractional_passes_match_a_single_tile(self):
        rng = np.random.default_rng(81)
        values = rng.integers(0, 256, (35, 49, 4), dtype=np.uint8)
        values[:, :12, 3] = 0
        image = Image.fromarray(values)
        for size in ((98, 70), (123, 88), (392, 280)):
            with self.subTest(size=size):
                large = upscale_artwork(image, size)
                with patch("enhancer.super_resolution._TILE", 24):
                    small = upscale_artwork(image, size)
                np.testing.assert_array_equal(np.asarray(large), np.asarray(small))

    def test_detail_guard_allows_letter_edges_and_limits_drift_in_flat_fills(self):
        src = Image.new("RGBA", (9, 3), "black")
        ImageDraw.Draw(src).rectangle((4, 0, 8, 2), fill="white")
        edge = 32
        edited = Image.new("RGBA", (72, 24), (245, 245, 245, 100))
        detail = protect_source(edited, src, "detail")
        self.assertEqual(detail.getpixel((4, 12)), (12, 12, 12, 100))
        self.assertEqual(detail.getpixel((edge, 12)), (245, 245, 245, 100))
        self.assertLess(protect_source(edited, src, "strict").getpixel((edge, 12))[0], 245)
        self.assertEqual(detail.getpixel((62, 12)), (245, 245, 245, 100))

    def test_clear_neighbors_cannot_change_an_edits_color_bounds(self):
        edit = Image.new("RGBA", (81, 45), (40, 180, 200, 100))
        guarded = []
        for invisible in ((255, 0, 255, 0), (0, 255, 0, 0)):
            source = Image.new("RGBA", (9, 5), invisible)
            source.putpixel((4, 2), (25, 70, 190, 200))
            guarded.append(protect_source(edit, source, "detail"))
        self.assertEqual(guarded[0].tobytes(), guarded[1].tobytes())
        self.assertEqual(guarded[0].getpixel((0, 0)), edit.getpixel((0, 0)))
        self.assertEqual(guarded[0].getpixel((40, 22)), (37, 82, 200, 100))

    def test_paid_default_supplies_neural_guide_and_preserves_learned_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "label.tga"
            source = Image.new("RGBA", (35, 35))
            ImageDraw.Draw(source).polygon([(3, 25), (26, 5), (24, 31)], fill=(10, 90, 200, 255))
            source.save(path)
            texture = load_texture(path)
            settings = Settings(scale=8)
            plan = plan_settings(source.size, settings)
            neural = upscale_artwork(texture.image, plan.target_size)
            self.assertIn("Remove interpolation blur", build_prompt(plan, settings))

            def echo(key, upload, request_plan, *args, **kwargs):
                with Image.open(BytesIO(upload)) as opened:
                    raw = opened.convert("RGBA")
                self.assertEqual(request_plan, plan)
                box = plan.crop
                expected = neural.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
                self.assertEqual(raw.crop(box).tobytes(), expected.tobytes())
                return ApiImage(raw, "mock", {})

            with patch("enhancer.api.request_image", side_effect=echo) as api:
                result = run_job(texture, settings, "sk-test", root / "out")
            api.assert_called_once()
            with Image.open(result.original_path) as saved:
                self.assertEqual(saved.size, (280, 280))
                self.assertEqual(saved.getchannel("A").tobytes(), neural.getchannel("A").tobytes())
                smooth = source.getchannel("A").resize(saved.size, Image.Resampling.BILINEAR)
                self.assertNotEqual(saved.getchannel("A").tobytes(), smooth.tobytes())

    def test_preflight_rejects_neural_data_textures_and_changed_batch_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "channels.dds"
            data.write_bytes(raw_dds(bits=8, masks=(0, 0, 0, 255), kind="alpha"))
            button = root / "button.png"
            Image.new("RGB", (35, 35)).save(button)
            settings = Settings(method="neural", scale=8)
            stop = threading.Event()
            data_item, valid = prepare_batch((data, button), settings, stop)
            self.assertIn("channel/data", data_item.error)
            with self.assertRaisesRegex(EnhancerError, "changed after preflight"):
                run_batch((valid,), replace(settings, scale=2), "", root / "out", stop)
            self.assertFalse((root / "out").exists())

    def test_wrong_backend_dimensions_cannot_be_saved_as_an_8x_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = root / "button.tga"
            Image.new("RGBA", (140, 140)).save(name)
            with patch("enhancer.api.upscale_artwork", return_value=Image.new("RGBA", (280, 280))):
                with self.assertRaisesRegex(EnhancerError, "expected 1120 x 1120"):
                    run_job(load_texture(name), Settings(scale=8, method="neural"), "", root / "out")
            self.assertFalse((root / "out" / name.name).exists())
