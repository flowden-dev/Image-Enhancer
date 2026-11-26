"""Protected areas, cropped requests and exact reconstruction of unedited pixels."""
from dataclasses import replace
import io
import tempfile
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from enhancer.api import run_job
from enhancer.batch import prepare_batch, run_batch
from enhancer.core import EnhancerError, Settings, make_canvas, plan_settings
from enhancer.translation import (canvas_boxes, composite_translation, finish_translation,
                                  has_unprotected_pixels, make_translation_mask, map_box)
from test_api import fields_from_request, response_body


def config(**kwargs):
    return replace(Settings(task="translate", scale=1, text_regions=((100, 80, 360, 160),)), **kwargs)


class ProtectionTests(unittest.TestCase):
    def test_protection_wins_over_translation_in_mask_and_pixels(self):
        settings = config(protected_regions=((150, 100, 220, 130),))
        source = Image.new("RGBA", (512, 256), (12, 34, 56, 97))
        edit = Image.new("RGBA", source.size, (250, 20, 30, 240))
        result = composite_translation(edit, source, settings)
        self.assertEqual(result.getpixel((160, 110)), source.getpixel((160, 110)))
        self.assertEqual(result.getpixel((110, 90)), edit.getpixel((110, 90)))
        self.assertEqual(result.getpixel((0, 0)), source.getpixel((0, 0)))
        plan = plan_settings(source.size, settings)
        with Image.open(io.BytesIO(make_translation_mask(plan, settings))) as mask:
            b = canvas_boxes(plan, settings, protected=True)[0]
            self.assertEqual(mask.crop(b).getchannel("A").getextrema(), (255, 255))

    def test_all_text_with_protected_areas_restores_only_protected_pixels(self):
        settings = config(translation_scope="all", text_regions=(), protected_regions=((2, 2, 10, 10),))
        source = Image.new("RGBA", (32, 32), (4, 5, 6, 12))
        edit = Image.new("RGBA", source.size, (90, 91, 92, 150))
        result = composite_translation(edit, source, settings)
        self.assertEqual(result.getpixel((4, 4)), source.getpixel((4, 4)))
        self.assertEqual(result.getpixel((20, 20)), edit.getpixel((20, 20)))
        plan = plan_settings(source.size, settings)
        self.assertIsNone(plan.source_crop)
        self.assertIsNotNone(make_translation_mask(plan, settings))

    def test_union_of_protected_boxes_blocks_empty_edit_before_request(self):
        settings = config(text_regions=((0, 0, 10, 10),),
                          protected_regions=((0, 0, 5, 10), (5, 0, 10, 10)))
        with self.assertRaisesRegex(EnhancerError, "cover everything"):
            plan_settings((20, 20), settings)
        plan_settings((20, 20), replace(settings, protected_regions=((0, 0, 9, 10),)))

    def test_coverage_sweep_matches_pixel_reference_for_random_rectangles(self):
        rng = np.random.default_rng(76)
        for _ in range(150):
            boxes = []
            expected = np.ones((12, 16), dtype=bool)
            for _ in range(12):
                x, y = int(rng.integers(0, 16)), int(rng.integers(0, 12))
                right, bottom = int(rng.integers(x+1, 17)), int(rng.integers(y+1, 13))
                boxes.append((x, y, right, bottom))
                expected[y:bottom, x:right] = False
            self.assertEqual(has_unprotected_pixels((0, 0, 16, 12), boxes), bool(expected.any()))

    def test_invalid_protected_regions_rejected(self):
        for boxes in (((0, 0, 0, 5),), ((0, 0, 900, 2),), ((False, 0, 4, 2),),
                      ((-1, 0, 4, 2),), ((0, 0, 1.5, 3),), [[0, 0, 3, 3]],
                      tuple((i, 0, i+1, 1) for i in range(33))):
            with self.subTest(boxes=boxes), self.assertRaises(EnhancerError):
                plan_settings((512, 256), config(protected_regions=boxes))

    def test_original_and_resized_protected_alpha_and_hidden_colors(self):
        source = Image.fromarray(np.random.default_rng(83).integers(0, 256, (32, 64, 4), dtype=np.uint8))
        settings = config(text_regions=((0, 0, 64, 32),), protected_regions=((0, 0, 12, 15),))
        for size in (source.size, (128, 64)):
            result = composite_translation(Image.new("RGBA", size, (250, 0, 0, 255)), source, settings)
            box = map_box(settings.protected_regions[0], source.size, size)
            from enhancer.formats.common import resize_artwork
            reference = resize_artwork(source, size)
            self.assertEqual(result.crop(box).tobytes(), reference.crop(box).tobytes())

    def test_batch_rejects_any_per_image_protected_areas(self):
        settings = config(translation_scope="all", text_regions=(), protected_regions=((0, 0, 2, 2),))
        with self.assertRaisesRegex(EnhancerError, "belong to one image"):
            prepare_batch((Path("unused.png"),), settings, threading.Event())
        with self.assertRaisesRegex(EnhancerError, "belong to one image"):
            run_batch((), settings, "", Path("unused"), threading.Event())


class CroppedRequestTests(unittest.TestCase):
    def test_cropped_plan_reduces_canvas_without_changing_output(self):
        settings = config()
        full = plan_settings((2048, 1024), replace(settings, optimize_translation=False))
        cropped = plan_settings((2048, 1024), settings)
        self.assertIsNotNone(cropped.source_crop)
        self.assertEqual(cropped.target_size, full.target_size)
        self.assertEqual(cropped.source_size, full.source_size)
        self.assertLess(np.prod(cropped.canvas_size), np.prod(full.canvas_size))
        self.assertFalse(cropped.resampled_output)

    def test_full_image_selection_never_crops_or_increases_canvas(self):
        settings = config(text_regions=((0, 0, 2048, 1024),))
        plan = plan_settings((2048, 1024), settings)
        self.assertIsNone(plan.source_crop)

    def test_crop_near_border_clips_protected_mask_coordinates(self):
        settings = config(text_regions=((0, 0, 100, 40),), protected_regions=((50, 10, 2000, 30),))
        plan = plan_settings((2048, 1024), settings)
        self.assertIsNotNone(plan.source_crop)
        for b in canvas_boxes(plan, settings, protected=True):
            self.assertGreaterEqual(b[0], plan.crop[0])
            self.assertLessEqual(b[2], plan.crop[2])

    def test_crop_upload_contains_only_context_and_respects_alpha(self):
        settings = config()
        source = Image.new("RGBA", (2048, 1024), (80, 90, 100, 0))
        source.paste((10, 20, 30, 255), (0, 0, 500, 300))
        plan = plan_settings(source.size, settings)
        png, has_alpha = make_canvas(source, plan)
        self.assertFalse(has_alpha)
        with Image.open(io.BytesIO(png)) as uploaded:
            self.assertEqual(uploaded.size, plan.canvas_size)
            self.assertEqual(uploaded.crop(plan.crop).getpixel((0, 0)), (10, 20, 30, 255))

    def test_finish_cropped_request_pastes_at_global_coordinates(self):
        source = Image.new("RGBA", (2048, 1024), (10, 30, 50, 140))
        for scale in (1, 2):
            settings = config(scale=scale, protected_regions=((130, 100, 200, 120),))
            plan = plan_settings(source.size, settings)
            raw = Image.new("RGBA", plan.canvas_size, (200, 30, 40, 255))
            result = finish_translation(raw, source, plan, settings)
            self.assertEqual(result.size, plan.target_size)
            reference = source.resize(plan.target_size, Image.Resampling.LANCZOS)
            area = map_box(settings.text_regions[0], source.size, plan.target_size)
            reference.paste((200, 30, 40, 255), area)
            protected = map_box(settings.protected_regions[0], source.size, plan.target_size)
            reference.paste(source.resize(plan.target_size, Image.Resampling.LANCZOS).crop(protected), protected)
            self.assertEqual(result.tobytes(), reference.tobytes())

    def test_cropped_native_job_one_request_recovery_and_exact_protection(self):
        from enhancer.core import load_texture
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "input.png"
            source = Image.new("RGBA", (2048, 1024), (32, 64, 96, 255))
            source.save(path)
            settings = config(protected_regions=((130, 100, 200, 120),), keep_intermediates=True)
            plan = plan_settings(source.size, settings)
            calls = []
            def respond(request):
                calls.append(fields_from_request(request))
                return httpx.Response(200, content=response_body(plan.canvas_size))
            result = run_job(load_texture(path), settings, "sk-test", root / "out", transport=httpx.MockTransport(respond))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["quality"], b"max")
            self.assertIn("mask", calls[0])
            self.assertTrue(result.raw_path.exists())
            self.assertEqual(result.warnings, ())
            with Image.open(result.original_path) as native:
                self.assertEqual(native.size, source.size)
                for point in ((0, 0), (135, 105), (1000, 500)):
                    self.assertEqual(native.convert("RGB").getpixel(point), (32, 64, 96))
