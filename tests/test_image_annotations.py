"""Per-image requests, optional masks, output protection and remembered queue annotations."""
from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image, ImageDraw

from enhancer.annotations import AnnotationStore, ImageAnnotation
from enhancer.api import ApiImage, request_image, run_job
from enhancer.batch import prepare_batch, request_count, run_batch
from enhancer.core import EnhancerError, Settings, build_prompt, is_up_to_date, load_texture, make_canvas, plan_settings
from enhancer.super_resolution import upscale_artwork
from enhancer.translation import map_box
from test_api import fields_from_request, response_body


class ImageAnnotationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        state = patch.dict(os.environ, {"IMAGE_ENHANCER_DATA_DIR": str(self.root / "state")})
        state.start()
        self.addCleanup(state.stop)
        self.source = self.root / "button.tga"
        art = Image.new("RGBA", (35, 35))
        draw = ImageDraw.Draw(art)
        draw.ellipse((1, 1, 33, 33), fill=(5, 150, 220, 255))
        draw.line((7, 18, 26, 18), fill="white", width=2)
        art.save(self.source)
        self.texture = load_texture(self.source)
        self.note = ImageAnnotation(self.texture.sha256, "Menü", "White rounded lettering", (4, 10, 31, 22))
        self.stop = threading.Event()
        self.settings = Settings(method="neural", scale=8)

    @staticmethod
    def response(key, upload, plan, settings, has_alpha, **kwargs):
        return ApiImage(Image.new("RGBA", plan.canvas_size, (240, 180, 70, 255)), "offline", {})

    def test_request_preserves_unicode_and_maps_area_to_padded_api_canvas(self):
        special = 'Menü\n„Été“ & スケート'
        note = replace(self.note, exact_text=special)
        settings = self.settings.for_image(note)
        plan = plan_settings((35, 35), settings)

        def receive(req):
            fields = fields_from_request(req)
            prompt = fields["prompt"].decode("utf-8")
            self.assertIn(json.dumps(special, ensure_ascii=False), prompt)
            self.assertIn("Do not translate", prompt)
            self.assertIn(note.note, prompt)
            with Image.open(BytesIO(fields["mask"])) as mask:
                self.assertEqual(mask.size, plan.canvas_size)
                self.assertEqual(mask.getchannel("A").point(lambda v: 255-v).getbbox(), note.canvas_area(plan))
            return httpx.Response(200, content=response_body(plan.canvas_size))

        upload, alpha = make_canvas(self.texture.image, plan)
        request_image("sk-test", upload, plan, settings, alpha, transport=httpx.MockTransport(receive))
        only_note = replace(note, exact_text="", area=None)
        self.assertIn(only_note.note, build_prompt(plan, self.settings.for_image(only_note)))

    def test_wording_alone_sends_one_unmasked_request_and_keeps_the_source_silhouette(self):
        annotation = replace(self.note, area=None, note="")
        settings = self.settings.for_image(annotation)
        sent = []

        def receive(request):
            fields = fields_from_request(request)
            self.assertNotIn("mask", fields)
            self.assertIn(b'"exact_text": "Men\xc3\xbc"', fields["prompt"])
            sent.append(request.method)
            w, h = map(int, fields["size"].decode().split("x"))
            return httpx.Response(200, content=response_body((w, h)))

        result = run_job(self.texture, settings, "sk-test", self.root / "out", transport=httpx.MockTransport(receive))
        local = upscale_artwork(self.texture.image, (280, 280))
        self.assertEqual(sent, ["POST"])
        self.assertEqual(result.image.getchannel("A").tobytes(), local.getchannel("A").tobytes())
        self.assertNotEqual(result.image.getpixel((100, 115)), local.getpixel((100, 115)))

    def test_annotated_label_survives_strong_protection_and_only_marked_pixels_change(self):
        for protection in ("strict", "balanced", "detail"):
            with self.subTest(protection=protection):
                shared = replace(self.settings, method="openai", source_protection=protection)
                settings = shared.for_image(self.note)
                with patch("enhancer.api.request_image", side_effect=self.response) as api:
                    result = run_job(self.texture, settings, "sk-test", self.root / "out")
                api.assert_called_once()
                local = np.asarray(upscale_artwork(self.texture.image, (280, 280)))
                with Image.open(result.original_path) as saved:
                    output = np.array(saved)
                area = np.zeros(local.shape[:2], dtype=bool)
                x, y, right, bottom = map_box(self.note.area, (35, 35), (280, 280))
                area[y:bottom, x:right] = True
                np.testing.assert_array_equal(local[~area], output[~area])
                self.assertTrue(np.all(output[area] == (240, 180, 70, 255)))

    def test_transparent_label_allows_new_dots_inside_the_annotation(self):
        label = Image.new("RGBA", (8, 8))
        label.putpixel((2, 5), (255, 255, 255, 255))
        label.save(self.source)
        tex = load_texture(self.source)
        note = ImageAnnotation(tex.sha256, "ü", area=(1, 1, 7, 7))
        settings = self.settings.for_image(note)

        def draw_dots(key, upload, plan, *args, **kwargs):
            raw = Image.new("RGBA", plan.canvas_size)
            x, y, r, b = plan.crop
            for px in (2, 5):
                box = map_box((px, 2, px+1, 3), (8, 8), (r-x, b-y), (x, y))
                raw.paste((250, 250, 250, 255), box)
            return ApiImage(raw, "offline", {})

        with patch("enhancer.api.request_image", side_effect=draw_dots):
            result = run_job(tex, settings, "sk-test", self.root / "out")
        self.assertGreater(result.image.getpixel((20, 20))[3], 240)
        self.assertGreater(result.image.getpixel((44, 20))[3], 240)
        self.assertEqual(result.image.getpixel((4, 4))[3], 0)

    def test_mixed_folder_separates_language_notes_and_reuses_matching_edits(self):
        paths = [self.root / f"button{i}.tga" for i in range(5)]
        for path in paths:
            path.write_bytes(self.source.read_bytes())
        # A different encoding of the same image is still eligible for reuse.
        self.texture.image.save(paths[2], compression="tga_rle")
        same = replace(self.note, source_sha256=load_texture(paths[2]).sha256)
        notes = {paths[1]: self.note, paths[2]: same, paths[3]: replace(self.note, exact_text="Menu")}
        shared = replace(self.settings, keep_intermediates=True)
        items = prepare_batch(tuple(paths), shared, self.stop, annotations=notes)
        self.assertTrue(all(not item.error for item in items))
        self.assertEqual(request_count(items, shared), 2)
        self.assertEqual(request_count(items, replace(shared, reuse_identical=False)), 3)
        self.assertEqual(items[0].reuse_key, items[4].reuse_key)
        self.assertEqual(items[1].reuse_key, items[2].reuse_key)
        self.assertEqual(len({i.reuse_key for i in items}), 3)
        with patch("enhancer.api.request_image", side_effect=self.response) as api:
            result = run_batch(items, shared, "sk-test", self.root / "out", self.stop)
        self.assertEqual([c.args[3].annotation.exact_text for c in api.call_args_list], ["Menü", "Menu"])
        self.assertEqual((result.completed, result.failed, result.warning), (5, 0, ""))
        with Image.open(result.directory / paths[1].name) as a, Image.open(result.directory / paths[2].name) as b:
            self.assertEqual(a.tobytes(), b.tobytes())
        self.assertNotEqual((result.directory / paths[0].name).read_bytes(), (result.directory / paths[1].name).read_bytes())
        records = json.loads((result.recovery_directory / "batch.json").read_text())
        self.assertEqual([e["method"] for e in records["items"]], ["neural", "openai", "openai", "openai", "neural"])
        for path in result.recovery_directory.rglob("*.json"):
            data = json.dumps(json.loads(path.read_text()), ensure_ascii=False)
            self.assertNotIn("Menü", data)
            self.assertNotIn(self.note.note, data)

    def test_reuse_includes_notes_and_region_while_translation_ignores_artwork_annotations(self):
        variants = [self.note, replace(self.note, note="Another style"), replace(self.note, area=None)]
        self.assertEqual(len({v.reuse_key("pixels") for v in variants}), 3)
        settings = Settings(task="translate", translation_scope="all")
        items = prepare_batch((self.source,), settings, self.stop, annotations={self.source: self.note})
        self.assertIsNone(items[0].annotation)
        prompt = build_prompt(items[0].plan, settings)
        self.assertNotIn(self.note.exact_text, prompt)
        with self.assertRaises(EnhancerError):
            prepare_batch((self.source,), self.settings.for_image(self.note), self.stop)

    def test_stale_or_invalid_annotations_fail_before_a_request(self):
        for invalid in (replace(self.note, area=(1, 1, 100, 100)), replace(self.note, source_sha256="a"*64),
                        replace(self.note, note="\x00"), replace(self.note, exact_text="x"*513)):
            with self.subTest(area=invalid.area):
                items = prepare_batch((self.source,), self.settings, self.stop, annotations={self.source: invalid})
                self.assertTrue(items[0].error)
                self.assertEqual(request_count(items, self.settings), 0)
                with patch("enhancer.api.request_image") as api, self.assertRaises(EnhancerError):
                    run_job(self.texture, self.settings.for_image(invalid), "sk-test", self.root / "out")
                api.assert_not_called()
        with self.assertRaises(EnhancerError):
            replace(self.note, note="", exact_text=" ").validate()

    def test_already_large_annotation_still_corrects_label_and_untagged_copy_stays_free(self):
        copy = self.root / "copy.tga"
        copy.write_bytes(self.source.read_bytes())
        shared = replace(self.settings, target_edge=32)
        items = prepare_batch((copy, self.source), shared, self.stop, annotations={self.source: self.note})
        self.assertTrue(is_up_to_date(shared, items[0].plan))
        self.assertTrue(items[1].needs_request(shared))
        self.assertEqual(request_count(items, shared), 1)
        with patch("enhancer.api.request_image", side_effect=self.response) as api:
            result = run_batch(items, shared, "sk-test", self.root / "out", self.stop)
        api.assert_called_once()
        self.assertEqual(result.completed, 2)
        self.assertEqual((result.directory / copy.name).read_bytes(), copy.read_bytes())
        with Image.open(result.directory / self.source.name) as saved:
            self.assertEqual(saved.size, (35, 35))

    def test_opt_in_store_scopes_by_full_path_and_skin_revision(self):
        first = AnnotationStore(self.root / "state" / "annotations")
        other = self.root / "another" / self.source.name
        self.assertEqual(first.load([self.source, other]), ({}, []))
        self.assertFalse(first.directory.exists())
        first.save(self.source, self.note)
        fresh = AnnotationStore(first.directory)
        self.assertEqual(fresh.load([self.source, other]), ({self.source: self.note}, []))
        with self.assertRaisesRegex(EnhancerError, "image changed"):
            fresh.load([self.source])[0][self.source].check_source("a"*64)
        fresh.save(self.source, None)
        self.assertEqual(first.load([self.source]), ({}, []))
        self.assertEqual([p.name for p in self.root.glob("*.tga")], [self.source.name])
