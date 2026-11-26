"""Reuse must reduce paid calls without losing outputs or bypassing integrity checks."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image
from enhancer.api import ApiImage, run_job
from enhancer.batch import prepare_batch, request_count, run_batch
from enhancer.core import Settings, load_texture, plan_settings
from enhancer.formats import tga
from format_fixtures import tga_fixture


class BatchReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        state = patch.dict(os.environ, {"IMAGE_ENHANCER_DATA_DIR": str(self.root / "state")})
        state.start()
        self.addCleanup(state.stop)
        self.paths = tuple(self.root / f"{index}.png" for index in range(3))
        for path in self.paths:
            Image.new("RGBA", (32, 16), (10, 20, 30, 140)).save(path)
        self.settings = Settings()
        self.stop = threading.Event()

    def run_batch(self, settings=None, emit=lambda event: None):
        settings = settings or self.settings
        items = prepare_batch(self.paths, settings, self.stop)
        plan = plan_settings((32, 16), settings)
        with patch("enhancer.api.request_image", return_value=ApiImage(Image.new("RGBA", plan.canvas_size, (30, 50, 70, 255)), "test-request", {})) as api:
            summary = run_batch(items, settings, "sk-test", self.root / "out", self.stop, emit=emit)
        return items, summary, api.call_count

    def test_identical_files_one_request_three_native_outputs(self):
        items, summary, calls = self.run_batch(replace(self.settings, keep_intermediates=True))
        self.assertEqual(calls, 1)
        self.assertEqual(request_count(items, self.settings), 1)
        self.assertEqual((summary.completed, summary.failed), (3, 0))
        outputs = sorted(summary.directory.glob("*.png"))
        self.assertEqual(len(outputs), 3)
        self.assertEqual(len({path.read_bytes() for path in outputs}), 1)
        self.assertEqual(summary.last_result.image.size, (64, 32))
        self.assertEqual(summary.last_result.usage, {})
        record = json.loads((summary.recovery_directory / "batch.json").read_text())
        self.assertEqual(record["planned_api_requests"], 1)
        self.assertEqual(record["items"][2]["reused_from_index"], 0)

    def test_disable_reuse_makes_explicit_separate_requests(self):
        settings = replace(self.settings, reuse_identical=False)
        items, summary, calls = self.run_batch(settings)
        self.assertEqual(calls, 3)
        self.assertEqual(request_count(items, settings), 3)
        self.assertEqual(summary.completed, 3)

    def test_different_source_content_does_not_reuse(self):
        Image.new("RGBA", (32, 16), (90, 100, 110, 140)).save(self.paths[1])
        items, summary, calls = self.run_batch()
        self.assertEqual(calls, 2)
        self.assertEqual(request_count(items, self.settings), 2)
        self.assertEqual(summary.completed, 3)

    def test_matching_pixels_different_tga_layouts_and_png_share_one_edit(self):
        self.paths = (self.root / 'icon (en).tga', self.root / 'icon (de).tga',
                      self.root / 'icon (fr).png')
        for path, rle, origin in ((self.paths[0], False, 0x20), (self.paths[1], True, 0x10)):
            path.write_bytes(tga_fixture(size=(32, 16), rle=rle, origin=origin))
        load_texture(self.paths[0]).image.save(self.paths[2])
        items, summary, calls = self.run_batch(replace(self.settings, scale=12))
        self.assertEqual(len({i.sha256 for i in items}), 3)
        self.assertEqual(len({i.reuse_key for i in items}), 1)
        self.assertEqual((calls, summary.completed, summary.failed, summary.warning), (1, 3, 0, ''))
        outputs = [summary.directory / p.name for p in self.paths]
        for src, dst in zip(self.paths[:2], outputs[:2]):
            before, after = tga.inspect(src.read_bytes()), tga.inspect(dst.read_bytes(), output=True)
            self.assertEqual((before.rle, before.header[17], before.identifier),
                             (after.rle, after.header[17], after.identifier))
        self.assertEqual(load_texture(outputs[0]).image.size, (384, 192))
        self.assertEqual(len({load_texture(p).image.tobytes() for p in outputs}), 1)

    def test_shape_alpha_and_hidden_rgb_are_distinct(self):
        self.paths = tuple(self.root / f'state{i}.png' for i in range(4))
        for path, size, pixel in zip(self.paths, [(32, 16), (16, 32), (32, 16), (32, 16)],
                                     [(10, 20, 30, 0), (10, 20, 30, 0), (10, 21, 30, 0), (10, 20, 30, 1)]):
            image = Image.new('RGBA', size, 'white')
            image.putpixel((0, 0), pixel)
            image.save(path)
        items = prepare_batch(self.paths, self.settings, self.stop)
        self.assertEqual(request_count(items, self.settings), 4)

    def test_local_folder_reuses_resize_and_exports_every_filename(self):
        settings = replace(self.settings, method='lanczos', scale=10)
        items = prepare_batch(self.paths, settings, self.stop)
        calls = []
        def runner(texture, config, key, directory):
            calls.append(texture.path)
            return run_job(texture, config, key, directory)
        summary = run_batch(items, settings, '', self.root/'out', self.stop, runner=runner)
        self.assertEqual(calls, [self.paths[0]])
        self.assertEqual((summary.completed, summary.failed, request_count(items, settings)), (3, 0, 0))
        self.assertEqual({p.name for p in summary.directory.iterdir()}, {p.name for p in self.paths})
        self.assertEqual(len({(summary.directory/p.name).read_bytes() for p in self.paths}), 1)

    def test_tampered_native_cache_stops_without_extra_request(self):
        def corrupt(event):
            if event.state == "Done" and event.index == 0:
                event.output.write_bytes(b"changed")
        _, summary, calls = self.run_batch(emit=corrupt)
        self.assertEqual(calls, 1)
        self.assertEqual((summary.completed, summary.failed), (1, 1))
        self.assertIn("changed", summary.warning)

    def test_missing_png_cache_stops_without_extra_request(self):
        def remove(event):
            if event.state == "Done" and event.index == 0:
                for path in self.root.rglob("*_enhanced.png"):
                    path.unlink()
        _, summary, calls = self.run_batch(emit=remove)
        self.assertEqual(calls, 1)
        self.assertEqual((summary.completed, summary.failed), (1, 1))

    def test_changed_duplicate_source_is_still_rejected(self):
        def change(event):
            if event.state == "Done" and event.index == 0:
                Image.new("RGBA", (32, 16), "red").save(self.paths[1])
        _, summary, calls = self.run_batch(emit=change)
        self.assertEqual(calls, 1)
        self.assertIn("Source changed", summary.warning)
        self.assertEqual((summary.completed, summary.failed), (1, 1))

    def test_local_processing_never_counted_as_paid_reuse(self):
        settings = replace(self.settings, method="nearest")
        items, summary, calls = self.run_batch(settings)
        self.assertEqual(calls, 0)
        self.assertEqual(request_count(items, settings), 0)
        self.assertEqual(summary.completed, 3)
