"""Queue, storage and transparent-texel regressions; no live HTTP requests."""
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from enhancer.api import ApiImage, run_job
from enhancer.batch import prepare_batch, request_count, run_batch
from enhancer.core import EnhancerError, Settings, load_texture, plan_settings
from enhancer.formats import decode_source, encode_original, verify_export
from enhancer.formats.common import resize_artwork
from enhancer.formats.dds import _mip
from enhancer.gui import App
from format_fixtures import raw_dds
import test_gui as gui_tests


class UpscaleWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.stop = threading.Event()

    def source(self, name="source.png", color=(95, 125, 165, 0), size=(32, 16)):
        path = self.root / name
        image = Image.new("RGBA", size, color)
        image.paste((60, 100, 150, 190), (size[0]//4, size[1]//4, size[0]*3//4, size[1]*3//4))
        image.save(path)
        return load_texture(path)

    def test_hidden_rgb_is_retained_without_changing_alpha_or_visible_pixels(self):
        source = self.source().image
        size = (256, 128)
        result = resize_artwork(source, size)
        ordinary = source.resize(size, Image.Resampling.LANCZOS)
        self.assertEqual(result.getpixel((0, 0)), (95, 125, 165, 0))
        self.assertEqual(result.getchannel("A").tobytes(), ordinary.getchannel("A").tobytes())
        for background in ((0, 0, 0, 255), (255, 255, 255, 255)):
            base = Image.new("RGBA", size, background)
            self.assertEqual(Image.alpha_composite(base, result).tobytes(), Image.alpha_composite(base, ordinary).tobytes())
        for srgb in (False, True):
            mip = _mip(result, (128, 64), srgb)
            self.assertEqual(mip.getpixel((0, 0)), (95, 125, 165, 0))

    def test_local_dds_output_retains_layout_hidden_color_and_no_extra_png(self):
        _, profile = decode_source(raw_dds((32, 16)), ".dds")
        image = self.source().image
        path = self.root / "asset.DDS"
        path.write_bytes(encode_original(image, profile))
        before = path.read_bytes()
        texture = load_texture(path)
        result = run_job(texture, Settings(method="lanczos", target_edge=256), "", self.root / "out")
        self.assertEqual(result.original_path.suffix, ".DDS")
        data = result.original_path.read_bytes()
        verify_export(data, texture.profile, (256, 128))
        decoded, _ = decode_source(data, ".dds")
        self.assertEqual(decoded.getpixel((0, 0)), (95, 125, 165, 0))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list((self.root / "out").iterdir()), [result.original_path])
        self.assertIsNone(result.metadata_path)

    def test_4k_batch_continues_after_enlargement_notice_and_reuses_identical_image(self):
        first = self.source("one.png")
        second = self.source("two.png", color=(190, 20, 30, 255))
        duplicate = self.root / "duplicate.png"
        duplicate.write_bytes(first.path.read_bytes())
        paths = (first.path, second.path, duplicate)
        settings = Settings(target_edge=4096)
        items = prepare_batch(paths, settings, self.stop)
        self.assertEqual(request_count(items, settings), 2)
        self.assertTrue(all(item.plan.resampled_output for item in items))
        events = []
        def respond(key, upload, plan, settings, alpha, **kwargs):
            return ApiImage(Image.new("RGBA", plan.canvas_size, (80, 100, 150, 255)), "offline", {"total_tokens": 7})
        with patch("enhancer.api.request_image", side_effect=respond) as api:
            summary = run_batch(items, settings, "sk-offline", self.root / "out", self.stop, events.append)
        self.assertEqual(api.call_count, 2)
        self.assertEqual((summary.completed, summary.failed, summary.warning), (3, 0, ""))
        self.assertEqual([(e.index, e.state) for e in events],
                         [(0, "Processing"), (0, "Done"), (1, "Processing"), (1, "Done"), (2, "Processing"), (2, "Done")])
        self.assertEqual(summary.last_result.image.size, (4096, 2048))
        files = [p.relative_to(summary.directory).as_posix() for p in summary.directory.rglob("*") if p.is_file()]
        self.assertEqual(sorted(files), ["duplicate.png", "one.png", "two.png"])
        self.assertTrue(all(p.is_file() for p in summary.directory.iterdir()))
        self.assertIsNone(summary.recovery_directory)
        self.assertEqual(summary.last_result.usage, {})
        self.assertEqual((summary.directory / "one.png").read_bytes(), (summary.directory / "duplicate.png").read_bytes())

    def test_up_to_date_files_are_byte_exact_copies_without_api_or_key(self):
        texture = self.source()
        settings = Settings(target_edge=32)
        items = prepare_batch((texture.path,), settings, self.stop)
        self.assertEqual(request_count(items, settings), 0)
        with patch("enhancer.api.request_image") as api:
            result = run_job(texture, settings, "", self.root / "out")
        api.assert_not_called()
        self.assertEqual(result.original_path.read_bytes(), texture.path.read_bytes())
        self.assertTrue(result.notices)
        self.assertEqual(result.warnings, ())
        self.assertEqual(request_count(prepare_batch((texture.path,), replace(settings, skip_up_to_date=False), self.stop),
                                       replace(settings, skip_up_to_date=False)), 1)

    def test_existing_destination_is_untouched_and_blocks_request(self):
        texture = self.source()
        destination = self.root / "existing.png"
        destination.write_bytes(b"leave this alone")
        with patch("enhancer.api.request_image") as api, self.assertRaises(EnhancerError):
            run_job(texture, Settings(), "sk-offline", self.root / "out", native_destination=destination)
        api.assert_not_called()
        self.assertEqual(destination.read_bytes(), b"leave this alone")

    def test_failed_paid_export_keeps_recovery_and_stops_next_request(self):
        paths = (self.source("first.png").path, self.source("second.png").path)
        settings = Settings()
        items = prepare_batch(paths, settings, self.stop)
        plan = plan_settings((32, 16), settings)
        with patch("enhancer.api.request_image", return_value=ApiImage(Image.new("RGBA", plan.canvas_size, "white"), "offline", {})) as api, \
                patch("enhancer.api.save_original_atomic", side_effect=OSError("test disk error")):
            summary = run_batch(items, settings, "sk-offline", self.root / "out", self.stop)
        api.assert_called_once()
        self.assertEqual((summary.completed, summary.failed), (0, 1))
        self.assertTrue(summary.last_result.png_path.is_file())
        self.assertTrue(summary.last_result.raw_path.is_file())
        self.assertFalse(list(summary.directory.glob(".reuse-*")))


class DefaultWorkflowGuiTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)

    def test_starts_with_openai_max_2k_and_routes_one_image_through_the_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "single.png"
            Image.new("RGBA", (64, 32), (60, 100, 150, 190)).save(source)
            app = App(recovery_directory=root / "out", prompt_for_key=False)
            try:
                app.withdraw()
                self.assertEqual(app._settings().method, "openai")
                self.assertEqual(app._settings().quality, "max")
                self.assertEqual(app._settings().target_edge, 2048)
                self.assertFalse(app._settings().keep_intermediates)
                self.assertEqual(app.enhance_edits.winfo_manager(), "pack")
                app._batch_paths = [source]
                app.batch_tree.insert("", "end", iid="0", values=(source.name, "—", "Queued"))
                app.load_file(source)
                with patch.object(app, "process_batch") as batch, patch.object(app, "process") as single:
                    app.tabs.select(app.preview_tab)
                    app._run_active()
                    app.tabs.select(app.batch_tab)
                    app._run_active()
                self.assertEqual(batch.call_count, 2)
                single.assert_not_called()
            finally:
                app.destroy()
