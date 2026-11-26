from dataclasses import replace
from pathlib import Path
import io
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from enhancer.api import run_job
from enhancer.batch import BatchItem, collect_files, prepare_batch, run_batch
from enhancer.core import EnhancerError, Settings, plan_settings, target_dimensions
from enhancer.gui import CUSTOM_SCALE, SCALE_PRESETS, output_target
from format_fixtures import dds_fixture, tga_fixture


class ResolutionTests(unittest.TestCase):
    def test_factor_presets_custom_factors_and_pixel_rounding(self):
        self.assertEqual(tuple(SCALE_PRESETS.values()), tuple(range(2, 9)))
        for option, factor in SCALE_PRESETS.items():
            self.assertEqual(output_target(option, '', ''), (factor, None))
        for value, size, expected in (
                ('8', (197, 63), (1576, 504)), ('10X', (1024, 1024), (10240, 10240)),
                ('12×', (1024, 1024), (12288, 12288)), ('15', (3, 2), (45, 30)),
                ('2,5', (19, 17), (48, 43)), ('1.005', (100, 17), (101, 17))):
            scale, edge = output_target(CUSTOM_SCALE, 'ignored', value)
            for method in ('openai', 'lanczos', 'nearest'):
                with self.subTest(value=value, method=method):
                    plan = plan_settings(size, Settings(method=method, scale=scale, target_edge=edge))
                    self.assertEqual(plan.target_size, expected)
                    if method == 'openai':
                        self.assertLessEqual(max(plan.canvas_size), 3840)

    def test_invalid_factors_and_oversized_dimensions(self):
        for factor in (True, 0.99, float('nan'), float('inf'), -float('inf'), '12', 10**100):
            with self.subTest(factor=factor), self.assertRaises(EnhancerError):
                target_dimensions((1, 1), factor)
        for size, factor in (((2049, 1), 8), ((1024, 1025), 12)):
            with self.subTest(size=size), self.assertRaisesRegex(EnhancerError, 'Choose a smaller size'):
                target_dimensions(size, factor)
        self.assertEqual(target_dimensions((2048, 1024), 8), (16384, 8192))

    def test_texture_presets(self):
        for source, edge, target in (((512, 256), 2048, (2048, 1024)),
                                     ((256, 512), 4096, (2048, 4096)),
                                     ((1920, 1080), 2048, (2048, 1152)),
                                     ((1920, 1080), 4096, (4096, 2304)),
                                     ((1000, 333), 2048, (2048, 682)),
                                     ((3000, 1500), 2048, (3000, 1500))):
            with self.subTest(source=source, edge=edge):
                self.assertEqual(target_dimensions(source, target_edge=edge), target)
                self.assertLessEqual(abs(source[0] * target[1] - source[1] * target[0]), max(source) / 2)

    def test_unusual_ratio_reaches_the_requested_edge_with_pixel_rounding(self):
        self.assertEqual(target_dimensions((1267, 701), target_edge=2048), (2048, 1133))
        self.assertEqual(target_dimensions((1267, 701), target_edge=4096), (4096, 2266))

    def test_presets_share_ai_and_local_dimensions(self):
        for edge in (2048, 4096, 1234):
            for size in ((17, 11), (512, 64), (1920, 1080), (1, 500)):
                ai = plan_settings(size, Settings(target_edge=edge))
                local = plan_settings(size, Settings(target_edge=edge, method="nearest"))
                self.assertEqual(ai.target_size, local.target_size)
                self.assertLessEqual(abs(size[0] * ai.target_size[1] - size[1] * ai.target_size[0]), max(size) / 2)
                self.assertTrue(all(v % 16 == 0 for v in ai.canvas_size))
                self.assertLessEqual(max(ai.canvas_size), 3840)

    def test_invalid_target(self):
        for edge in (0, -1, 16385, True, 2048.0, "2048"):
            with self.subTest(edge=edge), self.assertRaises(EnhancerError):
                Settings(target_edge=edge).validate()


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "outputs"
        self.stop = threading.Event()
        self.events = []
        self.settings = Settings(method="nearest")

    def source(self, name="texture.png", color=(20, 80, 100, 180)):
        path = self.root / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGBA", (16, 8), color)
        if path.suffix.lower() == ".dds":
            path.write_bytes(dds_fixture("DXT5", (16, 8), levels=5))
        elif path.suffix.lower() == ".tga":
            path.write_bytes(tga_fixture((16, 8), rle=True))
        elif path.suffix.lower() in (".jpeg", ".jpg"):
            image.convert("RGB").save(path, quality=91, subsampling=0, progressive=True)
        else:
            image.save(path)
        return path

    def prepare(self, *paths):
        return prepare_batch(paths, self.settings, self.stop, self.events.append)

    def run_items(self, items, **kwargs):
        return run_batch(items, self.settings, "", self.out, self.stop, self.events.append, **kwargs)

    def test_real_mixed_format_batch_and_duplicate_basenames(self):
        paths = [self.source(n) for n in ("a/shared.PNG", "b/shared.PNG", "alpha.DDS", "rle.tga", "photo.jpeg")]
        originals = [p.read_bytes() for p in paths]
        summary = self.run_items(self.prepare(*paths))
        self.assertEqual((summary.completed, summary.failed), (5, 0))
        self.assertEqual(summary.directory, self.out)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         ["alpha.DDS", "photo.jpeg", "rle.tga", "shared (2).PNG", "shared.PNG"])
        self.assertEqual(originals, [p.read_bytes() for p in paths])
        self.assertEqual(summary.last_result.image.size, (32, 16))
        self.assertIsNone(summary.recovery_directory)
        self.assertEqual(len([e for e in self.events if e.state == "Done"]), 5)

    def test_invalid_input_is_preflighted_and_skipped(self):
        good = self.source()
        bad = good.with_name("broken.dds")
        bad.write_bytes(b"broken")
        items = self.prepare(bad, good)
        self.assertTrue(items[0].error)
        summary = self.run_items(items)
        self.assertEqual(summary.completed, 1)

    def test_source_change_never_calls_runner(self):
        path = self.source()
        items = self.prepare(path)
        self.source(color=(1, 2, 3, 255))
        with patch("enhancer.batch.run_job") as runner:
            summary = self.run_items(items, runner=runner)
        runner.assert_not_called()
        self.assertEqual(summary.failed, 1)
        self.assertIn("changed", summary.warning)

    def test_cancel_before_first_job(self):
        items = self.prepare(self.source())
        self.stop.set()
        with patch("enhancer.batch.run_job") as runner:
            summary = self.run_items(items, runner=runner)
        runner.assert_not_called()
        self.assertTrue(summary.cancelled)
        self.assertEqual(summary.completed, 0)

    def test_cancel_finishes_current_and_does_not_start_next(self):
        items = self.prepare(self.source("one.png"), self.source("two.png"))
        calls = []
        def runner(*args):
            calls.append(args[0].path)
            self.stop.set()
            return run_job(*args)
        summary = self.run_items(items, runner=runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary.completed, 1)
        self.assertTrue(summary.last_result.original_path.exists())
        self.assertTrue(summary.cancelled)

    def test_first_failure_stops_queue_without_retries(self):
        items = self.prepare(self.source("one.png"), self.source("two.png"))
        with patch("enhancer.batch.run_job", side_effect=EnhancerError("connection failed")) as runner:
            summary = self.run_items(items, runner=runner)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(json.loads((summary.recovery_directory / 'batch.json').read_text())["items"][1]["status"], "pending")

    def test_failed_native_output_retains_image_and_png(self):
        items = self.prepare(self.source("one.png"), self.source("two.png"))
        with patch("enhancer.api.save_original_atomic", side_effect=OSError("native export blocked")):
            summary = self.run_items(items)
        self.assertEqual((summary.completed, summary.failed), (0, 1))
        self.assertTrue(summary.last_result.png_path.exists())
        self.assertEqual(summary.last_result.image.size, (32, 16))

    def test_no_secrets_or_prompt_in_batch_manifest(self):
        items = self.prepare(self.source())
        summary = run_batch(items, replace(self.settings, instruction="private prompt", keep_intermediates=True), "sk-test-secret", self.out, self.stop)
        text = (summary.recovery_directory / "batch.json").read_text()
        self.assertNotIn("sk-test-secret", text)
        self.assertNotIn("private prompt", text)
        self.assertNotIn(str(self.root), text)

    def test_low_disk_space_before_runner(self):
        items = self.prepare(self.source())
        with patch("enhancer.batch.shutil.disk_usage", return_value=shutil_usage(100, 99, 1)), patch("enhancer.batch.run_job") as runner:
            with self.assertRaisesRegex(EnhancerError, "space"):
                self.run_items(items, runner=runner)
        runner.assert_not_called()

    def test_traversal_path_rejected_before_runner(self):
        items = self.prepare(self.source())
        items = (replace(items[0], relative_path=Path("../escape.png")),)
        with patch("enhancer.batch.run_job") as runner, self.assertRaisesRegex(EnhancerError, "Unsafe"):
            self.run_items(items, runner=runner)
        runner.assert_not_called()

    def test_collect_handles_spaces_unicode_duplicates_and_hardlinks(self):
        path = self.source("space ü {a}.png")
        alias = path.with_name("alias.png")
        os.link(path, alias)
        files, notes = collect_files((path, path.parent, alias))
        self.assertEqual(files, [path, alias])
        self.assertEqual(notes, [])
        self.assertEqual(collect_files((path,), files)[0], [])

    def test_collect_skips_symlink_loops_and_nonimages(self):
        path = self.source()
        (path.parent / "ignore.txt").write_text("ignored")
        try:
            (path.parent / "loop").symlink_to(path.parent, target_is_directory=True)
        except OSError:
            pass
        files, notes = collect_files((path.parent,))
        self.assertEqual(files, [path])

    def test_folder_scan_skips_nested_output_and_allows_explicit_selection(self):
        source = self.source()
        output = source.parent / 'enhanced'
        output.mkdir()
        previous = output / source.name
        previous.write_bytes(source.read_bytes())
        self.assertEqual(collect_files((source.parent,), output_directory=output), ([source], []))
        self.assertEqual(collect_files((output,), output_directory=output), ([previous], []))

    def test_file_count_limit(self):
        paths = [self.source(f"{i}.png") for i in range(3)]
        with patch("enhancer.batch.MAX_BATCH_FILES", 2):
            files, notes = collect_files(paths)
        self.assertEqual(len(files), 2)
        self.assertTrue(notes)

    def test_cancel_during_preflight(self):
        paths = [self.source(f"{i}.png") for i in range(3)]
        def emit(event):
            if event.state == "Ready":
                self.stop.set()
        items = prepare_batch(tuple(paths), self.settings, self.stop, emit)
        self.assertEqual(len(items), 1)

    def test_repeated_batches_never_overwrite(self):
        items = self.prepare(self.source())
        one = self.run_items(items)
        two = self.run_items(items)
        self.assertEqual(one.directory, two.directory)
        self.assertNotEqual(one.last_result.original_path, two.last_result.original_path)
        self.assertTrue(one.last_result.original_path.exists())
        self.assertTrue(two.last_result.original_path.exists())


def shutil_usage(total, used, free):
    from collections import namedtuple
    return namedtuple("usage", "total used free")(total, used, free)

class ReparsePointTests(unittest.TestCase):
    def test_windows_reparse_point_not_followed(self):
        from types import SimpleNamespace
        from enhancer.batch import is_link
        with patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'lstat', return_value=SimpleNamespace(st_file_attributes=0x400)):
            self.assertTrue(is_link(Path('junction')))

class BatchApiTests(unittest.TestCase):
    setUp = BatchTests.setUp
    source = BatchTests.source
    prepare = BatchTests.prepare
    run_items = BatchTests.run_items
    def test_separate_api_requests_and_stop_on_rate_limit(self):
        import httpx
        from test_api import response_body, fields_from_request
        self.settings = Settings(reuse_identical=False)
        paths = [self.source(f'{i}.png') for i in range(3)]
        items = self.prepare(*paths)
        calls = []
        def handler(request):
            fields = fields_from_request(request)
            calls.append(fields)
            self.assertEqual(fields['n'], b'1')
            if len(calls) == 2:
                return httpx.Response(429, json={'error': {'message': 'limit reached'}})
            size = tuple(int(n) for n in fields['size'].decode().split('x'))
            return httpx.Response(200, content=response_body(size))
        transport = httpx.MockTransport(handler)
        def runner(texture, settings, key, directory):
            return run_job(texture, settings, key, directory, transport=transport)
        summary = run_batch(items, self.settings, 'sk-test-key', self.out, self.stop, self.events.append, runner=runner)
        self.assertEqual((summary.completed, summary.failed), (1, 1))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(list(summary.directory.glob('*.png'))), 1)
        self.assertEqual([r['status'] for r in json.loads((summary.recovery_directory / 'batch.json').read_text())['items']], ['done', 'failed', 'pending'])

    def test_manifest_failure_does_not_reclassify_saved_image_as_failed(self):
        items = self.prepare(self.source('one.png'), self.source('two.png'))
        from enhancer.storage import write_atomic as original_write
        count = []
        def write(data, destination, *args, **kwargs):
            if Path(destination).name == 'batch.json':
                count.append(1)
                if len(count) == 3:
                    raise OSError('disk error after native save')
            return original_write(data, destination, *args, **kwargs)
        with patch('enhancer.batch.write_atomic', side_effect=write):
            summary = self.run_items(items)
        self.assertEqual((summary.completed, summary.failed), (1, 0))
        self.assertTrue(summary.last_result.original_path.exists())
        self.assertEqual([e.state for e in self.events if e.state in ('Done','Failed')], ['Done'])
        self.assertIn('summary', summary.warning)

class CanvasBoundaryTests(unittest.TestCase):
    def test_alignment_near_api_pixel_limit(self):
        from enhancer.core import plan_image
        for source, settings in (((2395, 3462), Settings(target_edge=2048)), ((1197, 1731), Settings(scale=2))):
            plan = plan_settings(source, settings)
            self.assertLessEqual(plan.canvas_size[0] * plan.canvas_size[1], 8294400)
            self.assertEqual(plan.target_size, target_dimensions(source, settings.scale, settings.target_edge))
            self.assertTrue(plan.resampled_output)
