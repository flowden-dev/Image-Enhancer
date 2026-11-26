"""Source drift bounds, size selection and elapsed-clock regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from enhancer.api import ApiImage, run_job
from enhancer.core import EnhancerError, Settings, build_prompt, load_texture, plan_settings
from enhancer.fidelity import protect_source
from enhancer.formats.common import resize_artwork
from enhancer.gui import format_elapsed
from enhancer.batch import BatchEvent
import test_gui as gui_tests


class FidelityTests(unittest.TestCase):
    def test_rgb_limits_random_extremes_and_strips(self):
        rng = np.random.default_rng(61)
        source = Image.fromarray(rng.integers(0, 256, (131, 43, 4), dtype=np.uint8))
        edited = Image.fromarray(rng.integers(0, 256, (262, 86, 4), dtype=np.uint8))
        base = np.array(resize_artwork(source, edited.size).convert('RGB'), dtype=np.int16)
        unchanged = edited.tobytes()
        for protection, limit in (('strict', 12), ('balanced', 32)):
            result = protect_source(edited, source, protection)
            delta = np.array(result.convert('RGB'), dtype=np.int16) - base
            self.assertLessEqual(int(np.abs(delta).max()), limit)
            self.assertEqual(result.getchannel('A').tobytes(), edited.getchannel('A').tobytes())
        self.assertEqual(edited.tobytes(), unchanged)

    def test_unrestricted_preserves_result(self):
        source = Image.new('RGB', (4, 3), 'white')
        edited = Image.new('RGB', (8, 6), 'black')
        self.assertIs(protect_source(edited, source, 'off'), edited)
        protected = protect_source(edited, source, 'strict')
        self.assertEqual(protected.mode, 'RGB')
        self.assertEqual(protected.getpixel((0, 0)), (243, 243, 243))

    def test_small_corrections_retained(self):
        source = Image.new('RGBA', (3, 2), (120, 130, 140, 255))
        edited = Image.new('RGB', (6, 4), (123, 125, 140))
        self.assertEqual(protect_source(edited, source, 'strict').tobytes(), edited.tobytes())

    def test_settings_prevent_conflicting_edits(self):
        with self.assertRaisesRegex(EnhancerError, 'Additional edits'):
            Settings(instruction='replace the logo').validate()
        Settings(instruction='replace the logo', source_protection='off').validate()
        Settings(instruction='ignored', method='nearest').validate()
        with self.assertRaises(EnhancerError):
            Settings(source_protection='bad').validate()

    def test_prompt_preserves_faded_art_and_small_text(self):
        settings = Settings(source_protection="strict")
        prompt = build_prompt(plan_settings((1024, 768), settings), settings)
        self.assertIn('keep it equally faint', prompt)
        self.assertIn('copyright', prompt)
        self.assertIn('Do not invent detail', prompt)

    def test_pipeline_saves_guarded_png_and_keeps_raw(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'faded.png'
            Image.new('RGBA', (32, 24), (230, 230, 230, 255)).save(path)
            texture = load_texture(path)
            settings = Settings(keep_intermediates=True, source_protection="strict")
            plan = plan_settings(texture.image.size, settings)
            raw = Image.new('RGBA', plan.canvas_size, (0, 0, 0, 255))
            with patch('enhancer.api.request_image', return_value=ApiImage(raw, 'mock', {})) as request:
                result = run_job(texture, settings, 'sk-test', root / 'out')
            request.assert_called_once()
            self.assertEqual(result.image.getpixel((0, 0)), (218, 218, 218))
            with Image.open(result.raw_path) as recovered:
                self.assertEqual(recovered.getpixel((0, 0)), (0, 0, 0, 255))
            with Image.open(result.original_path) as saved:
                self.assertEqual(saved.convert('RGB').getpixel((0, 0)), (218, 218, 218))
            metadata = json.loads(result.metadata_path.read_text())
            self.assertEqual(metadata['source_protection'], 'strict')
            self.assertGreaterEqual(metadata['processing_seconds'], 0)

    def test_elapsed_format(self):
        for value, expected in ((0, '00:00:00'), (65.9, '00:01:05'), (3661, '01:01:01'),
                                (360000, '100:00:00'), (-2, '00:00:00')):
            self.assertEqual(format_elapsed(value), expected)


class ControlsTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)
    setUp = gui_tests.GuiTests.setUp
    tearDown = gui_tests.GuiTests.tearDown
    wait = gui_tests.GuiTests.wait
    result_for = gui_tests.GuiTests.result_for

    def test_size_choices_produce_exact_sizes(self):
        self.app.load_file(self.source)
        self.app.custom_edge.set('bad hidden value')
        for option, expected in (('2× original', (128, 64)), ('4× original', (256, 128)),
                                 ('2K (2048)', (2048, 1024)), ('4K (4096)', (4096, 2048))):
            self.app.output_size.set(option)
            self.assertEqual(plan_settings((64, 32), self.app._settings()).target_size, expected)
            self.assertEqual(self.app.custom_size_frame.winfo_manager(), '')
            self.assertIn(f'{expected[0]} x {expected[1]}', self.app.dimensions.get())
        self.app.output_size.set('Custom longest edge')
        self.assertEqual(self.app.custom_size_frame.winfo_manager(), 'pack')
        with self.assertRaises(EnhancerError):
            self.app._settings()
        self.app.custom_edge.set('800')
        self.assertEqual(plan_settings((64, 32), self.app._settings()).target_size, (800, 400))
        self.app.output_size.set('2× original')
        self.assertEqual(self.app._settings().target_edge, None)

    def test_custom_factor_field_updates_preview_and_ignores_hidden_edge(self):
        self.app.load_file(self.source)
        self.app.custom_edge.set('unused')
        self.app.output_size.set('Custom scale factor')
        self.assertEqual(self.app.custom_scale_frame.winfo_manager(), 'pack')
        self.assertEqual(self.app.custom_size_frame.winfo_manager(), '')
        for value, dimensions in (('10', '640 x 320'), ('12x', '768 x 384'), ('2,5', '160 x 80')):
            self.app.custom_scale.set(value)
            self.assertIn(dimensions, self.app.dimensions.get())
        self.app.custom_scale.set('bad')
        with self.assertRaises(EnhancerError):
            self.app._settings()
        self.app.output_size.set('8× original')
        self.assertEqual(self.app.custom_scale_frame.winfo_manager(), '')
        self.assertIn('512 x 256', self.app.dimensions.get())

    def test_elapsed_ticks_freezes_and_resets(self):
        with patch('enhancer.gui.time.monotonic', return_value=100):
            self.app._set_busy(True)
        with patch('enhancer.gui.time.monotonic', return_value=165):
            self.app._update_elapsed()
            self.assertEqual(self.app.elapsed.get(), 'Elapsed: 00:01:05')
            self.app._set_busy(True)
            self.assertEqual(self.app._started_at, 100)
            self.app._set_busy(False)
        with patch('enhancer.gui.time.monotonic', return_value=1000):
            self.app._update_elapsed()
            self.assertEqual(self.app.elapsed.get(), 'Elapsed: 00:01:05')
            self.app._set_busy(True)
            self.assertEqual(self.app.elapsed.get(), 'Elapsed: 00:00:00')
            self.app._set_busy(False)

    def test_batch_file_clock_uses_event_time(self):
        self.app._batch_paths = [self.source]
        self.app._batch_rows = (0,)
        self.app.batch_tree.insert('', 'end', iid='0', values=('texture.png', '', 'Ready'))
        with patch('enhancer.gui.time.monotonic', return_value=10):
            self.app._set_busy(True)
        with patch('enhancer.gui.time.monotonic', return_value=20):
            self.app._batch_event(BatchEvent(0, 'Processing'))
        with patch('enhancer.gui.time.monotonic', return_value=85):
            self.app._poll_batch()
            self.app._update_elapsed()
            self.assertEqual(self.app.elapsed.get(), 'Elapsed: 00:01:15  |  Current file: 00:01:05')
            self.app._batch_event(BatchEvent(0, 'Done'))
            self.app._poll_batch()
            self.app._set_busy(False)
            self.assertEqual(self.app.elapsed.get(), 'Elapsed: 00:01:15')

    def test_additional_edit_requires_explicit_protection_off(self):
        self.app.load_file(self.source)
        self.app.instruction.set('Translate PLAY')
        self.app.process()
        self.assertEqual(self.calls, [])
        self.dialogs['askyesno'].assert_not_called()
        self.app.protection.set('Off (allow edits)')
        self.app.process()
        self.wait()
        self.assertEqual(self.calls[0][1].source_protection, 'off')
