from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

from PIL import Image
from enhancer.api import ApiImage, run_job
from enhancer.core import EnhancerError
from format_fixtures import tga_fixture
import test_gui as gui_tests


class BatchGuiTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)
    setUp = gui_tests.GuiTests.setUp
    tearDown = gui_tests.GuiTests.tearDown
    wait = gui_tests.GuiTests.wait
    result_for = gui_tests.GuiTests.result_for

    def queue(self, *paths):
        self.app.add_paths(tuple(paths))
        self.wait()

    def test_native_drop_extension_loaded(self):
        self.assertTrue(self.app.drag_drop_enabled)

    def test_single_drop_with_spaces_and_braces(self):
        path = self.root / 'ü space {icon}.png'
        path.write_bytes(self.source.read_bytes())
        data = self.app.tk.call('list', str(path))
        # Tk may return a tuple with wantobjects; the binding's %D arrives as a string
        self.app.tk.call('set', '_test_drop', data)
        text = self.app.tk.eval('set _test_drop')
        self.assertEqual(self.app._drop(SimpleNamespace(data=text)), 'copy')
        self.wait()
        self.assertEqual(self.app.texture.path, path)

    def test_multiple_drop_queues_files_without_api(self):
        other = self.root / 'another image.png'
        other.write_bytes(self.source.read_bytes())
        self.app.tk.call('set', '_test_drop', self.app.tk.call('list', str(self.source), str(other)))
        self.app._drop(SimpleNamespace(data=self.app.tk.eval('set _test_drop')))
        self.wait()
        self.assertEqual(len(self.app._batch_paths), 2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.app.batch_tree.set('0', 'state'), 'Queued')

    def test_batch_local_real_output_and_no_cost_prompt(self):
        self.queue(self.source)
        self.app.method.set('Local nearest')
        self.app._runner = run_job
        self.app.process_batch()
        self.wait()
        self.assertEqual(self.app.batch_tree.set('0', 'state'), 'Done')
        self.assertTrue(self.app.result.original_path.exists())
        self.dialogs['askyesno'].assert_not_called()

    def test_declined_batch_confirmation_sends_no_request(self):
        self.queue(self.source)
        self.dialogs['askyesno'].return_value = False
        self.app.process_batch()
        self.wait()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.app.batch_tree.set('0', 'state'), 'Ready')
        self.assertIn('Process 1 image and', self.dialogs['askyesno'].call_args.args[1])

    def test_add_folder_custom_scale_generates_once_and_keeps_locale_names(self):
        folder = self.root / 'toolbar'
        folder.mkdir()
        names = ('icon (en).tga', 'icon (de).tga', 'icon (fr).tga')
        for name in names:
            (folder/name).write_bytes(tga_fixture())
        with patch('enhancer.gui.filedialog.askdirectory', return_value=str(folder)):
            self.app.add_folder()
        self.wait()
        self.app.output_size.set('Custom scale factor')
        self.app.custom_scale.set('12')
        self.app._runner = run_job
        def respond(key, upload, plan, settings, alpha, **kwargs):
            return ApiImage(Image.new('RGBA', plan.canvas_size, 'white'), 'offline', {})
        with patch('enhancer.api.request_image', side_effect=respond) as api:
            self.app.run_button.invoke()
            self.wait()
        api.assert_called_once()
        confirmation = self.dialogs['askyesno'].call_args.args[1]
        self.assertIn('up to 1 paid image edit', confirmation)
        self.assertIn('Matching images sharing an earlier result: 2', confirmation)
        outputs = self.app.recovery_directory
        self.assertEqual({p.name for p in outputs.iterdir()}, set(names))
        self.assertEqual(len({(outputs/name).read_bytes() for name in names}), 1)
        self.assertEqual(self.app.result.image.size, (204, 132))

    def test_8x_folder_uses_each_files_dimensions_with_reuse_enabled(self):
        folder = self.root / 'mixed'
        folder.mkdir()
        sizes = {'a-square.tga': (35, 35), 'b-wide.tga': (36, 35),
                 'c-tall.tga': (35, 36), 'd-square-copy.tga': (35, 35)}
        for name, size in sizes.items():
            # The two rectangles have identical RGBA byte streams and different shapes.
            Image.new('RGBA', size, (20, 80, 100, 180)).save(folder / name)
        self.queue(folder)
        preview_index = self.app._batch_paths.index(folder / 'c-tall.tga')
        self.app.batch_tree.selection_set(str(preview_index))
        self.app.preview_selected()
        self.assertEqual(self.app.texture.image.size, (35, 36))
        self.app.output_size.set('8× original')
        self.app._runner = run_job

        def respond(key, upload, plan, settings, alpha, **kwargs):
            return ApiImage(Image.new('RGBA', plan.canvas_size, 'white'), 'offline', {})

        with patch('enhancer.api.request_image', side_effect=respond) as api:
            self.app.run_button.invoke()
            self.wait()
        self.assertCountEqual([call.args[2].source_size for call in api.call_args_list],
                              [(35, 35), (36, 35), (35, 36)])
        output = self.app.recovery_directory
        self.assertEqual({p.name for p in output.iterdir()}, set(sizes))
        for index, source in enumerate(self.app._batch_paths):
            with self.subTest(source=source.name):
                width, height = sizes[source.name]
                self.assertEqual(self.app.batch_tree.set(str(index), 'input'), f'{width} x {height}')
                self.assertEqual(self.app.batch_tree.set(str(index), 'target'), f'{width * 8} x {height * 8}')
                self.assertEqual(self.app.batch_tree.set(str(index), 'state'), 'Done')
                with Image.open(output / source.name) as saved:
                    saved.load()
                    self.assertEqual(saved.size, (width * 8, height * 8))
        self.assertEqual((output / 'a-square.tga').read_bytes(), (output / 'd-square-copy.tga').read_bytes())

    def test_8x_neural_button_uses_selected_input_and_displays_saved_size(self):
        path = self.root / 'button.tga'
        Image.new('RGBA', (140, 140), (25, 80, 200, 150)).save(path)
        self.queue(path)
        self.app.method.set('Neural artwork (free)')
        self.app.output_size.set('8× original')
        self.app._runner = run_job
        with patch('enhancer.api.request_image') as request:
            self.app.run_button.invoke()
            self.wait()
        request.assert_not_called()
        self.assertEqual(self.app.batch_tree.set('0', 'input'), '140 x 140')
        self.assertEqual(self.app.batch_tree.set('0', 'target'), '1120 x 1120')
        self.assertIn('1120 × 1120', self.app.result_caption.get())
        with Image.open(self.app.result.original_path) as saved:
            self.assertEqual(saved.size, (1120, 1120))

    def test_processing_failure_is_not_implicitly_retried(self):
        self.queue(self.source)
        self.app.method.set('Local nearest')
        calls = []
        def fail(*args):
            calls.append(1)
            raise EnhancerError('failed')
        self.app._runner = fail
        self.app.process_batch()
        self.wait()
        self.assertEqual(self.app.batch_tree.set('0', 'state'), 'Failed')
        self.dialogs['askyesno'].return_value = False
        self.app.process_batch()
        self.wait()
        self.assertEqual(calls, [1])
        self.assertFalse(self.app._busy)
        self.assertEqual(self.app.batch_tree.set('0', 'state'), 'Failed')

    def test_no_key_for_api_batch(self):
        self.queue(self.source)
        self.app.key.set('')
        self.app.process_batch()
        self.wait()
        self.assertFalse(self.app._busy)
        self.assertEqual(self.calls, [])

    def test_busy_drop_refused_and_controls_locked(self):
        self.app._set_busy(True)
        self.assertEqual(self.app._drop(SimpleNamespace(data=str(self.source))), 'refuse_drop')
        self.assertEqual(str(self.app.batch_run_button['state']), 'disabled')
        self.app._set_busy(False)

    def test_queue_dedup_remove_and_readd(self):
        self.queue(self.source, self.source)
        self.assertEqual(len(self.app._batch_paths), 1)
        self.app.batch_tree.selection_set('0')
        self.app.remove_batch()
        self.assertEqual(self.app._batch_paths, [])
        self.queue(self.source)
        self.assertEqual(len(self.app._batch_paths), 1)

    def test_resolution_preview_and_import_use_same_size(self):
        self.app.load_file(self.source)
        self.app.output_size.set('2K (2048)')
        self.assertIn('2048 x 1024', self.app.dimensions.get())
        self.app.output_size.set('4K (4096)')
        self.assertIn('4096 x 2048', self.app.dimensions.get())
        self.app.output_size.set('Custom longest edge')
        self.app.custom_edge.set('512')
        self.assertIn('512 x 256', self.app.dimensions.get())
        self.assertEqual(self.app._settings().target_edge, 512)

    def test_invalid_input_does_not_prompt_for_cost(self):
        path = self.root / 'bad.dds'
        path.write_bytes(b'bad')
        self.queue(path)
        self.app.process_batch()
        self.wait()
        self.assertEqual(self.app.batch_tree.set('0', 'state'), 'Invalid')
        self.dialogs['askyesno'].assert_not_called()

    def test_worker_failure_restores_batch_controls(self):
        self.queue(self.source)
        with patch('enhancer.gui.threading.Thread.start', side_effect=RuntimeError('no thread')):
            self.app.process_batch()
        self.assertFalse(self.app._busy)
        self.assertFalse(self.app._batch_active)
        self.assertEqual(self.app._batch_key, '')
        self.assertEqual(str(self.app.batch_stop_button['state']), 'disabled')

    def test_direct_preview_refresh_does_not_orphan_timer(self):
        self.app._schedule_preview()
        pending = self.app._preview_id
        self.app._refresh_previews()
        self.assertNotIn(pending, self.app.tk.call('after', 'info'))
        self.assertIsNone(self.app._preview_id)

    def test_no_preview_timer_scheduled_after_destroy(self):
        self.app.destroy()
        self.app._schedule_preview()
        self.assertIsNone(self.app._preview_id)
