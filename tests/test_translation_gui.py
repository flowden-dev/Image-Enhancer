"""Real Tk interactions for translation mode, region drawing and import."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from enhancer.core import EnhancerError
from enhancer.translation_gui import RegionEditor, WordingEditor
import test_gui as gui_tests


class TranslationGuiTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)
    setUp = gui_tests.GuiTests.setUp
    tearDown = gui_tests.GuiTests.tearDown
    wait = gui_tests.GuiTests.wait
    result_for = gui_tests.GuiTests.result_for

    def setup_translation(self):
        self.app.load_file(self.source)
        self.app.task.set('Translate text')
        self.app.text_regions = ((5, 4, 25, 20),)

    def test_task_switch_defaults_original_and_restores_enhancement_choices(self):
        self.app.load_file(self.source)
        self.app.output_size.set('4× original')
        self.app.method.set('Local nearest')
        self.app.task.set('Translate text')
        self.assertEqual(self.app.output_size.get(), 'Original size')
        self.assertIn('Output: 64 x 32', self.app.dimensions.get())
        self.assertEqual(self.app.enhance_options.winfo_manager(), '')
        self.assertEqual(self.app.enhance_edits.winfo_manager(), '')
        self.assertEqual(self.app.translation_options.winfo_manager(), 'pack')
        self.assertIn('Translate', self.app.run_button['text'])
        self.app.output_size.set('2K (2048)')
        self.app.task.set('Enhance artwork')
        self.assertEqual(self.app.output_size.get(), '4× original')
        self.assertEqual(self.app.method.get(), 'Local nearest')
        self.app.task.set('Translate text')
        self.assertEqual(self.app.output_size.get(), '2K (2048)')

    def test_missing_areas_blocks_before_cost_confirmation(self):
        self.app.load_file(self.source)
        self.app.task.set('Translate text')
        self.app.process()
        self.assertEqual(self.calls, [])
        self.dialogs['askyesno'].assert_not_called()
        self.assertIn('Select Text Areas', self.dialogs['showwarning'].call_args.args[1])

    def test_translation_snapshot_overrides_old_alpha_and_guard(self):
        self.setup_translation()
        self.app.instruction.set('old enhancement instruction')
        self.app.process()
        self.wait()
        config = self.calls[0][1]
        self.assertEqual(config.task, 'translate')
        self.assertEqual(config.scale, 1)
        self.assertEqual(config.instruction, '')
        self.assertFalse(config.preserve_alpha)
        self.assertEqual(config.source_protection, 'off')
        self.assertEqual(config.text_regions, ((5,4,25,20),))
        self.assertIn('Auto-detect to German', self.dialogs['askyesno'].call_args.args[1])
        self.app.target_language.set('French')
        self.assertEqual(config.target_language, 'German')

    def test_new_source_clears_regions_but_keeps_language(self):
        self.setup_translation()
        self.app.target_language.set('French')
        self.app.load_file(self.source)
        self.assertEqual(self.app.text_regions, ())
        self.assertEqual(self.app.target_language.get(), 'French')

    def test_all_text_ignores_retained_regions_and_discloses_background_risk(self):
        self.setup_translation()
        self.app.translation_scope.set('All text (batch compatible)')
        config = self.app._settings()
        self.assertEqual(config.text_regions, ())
        self.app.process()
        self.wait()
        self.assertIn('AI may change background', self.dialogs['askyesno'].call_args.args[1])

    def test_preview_areas_follow_task_scope_and_new_source(self):
        self.setup_translation()
        self.app._update_translation_summary()
        self.assertEqual(len(self.app.input_preview.find_withtag('text_area')), 2)
        self.app.translation_scope.set('All text (batch compatible)')
        self.assertEqual(self.app.input_preview.find_withtag('text_area'), ())
        self.app.translation_scope.set('Selected text areas')
        self.assertEqual(len(self.app.input_preview.find_withtag('text_area')), 2)
        self.app.task.set('Enhance artwork')
        self.assertEqual(self.app.input_preview.find_withtag('text_area'), ())
        self.app.task.set('Translate text')
        self.app.load_file(self.source)
        self.assertEqual(self.app.input_preview.find_withtag('text_area'), ())

    def test_cancelled_region_and_wording_dialogs_preserve_previous_choices(self):
        self.setup_translation()
        self.app.replacements = 'PLAY => SPIELEN'
        self.app.protected_terms = 'Field Notes'
        with patch('enhancer.gui.RegionEditor', return_value=SimpleNamespace(result=None)), \
             patch('enhancer.gui.WordingEditor', return_value=SimpleNamespace(result=None)), \
             patch.object(self.app, 'wait_window'):
            self.app.select_text_areas()
            self.app.edit_wording()
        self.assertEqual(self.app.text_regions, ((5, 4, 25, 20),))
        self.assertEqual(self.app.replacements, 'PLAY => SPIELEN')
        self.assertEqual(self.app.protected_terms, 'Field Notes')

    def test_batch_selected_areas_blocked_before_paid_confirmation(self):
        self.setup_translation()
        self.app.add_paths((self.source,))
        self.wait()
        self.app.process_batch()
        self.assertEqual(self.calls, [])
        self.dialogs['askyesno'].assert_not_called()
        self.assertFalse(self.app._busy)

    def test_import_translation_restores_outside_pixels_and_new_alpha(self):
        self.setup_translation()
        edited = self.root/'translated.png'
        Image.new('RGBA', (64,32), (250,10,20,0)).save(edited)
        with patch('enhancer.gui.filedialog.askopenfilename', return_value=str(edited)):
            self.app.import_edited_png()
        self.dialogs['showerror'].assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.app.result.image.getpixel((0,0)), self.app.texture.image.getpixel((0,0)))
        self.assertEqual(self.app.result.image.getpixel((10,10))[3], 0)
        self.assertEqual(self.app.result.settings.task, 'translate')
        self.assertIsNone(self.app.result.saved_path)

    def test_region_editor_draw_reverse_undo_and_cancel(self):
        editor = RegionEditor(self.app, Image.new('RGBA', (100,80), 'white'))
        try:
            editor.zoom.set('100%');editor.render();editor.update()
            editor.start_drag(SimpleNamespace(x=40,y=30))
            editor.drag(SimpleNamespace(x=10,y=5))
            editor.end_drag(SimpleNamespace(x=10,y=5))
            self.assertEqual(editor.regions, [(10,5,40,30)])
            editor.undo()
            self.assertEqual(editor.regions, [])
            editor.start_drag(SimpleNamespace(x=200,y=200))
            editor.end_drag(SimpleNamespace(x=10,y=5))
            self.assertEqual(editor.regions, [])
            self.assertIsNone(editor.result)
        finally:
            editor.destroy()

    def test_region_editor_clamps_borders_rejects_overlap_and_accepts(self):
        editor = RegionEditor(self.app, Image.new('RGBA', (100,80), 'white'))
        try:
            editor.zoom.set('100%');editor.render();editor.update()
            editor.start_drag(SimpleNamespace(x=70,y=60))
            editor.end_drag(SimpleNamespace(x=500,y=500))
            self.assertEqual(editor.regions, [(70,60,100,80)])
            editor.start_drag(SimpleNamespace(x=75,y=65))
            editor.end_drag(SimpleNamespace(x=95,y=75))
            self.assertEqual(len(editor.regions), 1)
            editor.accept()
            self.assertEqual(editor.result, ((70,60,100,80),))
        finally:
            if editor.winfo_exists():
                editor.destroy()

    def test_region_editor_zoom_does_not_change_source_regions(self):
        editor = RegionEditor(self.app, Image.new('RGBA', (100,80), 'white'), ((10,5,40,30),))
        try:
            editor.zoom.set('200%');editor.render();editor.update()
            self.assertEqual(editor._preview_size, (200,160))
            self.assertEqual(editor.regions, [(10,5,40,30)])
            editor.clear();editor.accept()
            self.assertEqual(editor.result, ())
        finally:
            if editor.winfo_exists():
                editor.destroy()

    def test_region_editor_scrolled_selection_uses_image_coordinates(self):
        editor = RegionEditor(self.app, Image.new('RGBA', (2000, 1000), 'white'))
        try:
            editor.zoom.set('100%'); editor.render(); editor.update()
            editor.canvas.xview_moveto(0.25)
            editor.canvas.yview_moveto(0.25)
            editor.update()
            x, y = int(editor.canvas.canvasx(10)), int(editor.canvas.canvasy(20))
            self.assertGreater(x, 10)
            self.assertGreater(y, 20)
            editor.start_drag(SimpleNamespace(x=10, y=20))
            editor.end_drag(SimpleNamespace(x=60, y=70))
            self.assertEqual(editor.regions, [(x, y, x+50, y+50)])
        finally:
            editor.destroy()

    def test_wording_editor_validates_before_accept(self):
        editor = WordingEditor(self.app, 'PLAY', 'Field Notes')
        try:
            editor.accept()
            self.assertIsNone(editor.result)
            self.assertTrue(editor.winfo_exists())
            editor.replacements.delete('1.0','end')
            editor.replacements.insert('1.0','PLAY => SPIELEN')
            editor.accept()
            self.assertEqual(editor.result, ('PLAY => SPIELEN','Field Notes'))
        finally:
            if editor.winfo_exists():
                editor.destroy()
