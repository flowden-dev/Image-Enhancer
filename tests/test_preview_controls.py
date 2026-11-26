"""Real desktop interactions for wheel zoom, right-drag and protected-area editing."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

from enhancer.preview import ImagePreview
from enhancer.translation_gui import RegionEditor
import test_gui as gui_tests


class PreviewTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)
    setUp = gui_tests.GuiTests.setUp
    tearDown = gui_tests.GuiTests.tearDown
    wait = gui_tests.GuiTests.wait
    result_for = gui_tests.GuiTests.result_for

    def prepare_preview(self):
        self.app.deiconify()
        self.app.update()
        canvas = self.app.input_preview
        canvas.set_image(Image.new("RGBA", (2000, 1200), (20, 40, 60, 255)))
        canvas.set_zoom(1.0)
        self.app.update()
        return canvas

    def test_pointer_anchored_wheel_zoom_and_linux_buttons(self):
        canvas = self.prepare_preview()
        x, y = canvas.winfo_width()//2, canvas.winfo_height()//2
        before = canvas.image_point(x, y)
        canvas._wheel(SimpleNamespace(x=x, y=y, delta=120, num=0))
        after = canvas.image_point(x, y)
        self.assertAlmostEqual(canvas.scale, 1.2)
        for a, b in zip(before, after):
            self.assertAlmostEqual(a, b, delta=2)
        canvas._wheel(SimpleNamespace(x=x, y=y, num=5))
        self.assertAlmostEqual(canvas.scale, 1.0)
        canvas._wheel(SimpleNamespace(x=x, y=y, num=4))
        self.assertGreater(canvas.scale, 1.0)

    def test_right_drag_moves_view_without_modifying_image(self):
        canvas = self.prepare_preview()
        original = canvas.source.tobytes()
        before = canvas.canvasx(0), canvas.canvasy(0)
        canvas._pan_start(SimpleNamespace(x=200, y=200))
        canvas._pan_move(SimpleNamespace(x=150, y=170))
        canvas._pan_end(None)
        canvas.render()
        after = canvas.canvasx(0), canvas.canvasy(0)
        self.assertEqual(after, (before[0]+50, before[1]+30))
        self.assertEqual(canvas.source.tobytes(), original)
        self.assertFalse(canvas._panning)

    def test_zoom_render_allocation_is_bounded_by_viewport(self):
        canvas = self.prepare_preview()
        canvas.set_zoom(32)
        self.assertLessEqual(canvas.photo.width(), canvas.winfo_width()+200)
        self.assertLessEqual(canvas.photo.height(), canvas.winfo_height()+200)
        canvas.fit()
        self.assertTrue(canvas.fit_mode)

    def test_refresh_keeps_zoom_but_new_source_fits(self):
        canvas = self.prepare_preview()
        canvas.set_zoom(2)
        canvas.set_image(canvas.source)
        self.assertEqual(canvas.scale, 2)
        canvas.set_image(Image.new("RGBA", (100, 60)))
        self.assertTrue(canvas.fit_mode)

    def test_translation_editor_tracks_both_kinds_and_undoes_last(self):
        self.app.deiconify(); self.app.update()
        editor = RegionEditor(self.app, Image.new("RGBA", (100, 80), "white"), ((5, 5, 50, 40),))
        try:
            editor.change_zoom("100%")
            editor.mode.set("Protect")
            editor.start_drag(SimpleNamespace(x=10, y=10))
            editor.end_drag(SimpleNamespace(x=20, y=20))
            self.assertEqual(editor.protected_regions, [(10, 10, 20, 20)])
            self.assertEqual(editor.regions, [(5, 5, 50, 40)])
            editor.undo()
            self.assertEqual(editor.protected_regions, [])
            self.assertEqual(editor.regions, [(5, 5, 50, 40)])
            editor.accept()
            self.assertEqual(editor.protected_result, ())
        finally:
            if editor.winfo_exists():
                editor.destroy()

    def test_all_text_with_protected_boxes_blocks_batch_before_confirmation(self):
        self.app.load_file(self.source)
        self.app.task.set("Translate text")
        self.app.translation_scope.set("All text (batch compatible)")
        self.app.protected_regions = ((2, 2, 15, 10),)
        self.app.add_paths((self.source,)); self.wait()
        self.app.process_batch()
        self.assertEqual(self.calls, [])
        self.dialogs["askyesno"].assert_not_called()

    def test_new_source_clears_protection_and_settings_snapshot_freezes_it(self):
        self.app.load_file(self.source)
        self.app.task.set("Translate text")
        self.app.text_regions = ((2, 2, 32, 20),)
        self.app.protected_regions = ((4, 4, 10, 10),)
        self.app.process(); self.wait()
        self.assertEqual(self.calls[0][1].protected_regions, ((4, 4, 10, 10),))
        self.app.load_file(self.source)
        self.assertEqual(self.app.protected_regions, ())

    def test_primary_action_visible_at_minimum_window_size(self):
        self.app.deiconify(); self.app.geometry("1000x640"); self.app.update()
        self.app.task.set("Translate text"); self.app.update()
        button = self.app.run_button
        self.assertTrue(button.winfo_viewable())
        self.assertGreaterEqual(button.winfo_rooty(), self.app.winfo_rooty())
        self.assertLessEqual(button.winfo_rooty()+button.winfo_height(), self.app.winfo_rooty()+self.app.winfo_height())
        self.assertTrue(self.app.save_button.winfo_viewable())
        self.assertLessEqual(self.app.save_button.winfo_rooty()+self.app.save_button.winfo_height(),
                             self.app.winfo_rooty()+self.app.winfo_height())

    def test_primary_action_uses_pending_queue_in_both_views_and_stays_locked(self):
        self.app.deiconify()
        self.app.load_file(self.source)
        self.app.update()
        with patch.object(self.app, "process") as single, patch.object(self.app, "process_batch") as batch:
            self.app.run_button.invoke()
            single.assert_called_once()
            self.app._batch_paths = [self.source]
            self.app.batch_tree.insert("", "end", iid="0", values=(self.source.name, "—", "Queued"))
            self.app.tabs.select(self.app.batch_tab)
            self.app.update()
            self.assertEqual(self.app.run_button["text"], "Run Queue")
            self.assertNotIn("1 request", self.app.request_hint.get())
            self.app.run_button.invoke()
            batch.assert_called_once()
            self.app._set_busy(True)
            self.app.tabs.select(0)
            self.app.update()
            self.assertEqual(str(self.app.run_button["state"]), "disabled")
            self.app.run_button.invoke()
            single.assert_called_once()
            self.app._set_busy(False)
