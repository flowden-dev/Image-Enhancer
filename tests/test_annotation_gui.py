"""Folder annotations through the real desktop queue, editor and consent dialog."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from enhancer.annotation_gui import AnnotationEditor
from enhancer.annotations import AnnotationStore, ImageAnnotation
from enhancer.api import ApiImage, run_job
from enhancer.core import load_texture
from enhancer.gui import NEURAL_METHOD
import test_gui as gui_tests


class AnnotationGuiTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)
    tearDown = gui_tests.GuiTests.tearDown
    wait = gui_tests.GuiTests.wait
    result_for = gui_tests.GuiTests.result_for

    def setUp(self):
        gui_tests.GuiTests.setUp(self)
        self.app._annotation_store = AnnotationStore(self.root / "app-data" / "annotations")
        self.srcdir = self.root / "icons"
        self.srcdir.mkdir()
        self.files = [self.srcdir / name for name in ("back.tga", "menu.tga")]
        for path in self.files:
            Image.new("RGBA", (35, 35), (15, 100, 175, 255)).save(path)
        self.app.method.set(NEURAL_METHOD)
        self.app.output_size.set("8× original")
        self.app._runner = run_job

    def add(self):
        self.app.add_paths((self.srcdir,))
        self.wait()

    def select(self, index):
        self.app.batch_tree.selection_set(str(index))
        self.app.update()
        self.app._batch_selection()

    def edit_result(self, result):
        with patch("enhancer.gui.AnnotationEditor", return_value=SimpleNamespace(result=result)), \
             patch.object(self.app, "wait_window"):
            self.app.annotate_button.invoke()

    def test_local_queue_charges_only_annotated_row_and_edit_reruns_only_it(self):
        self.add()
        self.select(1)
        correct = ImageAnnotation(load_texture(self.files[1]).sha256, "Menü", area=(4, 8, 31, 22))
        self.edit_result((correct, False))
        self.assertEqual(self.app.batch_tree.set("1", "annotation"), "Menü")
        self.assertIn("OpenAI", self.app.request_hint.get())

        def respond(key, upload, plan, settings, alpha, **kwargs):
            return ApiImage(Image.new("RGBA", plan.canvas_size, "white"), "mock", {})

        with patch("enhancer.api.request_image", side_effect=respond) as api:
            self.app.key.set("")
            self.app.run_button.invoke()
            self.wait()
            api.assert_not_called()
            self.assertIn("API Settings", self.dialogs["showwarning"].call_args.args[1])
            self.dialogs["askyesno"].assert_not_called()
            self.app.key.set("sk-offline")
            self.dialogs["askyesno"].return_value = False
            self.app.run_button.invoke()
            self.wait()
            self.assertIn("up to 1 paid image edit?", self.dialogs["askyesno"].call_args.args[1])
            self.assertIn("Files processed locally: 1", self.dialogs["askyesno"].call_args.args[1])
            self.assertIn("menu.tga", self.dialogs["askyesno"].call_args.args[1])
            self.assertFalse(self.app.recovery_directory.exists())
            api.assert_not_called()
            self.dialogs["askyesno"].return_value = True
            self.app.run_button.invoke()
            self.wait()
            api.assert_called_once()
            self.assertEqual(api.call_args.args[3].annotation, correct)
            self.assertEqual(self.app._batch_done, {0, 1})
            self.select(1)
            self.edit_result((ImageAnnotation(correct.source_sha256, "Menü", "Keep both dots."), False))
            self.assertEqual(self.app.batch_tree.set("1", "state"), "Queued")
            self.assertEqual(self.app.batch_tree.set("0", "state"), "Done")
            self.app.run_button.invoke()
            self.wait()
            self.assertEqual(api.call_count, 2)
            self.assertEqual({p.name for p in self.app.recovery_directory.iterdir()}, {"back.tga", "menu.tga", "menu (2).tga"})

    def test_selection_removal_restoration_and_translation(self):
        self.add()
        self.select(1)
        original = ImageAnnotation(load_texture(self.files[1]).sha256, "Menü")
        self.edit_result((original, True))
        self.edit_result(None)
        self.select(0)
        self.app.remove_batch()
        self.assertEqual(self.app.batch_tree.set("0", "annotation"), "Menü")
        self.select(0)
        self.app.task.set("Translate text")
        self.assertEqual(str(self.app.annotate_button["state"]), "disabled")
        self.app.task.set("Enhance artwork")
        self.app.clear_batch()
        self.add()
        self.assertEqual(self.app.batch_tree.set("1", "annotation"), "Menü")
        self.assertEqual(self.app.batch_tree.set("0", "annotation"), "")
        self.app.batch_tree.selection_set("0", "1")
        self.app.update()
        self.assertEqual(str(self.app.annotate_button["state"]), "disabled")
        self.select(1)
        self.edit_result((None, False))
        self.app.clear_batch()
        self.add()
        self.assertFalse(self.app._annotations)

    def test_editor_accepts_unicode_and_draws_optional_source_pixel_area(self):
        self.app.deiconify()
        editor = AnnotationEditor(self.app, load_texture(self.files[1]), None, False)
        self.app.update()
        editor.exact.insert("1.0", "Menü")
        editor.note.insert("1.0", "Two dots above the ü.")
        canvas = editor.preview
        canvas.fit()
        def point(x, y):
            bx, by, *_ = canvas.screen_box((x, y, x, y))
            return SimpleNamespace(x=bx-canvas.canvasx(0), y=by-canvas.canvasy(0))
        editor._start_area(point(30.8, 21.8))
        editor._drag_area(point(4.1, 10.1))
        editor._finish_area(point(4.1, 10.1))
        self.assertEqual(editor.area, (4, 10, 31, 22))
        editor.accept()
        self.assertEqual(editor.result[0].exact_text, "Menü")
        self.assertEqual(editor.result[0].area, (4, 10, 31, 22))
        self.assertFalse(editor.result[1])

    def test_find_file_cycles_matches_without_changing_queue_order(self):
        self.add()
        self.app.file_search.insert(0, ".TGA")
        self.app._find_file()
        first = self.app.batch_tree.selection()
        self.app._find_file()
        self.assertNotEqual(self.app.batch_tree.selection(), first)
        self.app._find_file()
        self.assertEqual(self.app.batch_tree.selection(), first)
        self.app.file_search.delete(0, "end")
        self.app.file_search.insert(0, "MeNu")
        self.app._find_file()
        self.assertEqual(self.app.batch_tree.selection(), ("1",))
        self.assertIn(str(self.files[1]), self.app.batch_detail.get())
        self.assertEqual(self.app._batch_paths, self.files)
        self.app.file_search.insert("end", "missing")
        self.app._find_file()
        self.assertEqual(self.app.batch_tree.selection(), ())
        self.assertEqual(str(self.app.annotate_button["state"]), "disabled")

    def test_editor_saving_an_unchanged_note_keeps_done_row_done(self):
        self.add()
        self.select(1)
        note = ImageAnnotation(load_texture(self.files[1]).sha256, "Menü")
        self.edit_result((note, False))
        self.app.batch_tree.set("1", "state", "Done")
        self.app._batch_done.add(1)
        self.edit_result((note, True))
        self.assertEqual(self.app.batch_tree.set("1", "state"), "Done")
        self.assertEqual(self.app._batch_done, {1})
