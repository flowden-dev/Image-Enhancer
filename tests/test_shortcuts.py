"""Exercise keyboard events with real widget focus, modal editors and saved output."""
from tkinter import ttk
import unittest
from unittest.mock import patch

from PIL import Image

from enhancer.annotation_gui import AnnotationEditor
from enhancer.api import run_job
from enhancer.shortcuts import primary_modifier
from enhancer.translation_gui import RegionEditor, WordingEditor
import test_gui as gui_tests


class ShortcutTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)
    tearDown = gui_tests.GuiTests.tearDown
    wait = gui_tests.GuiTests.wait
    result_for = gui_tests.GuiTests.result_for

    def setUp(self):
        gui_tests.GuiTests.setUp(self)
        self.app.deiconify()
        self.app.update()
        self.app.focus_force()
        self.primary = primary_modifier(self.app)
        folder = self.root / "buttons"
        folder.mkdir()
        self.paths = [folder / name for name in ("back.png", "menu.png", "next.png")]
        for i, path in enumerate(self.paths):
            Image.new("RGB", (21, 13), (40 * i, 100, 210)).save(path)
        self.app._runner = run_job
        self.app.method.set("Pixel art (free)")
        self.app.add_paths((folder,))
        self.wait()

    def press(self, sequence, widget=None):
        target = widget or self.app.focus_get()
        self.assertIsNotNone(target, f"No focus for {sequence}")
        target.event_generate(f"<{sequence}>")
        self.app.update()

    def edit(self, key, *, cancel=False):
        opened = []
        failures = []

        def type_label():
            editor = next((w for w in self.app.winfo_children() if isinstance(w, AnnotationEditor)), None)
            if editor is None:
                return
            opened.append(True)
            try:
                self.assertIs(self.app.focus_get(), editor.exact)
                editor.exact.insert("1.0", "Menü")
                self.press("Return", editor.exact)
                self.press("a", editor.exact)
                self.press("Left", editor.exact)
                self.assertEqual(editor.exact.get("1.0", "end-1c"), "Menü\na")
                self.assertEqual(editor.exact.index("insert"), "2.0")
                self.press("Escape" if cancel else f"{self.primary}-Return", editor.exact)
                self.assertFalse(editor.winfo_exists())
            except Exception as exc:
                failures.append(exc)
            finally:
                if editor.winfo_exists():
                    editor.destroy()

        timer = self.app.after(70, type_label)
        try:
            self.press(key)
        finally:
            self.app.after_cancel(timer)
        if failures:
            raise failures[0]
        self.assertEqual(opened, [True])

    def test_review_annotate_save_and_continue_without_mouse(self):
        app = self.app
        self.assertEqual(app.batch_tree.selection(), ("0",))
        self.assertIs(app.focus_get(), app.batch_tree)
        self.press("Down")
        self.assertEqual(app.batch_tree.selection(), ("1",))
        self.press("Return")
        self.assertIs(app.focus_get(), app.input_preview)
        self.assertEqual(app.texture.path, self.paths[1])
        # A hidden file selection must never redirect an edit of the displayed image.
        app.batch_tree.selection_set("0")
        with patch("enhancer.api.request_image") as api:
            self.edit("a")
            self.assertEqual(set(app._annotations), {self.paths[1]})
            self.assertEqual(app._annotations[self.paths[1]].exact_text, "Menü\na")
            self.assertIs(app.focus_get(), app.input_preview)
            self.press("Right")
            self.assertEqual(app.texture.path, self.paths[2])
            self.edit("a", cancel=True)
            self.press("Escape")
            self.assertIs(app.focus_get(), app.batch_tree)
            self.assertEqual(app.batch_tree.selection(), ("2",))
            self.assertEqual(set(app._annotations), {self.paths[1]})
            api.assert_not_called()

    def test_find_keeps_typing_and_f2_annotates_match(self):
        app = self.app
        self.press("Return")
        self.press(f"{self.primary}-f")
        self.assertIs(app.focus_get(), app.file_search)
        self.press("a")
        self.assertEqual(app.file_search.get(), "a")
        self.press("Left")
        self.assertEqual(app.file_search.index("insert"), 0)
        app.file_search.delete(0, "end")
        app.file_search.insert(0, "menu")
        self.press("Return")
        self.assertEqual(app.batch_tree.selection(), ("1",))
        self.edit("F2")
        self.assertEqual(set(app._annotations), {self.paths[1]})
        self.assertIs(app.focus_get(), app.file_search)

    def test_arrows_zoom_saved_output_and_save_copy(self):
        app = self.app
        app.run_button.invoke()
        self.wait()
        self.press("Return", app.batch_tree)
        self.wait()
        app.tabs.focus_force()
        self.press("Left")
        self.wait()
        self.assertEqual(app.tabs.select(), str(app.preview_tab))
        self.assertEqual(app._saved_view.path.name, "menu.png")
        self.press("KeyPress-1")
        self.assertTrue(all(view.scale == 1 and not view.fit_mode for view in (app.input_preview, app.output_preview)))
        self.press("KeyPress-0")
        self.assertTrue(all(view.fit_mode for view in (app.input_preview, app.output_preview)))
        self.press("Alt-Left")
        self.wait()
        self.assertEqual(app._saved_view.path.name, "back.png")
        self.press("Left")
        self.assertEqual(app.preview_position.get(), "1 / 3")
        copy = self.root / "saved.png"
        with patch("enhancer.gui.filedialog.asksaveasfilename", return_value=str(copy)):
            self.press(f"{self.primary}-s")
            self.wait()
        with Image.open(copy) as result:
            self.assertEqual(result.size, (42, 26))

    def test_controls_modifiers_and_disabled_actions(self):
        app = self.app
        self.press("Return")
        app.input_preview.focus_force()
        with patch.object(app, "annotate_selected") as annotate:
            self.press(f"{self.primary}-a")
            self.press("Control-Alt-a")
            annotate.assert_not_called()
        app.task.set("Translate text")
        with patch("enhancer.gui.AnnotationEditor") as dialog:
            self.press("a")
            dialog.assert_not_called()
            self.assertTrue(app.preview_annotate_button.instate(["disabled"]))
            app.task.set("Enhance artwork")
            app._set_busy(True)
            app.tabs.focus_force()
            self.press("a")
            self.press("Right")
            self.press("Alt-Right")
            dialog.assert_not_called()
            self.assertEqual(app.tabs.select(), str(app.preview_tab))
            self.assertEqual(app._preview_path, self.paths[0])
            app._set_busy(False)
            self.press("Escape")
            app.batch_tree.selection_set("0", "1")
            self.press("a")
            dialog.assert_not_called()
        app._focus_file_search()
        entry = app.file_search
        entry.insert(0, "back")
        entry.icursor(2)
        self.press("Shift-Left")
        self.assertTrue(entry.selection_present())
        self.assertEqual(entry.index("insert"), 1)
        with patch.object(app, "add_files") as files, patch.object(app, "add_folder") as folder:
            self.press("Control-Alt-o")
            files.assert_not_called()
            folder.assert_not_called()
            self.press(f"{self.primary}-o")
            self.press(f"{self.primary}-Shift-O")
            files.assert_called_once_with()
            folder.assert_called_once_with()

    def test_help_and_editor_keys_stay_in_the_active_dialog(self):
        app = self.app
        self.press("F1")
        help_window = app.grab_current()
        self.assertEqual(help_window.title(), "Keyboard shortcuts")
        self.press("a")
        self.assertIs(app.grab_current(), help_window)
        self.press("Escape")
        self.assertFalse(help_window.winfo_exists())
        self.assertIs(app.focus_get(), app.batch_tree)
        wording = WordingEditor(app, "", "")
        app.update()
        wording.replacements.insert("1.0", "Play => Spielen")
        self.press(f"{self.primary}-Return", wording.replacements)
        self.assertEqual(wording.result, ("Play => Spielen", ""))
        area = RegionEditor(app, Image.new("RGB", (32, 32)), regions=((1, 2, 10, 12),))
        area.canvas.focus_force()
        app.update()
        self.press(f"{self.primary}-z")
        self.press(f"{self.primary}-Return")
        self.assertEqual(area.result, ())
        self.assertEqual(app.tabs.select(), str(app.batch_tab))

    def test_preview_click_reclaims_keys_from_editable_settings(self):
        app = self.app
        self.press("Return")
        entry = next(widget for widget, _ in app._editable if isinstance(widget, ttk.Entry)
                     and str(widget.cget("textvariable")) == str(app.instruction))
        app.method.set("OpenAI enhancement")
        app.update()
        entry.focus_force()
        self.press("a")
        self.press("Left")
        self.assertEqual(entry.get(), "a")
        self.assertEqual(entry.index("insert"), 0)
        self.assertEqual(app._preview_path, self.paths[0])
        app.output_preview.event_generate("<Button-1>", x=10, y=10)
        app.update()
        self.assertIs(app.focus_get(), app.output_preview)
        self.press("Right")
        self.assertEqual(app._preview_path, self.paths[1])
