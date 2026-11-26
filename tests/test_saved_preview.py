"""Exercise queue navigation and manual export using real saved files and Tk controls."""
import unittest
from unittest.mock import patch

from PIL import Image

import test_gui as gui_tests
from enhancer.api import run_job
from enhancer.gui import IMAGE_FILTERS
from format_fixtures import pattern


class SavedPreviewTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)
    tearDown = gui_tests.GuiTests.tearDown
    wait = gui_tests.GuiTests.wait
    result_for = gui_tests.GuiTests.result_for

    def setUp(self):
        gui_tests.GuiTests.setUp(self)
        self.folder = self.root / "cards"
        self.folder.mkdir()
        self.files = [self.folder / f for f in ("a.JPG", "b.png", "c.tga")]
        pattern((27, 15), opaque=True).convert("RGB").save(self.files[0], quality=83)
        for path in self.files[1:]:
            Image.new("RGBA", (22, 16), (200, 45, 150, 64)).save(path)
        self.app._runner = run_job
        self.app.method.set("Pixel art (free)")
        self.app.output_format.set("JPG / JPEG")
        self.app.add_paths((self.folder,))
        self.wait()

    def preview(self):
        self.app.tabs.select(self.app.preview_tab)
        self.app.update()
        self.wait()

    def run_folder(self):
        self.app.run_button.invoke()
        self.wait()
        self.preview()

    def test_input_filter_and_source_navigation_before_running(self):
        self.assertIn("*.JPG", dict(IMAGE_FILTERS)["JPEG images"])
        self.preview()
        self.assertEqual(self.app.preview_position.get(), "1 / 3")
        self.assertEqual(self.app.texture.path, self.files[0])
        self.assertEqual(str(self.app.previous_button["state"]), "disabled")
        self.app.next_button.invoke()
        self.assertEqual(self.app.texture.path, self.files[1])
        self.assertIsNone(self.app.output_preview.source)
        self.assertFalse(self.app.recovery_directory.exists())
        self.assertEqual(self.calls, [])

    def test_arrows_show_actual_jpg_pixels_and_save_copy_without_recompression(self):
        with patch("enhancer.api.request_image") as api:
            self.run_folder()
            self.assertEqual(self.app.preview_position.get(), "3 / 3")
            self.assertEqual(str(self.app.next_button["state"]), "disabled")
            self.app.previous_button.invoke()
            self.wait()
            self.assertEqual(self.app.preview_position.get(), "2 / 3")
            self.assertEqual(self.app.batch_tree.selection(), ("1",))
            saved = self.app.recovery_directory / "b.jpg"
            with Image.open(saved) as file:
                self.assertEqual(self.app.output_preview.source.tobytes(), file.convert("RGBA").tobytes())
            self.assertEqual(self.app.output_preview.source.getchannel("A").getextrema(), (255, 255))
            self.assertIn("Saved JPG", self.app.result_caption.get())
            copy = self.root / "inspection.jpg"
            with patch("enhancer.gui.filedialog.asksaveasfilename", return_value=str(copy)) as dialog:
                self.app.save_button.invoke()
            self.wait()
            self.assertEqual(dialog.call_args.kwargs["initialfile"], "b.jpg")
            self.assertEqual(copy.read_bytes(), saved.read_bytes())
            self.app.output_format.set("DDS")
            with patch("enhancer.gui.filedialog.asksaveasfilename", return_value=str(self.root / "inspection.dds")):
                self.app.save_button.invoke()
            self.wait()
            with Image.open(self.root / "inspection.dds") as file:
                self.assertEqual(file.format, "DDS")
            api.assert_not_called()
        self.dialogs["showerror"].assert_not_called()

    def test_latest_numbered_output_and_missing_source_or_result(self):
        self.run_folder()
        self.app.run_button.invoke()
        self.wait()
        self.assertIn("(2)", self.app._saved_view.path.name)
        self.files[1].unlink()
        self.app.previous_button.invoke()
        self.wait()
        self.assertIsNone(self.app.texture)
        self.assertEqual(self.app._saved_view.path.name, "b (2).jpg")
        self.assertIn("Original unavailable", self.app.status.get())
        self.app._saved_exports[self.files[0]].path.unlink()
        self.app.previous_button.invoke()
        self.wait()
        self.assertEqual(self.app.texture.path, self.files[0])
        self.assertIsNone(self.app.output_preview.source)
        self.assertIn("Saved output unavailable", self.app.status.get())
        self.app.next_button.invoke()
        self.wait()
        self.assertIsNotNone(self.app.output_preview.source)
        self.app.clear_batch()
        self.assertIsNone(self.app.output_preview.source)
        self.assertEqual(str(self.app.next_button["state"]), "disabled")

    def test_direct_selection_from_files_and_unsaved_edit(self):
        self.run_folder()
        self.app.tabs.select(self.app.batch_tab)
        self.app.batch_tree.selection_set("0")
        self.app.preview_selected()
        self.wait()
        self.assertEqual(self.app._saved_view.path.name, "a.jpg")
        working = self.root / "edited.png"
        Image.new("RGBA", self.app.result.plan.target_size, (50, 175, 60, 255)).save(working)
        with patch("enhancer.gui.filedialog.askopenfilename", return_value=str(working)):
            self.app.import_edited_png()
        self.assertIsNone(self.app.result.saved_path)
        self.dialogs["askyesno"].return_value = False
        self.app.next_button.invoke()
        self.assertEqual(self.app.preview_position.get(), "1 / 3")
        self.assertEqual(self.app.texture.path, self.files[0])
