"""Startup, deliberate repeat runs and files-only output; never contact the API."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from enhancer.api import ApiImage, private_recovery_directory, run_job
from enhancer.batch import prepare_batch, run_batch
from enhancer.core import EnhancerError, Settings, load_texture
from enhancer.gui import App
from enhancer.storage import check_free_space, output_paths
import test_gui as gui_tests


def widgets(parent):
    for widget in parent.winfo_children():
        yield widget
        yield from widgets(widget)


def generated(key, upload, plan, settings, alpha, **kwargs):
    return ApiImage(Image.new("RGBA", plan.canvas_size, (30, 80, 160, 255)), "offline", {})


class OutputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        state = patch.dict(os.environ, {"IMAGE_ENHANCER_DATA_DIR": str(self.root / "state")})
        state.start()
        self.addCleanup(state.stop)
        self.path = self.root / "icon.PNG"
        Image.new("RGBA", (32, 16), (50, 90, 140, 160)).save(self.path)
        self.texture = load_texture(self.path)
        self.addCleanup(self.texture.image.close)

    def test_local_and_paid_outputs_keep_only_images_and_clean_private_work(self):
        for method in ("nearest", "openai"):
            with self.subTest(method=method), patch("enhancer.api.request_image", side_effect=generated) as api:
                folder = self.root / method
                result = run_job(self.texture, Settings(method=method), "sk-offline", folder)
                self.assertEqual(list(folder.iterdir()), [result.original_path])
                self.assertEqual(result.original_path.name, "icon.PNG")
                self.assertEqual(list(private_recovery_directory().iterdir()), [])
                self.assertEqual(api.call_count, int(method == "openai"))
                self.assertIsNone(result.metadata_path)
                result.image.close()

    def test_same_input_output_folder_preserves_source_and_existing_results(self):
        before = self.path.read_bytes()
        first = run_job(self.texture, Settings(method="nearest"), "", self.root)
        first_bytes = first.original_path.read_bytes()
        second = run_job(self.texture, Settings(method="nearest"), "", self.root)
        self.assertEqual(first.original_path.name, "icon (2).PNG")
        self.assertEqual(second.original_path.name, "icon (3).PNG")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(first.original_path.read_bytes(), first_bytes)
        self.assertEqual(second.original_path.read_bytes(), first_bytes)
        first.image.close()
        second.image.close()

    def test_collision_names_are_portable_across_case_and_unicode_filesystems(self):
        out = self.root / "out"
        out.mkdir()
        (out / "ICON.png").write_bytes(b"existing")
        (out / "Café.png").mkdir()
        actual = output_paths(out, ("icon.PNG", "icon.PNG", "Cafe\u0301.png", "new.png", "new.png"))
        self.assertEqual([p.name for p in actual],
                         ["icon (2).PNG", "icon (3).PNG", "Cafe\u0301 (2).png", "new.png", "new (2).png"])
        self.assertEqual((out / "ICON.png").read_bytes(), b"existing")
        self.assertEqual(len(list(out.iterdir())), 2)

    def test_unsafe_filename_is_rejected_before_creating_any_file(self):
        for name in ("", ".", "..", "../escape.png", "folder/image.png", "folder\\image.png"):
            with self.subTest(name=name), self.assertRaises(EnhancerError):
                output_paths(self.root, (name,))

    def test_failed_paid_export_keeps_private_recovery_and_no_output_junk(self):
        with patch("enhancer.api.request_image", side_effect=generated) as api, \
                patch("enhancer.api.save_original_atomic", side_effect=OSError("blocked")):
            result = run_job(self.texture, Settings(), "sk-offline", self.root / "out")
        api.assert_called_once()
        self.assertEqual(list((self.root / "out").iterdir()), [])
        for path in (result.raw_path, result.png_path, result.metadata_path):
            self.assertTrue(path.is_relative_to(private_recovery_directory()))
            self.assertTrue(path.is_file())
        result.image.close()

    def test_space_budget_checks_shared_and_separate_drives(self):
        output = SimpleNamespace(stat=lambda: SimpleNamespace(st_dev=1))
        recovery = SimpleNamespace(stat=lambda: SimpleNamespace(st_dev=1))
        with patch("enhancer.storage.shutil.disk_usage", return_value=SimpleNamespace(free=90)):
            with self.assertRaises(EnhancerError):
                check_free_space(output, 60, recovery, 40)
        recovery.stat = lambda: SimpleNamespace(st_dev=2)
        with patch("enhancer.storage.shutil.disk_usage", side_effect=[SimpleNamespace(free=60), SimpleNamespace(free=40)]):
            check_free_space(output, 60, recovery, 40)
        with patch("enhancer.storage.shutil.disk_usage", side_effect=[SimpleNamespace(free=60), SimpleNamespace(free=39)]):
            with self.assertRaises(EnhancerError):
                check_free_space(output, 60, recovery, 40)

    def test_retained_intermediates_are_budgeted_for_the_whole_queue(self):
        second = self.root / "second.png"
        second.write_bytes(self.path.read_bytes())
        settings = Settings(keep_intermediates=True, reuse_identical=False)
        stop = threading.Event()
        items = prepare_batch((self.path, second), settings, stop)
        with patch("enhancer.batch.check_free_space", side_effect=EnhancerError("budget")) as space, \
                patch("enhancer.api.request_image") as api:
            for config in (replace(settings, keep_intermediates=False), settings):
                with self.assertRaisesRegex(EnhancerError, "budget"):
                    run_batch(items, config, "sk-offline", self.root / "out", stop)
        peak, retained = [call.args[3] for call in space.call_args_list]
        self.assertGreater(retained, peak)
        api.assert_not_called()


class RepeatGuiTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)
    setUp = gui_tests.GuiTests.setUp
    tearDown = gui_tests.GuiTests.tearDown
    wait = gui_tests.GuiTests.wait
    result_for = gui_tests.GuiTests.result_for

    def queue(self, *paths):
        self.app._runner = run_job
        self.app.add_paths(paths or (self.source,))
        self.wait()

    def local_run(self):
        self.queue()
        self.app.method.set("Faithful resize (free)")
        self.app.run_button.invoke()
        self.wait()
        self.assertEqual(self.app.batch_tree.set("0", "state"), "Done")

    def test_run_again_uses_same_queue_and_keeps_both_results(self):
        self.local_run()
        self.dialogs["askyesno"].assert_not_called()
        first = self.app.result.original_path
        contents = first.read_bytes()
        self.assertEqual(self.app.run_button["text"], "Run Again")
        self.assertEqual(str(self.app.rerun_button["state"]), "normal")
        self.app.run_button.invoke()
        self.wait()
        self.dialogs["askyesno"].assert_called_once()
        self.assertEqual(self.dialogs["askyesno"].call_args.args[0], "Run Queue Again")
        self.assertIn("no API requests", self.dialogs["askyesno"].call_args.args[1])
        self.assertEqual(self.app.result.original_path.name, "texture (2).png")
        self.assertEqual(first.read_bytes(), contents)
        self.assertEqual(self.app._batch_paths, [self.source])
        self.assertEqual(self.app._batch_done, {0})
        self.assertEqual(self.app.batch_count.get(), "1 / 1 files completed")

    def test_declining_rerun_restores_completed_rows_details_and_output(self):
        self.local_run()
        before = self.app.batch_tree.item("0", "values")
        detail = self.app._batch_details[0]
        result = self.app.result
        self.dialogs["askyesno"].return_value = False
        with patch("enhancer.api.request_image") as api:
            self.app.rerun_button.invoke()
            self.wait()
        api.assert_not_called()
        self.assertEqual(self.app.batch_tree.item("0", "values"), before)
        self.assertEqual(self.app._batch_details[0], detail)
        self.assertIs(self.app.result, result)
        self.assertEqual(self.app._batch_outputs[0], result.original_path)
        self.assertEqual(list(self.app.recovery_directory.iterdir()), [result.original_path])

    def test_paid_rerun_is_confirmed_once_and_duplicate_reuse_does_not_cross_runs(self):
        other = self.root / "duplicate.png"
        other.write_bytes(self.source.read_bytes())
        self.queue(self.source, other)
        with patch("enhancer.api.request_image", side_effect=generated) as api:
            self.app.run_button.invoke()
            self.wait()
            self.assertEqual(api.call_count, 1)
            self.app.rerun_button.invoke()
            self.wait()
            self.assertEqual(api.call_count, 2)
            self.assertTrue(all(call.args[3].quality == "max" for call in api.call_args_list))
        self.assertEqual(self.dialogs["askyesno"].call_count, 2)
        self.assertIn("new charges", self.dialogs["askyesno"].call_args.args[1])
        self.assertIn("1 paid image edit?", self.dialogs["askyesno"].call_args.args[1])
        self.assertEqual(len(list(self.app.recovery_directory.iterdir())), 4)
        self.assertEqual(self.app._batch_done, {0, 1})

    def test_invalid_key_or_missing_source_during_rerun_keeps_previous_result(self):
        self.local_run()
        previous = self.app.result
        self.app.method.set("OpenAI enhancement")
        self.app.key.set("")
        with patch("enhancer.api.request_image") as api:
            self.app.rerun_button.invoke()
            self.wait()
            self.assertEqual(self.app.batch_tree.set("0", "state"), "Done")
            self.source.unlink()
            self.app.rerun_button.invoke()
            self.wait()
        api.assert_not_called()
        self.assertIs(self.app.result, previous)
        self.assertEqual(self.app.batch_tree.set("0", "state"), "Done")
        self.dialogs["askyesno"].assert_not_called()
        self.assertEqual(self.dialogs["showwarning"].call_count, 2)

    def test_rerun_worker_start_failure_does_not_clear_completed_rows(self):
        self.local_run()
        with patch("enhancer.gui.threading.Thread.start", side_effect=RuntimeError("blocked")):
            self.app.rerun_button.invoke()
        self.assertEqual(self.app.batch_tree.set("0", "state"), "Done")
        self.assertFalse(self.app._busy)
        self.dialogs["askyesno"].assert_not_called()

    def test_second_worker_failure_after_confirmation_restores_completed_rows(self):
        self.local_run()
        start = threading.Thread.start
        calls = []
        def start_once(thread):
            calls.append(thread)
            if len(calls) == 1:
                return start(thread)
            raise RuntimeError("blocked")
        with patch("enhancer.gui.threading.Thread.start", new=start_once), patch("enhancer.api.request_image") as api:
            self.app.rerun_button.invoke()
            self.wait()
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.app.batch_tree.set("0", "state"), "Done")
        self.assertEqual(self.app._batch_done, {0})
        self.dialogs["askyesno"].assert_called_once()
        api.assert_not_called()

    def test_single_export_failure_reports_private_recovery_and_keeps_output_target(self):
        self.app.load_file(self.source)
        self.app.method.set("Faithful resize (free)")
        self.app._runner = run_job
        with patch("enhancer.api.save_original_atomic", side_effect=OSError("blocked")):
            self.app.process()
            self.wait()
        self.assertIn("Recovery PNG saved", self.app.status.get())
        self.assertEqual(self.app._last_output_directory, self.app.recovery_directory)
        self.assertIn(str(self.app.result.png_path), self.dialogs["showwarning"].call_args.args[1])

    def test_remove_and_clear_are_right_above_table_at_minimum_size(self):
        self.app.deiconify()
        self.app.geometry("1000x640")
        self.app.tabs.select(self.app.batch_tab)
        self.app.update()
        tree = self.app.batch_tree
        remove, clear = self.app.remove_button, self.app.clear_button
        self.assertTrue(remove.winfo_viewable())
        self.assertGreater(clear.winfo_rootx(), remove.winfo_rootx())
        self.assertEqual(remove.winfo_rooty(), clear.winfo_rooty())
        self.assertLessEqual(clear.winfo_rooty() + clear.winfo_height(), tree.winfo_rooty())
        self.assertLess(tree.winfo_rooty() - clear.winfo_rooty() - clear.winfo_height(), 15)
        self.assertLessEqual(abs(clear.winfo_rootx() + clear.winfo_width() -
                                 tree.winfo_rootx() - tree.winfo_width()), 25)

    def test_open_folder_buttons_target_selected_folder(self):
        folder = self.root / "selected output"
        with patch("enhancer.gui.filedialog.askdirectory", return_value=str(folder)):
            self.app.choose_output()
        buttons = [w for w in widgets(self.app) if isinstance(w, ttk.Button) and
                   w["text"] in ("Open Folder", "Open Output Folder")]
        self.assertEqual(len(buttons), 2)
        with patch.object(self.app, "_open_directory") as open_folder:
            for button in buttons:
                button.invoke()
        self.assertEqual([call.args[0] for call in open_folder.call_args_list], [folder, folder])


class StartupGuiTests(unittest.TestCase):
    setUpClass = classmethod(gui_tests.GuiTests.setUpClass.__func__)

    def test_startup_always_opens_key_dialog_once_including_environment_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-offline"}), \
                patch.object(App, "api_settings") as setup:
            app = App()
            try:
                visible = []
                setup.side_effect = lambda **kwargs: visible.append(app.winfo_viewable())
                deadline = time.monotonic() + 3
                while not setup.called and time.monotonic() < deadline:
                    app.update()
                    time.sleep(0.01)
                setup.assert_called_once_with(startup=True)
                self.assertEqual(visible, [True])
                self.assertEqual(app.method.get(), "OpenAI enhancement")
                self.assertEqual(app.quality.get(), "max")
            finally:
                app.destroy()

    def exercise_dialog(self, action):
        app = App(prompt_for_key=False)
        errors = []
        def interact():
            dialog = next(w for w in app.winfo_children() if isinstance(w, tk.Toplevel))
            try:
                action(app, dialog, list(widgets(dialog)))
            except Exception as exc:
                errors.append(exc)
            finally:
                if dialog.winfo_exists():
                    dialog.destroy()
        try:
            app.after(100, interact)
            app.api_settings(startup=True)
            if errors:
                raise errors[0]
        finally:
            app.destroy()

    def test_startup_masks_key_defaults_max_and_continues_without_network(self):
        def action(app, dialog, controls):
            entry = next(w for w in controls if isinstance(w, ttk.Entry))
            quality = next(w for w in controls if isinstance(w, ttk.Combobox))
            self.assertEqual(entry["show"], "*")
            self.assertEqual(quality.get(), "max")
            entry.delete(0, "end")
            entry.insert(0, "sk-offline-startup")
            next(w for w in controls if isinstance(w, ttk.Button) and w["text"] == "Continue").invoke()
            self.assertEqual(app.key.get(), "sk-offline-startup")
            self.assertEqual(app.method.get(), "OpenAI enhancement")
        with patch("enhancer.api.request_image") as api:
            self.exercise_dialog(action)
        api.assert_not_called()

    def test_empty_startup_key_requires_explicit_local_mode(self):
        def action(app, dialog, controls):
            entry = next(w for w in controls if isinstance(w, ttk.Entry))
            entry.delete(0, "end")
            buttons = {w["text"]: w for w in controls if isinstance(w, ttk.Button)}
            with patch("enhancer.gui.messagebox.showwarning") as warning:
                buttons["Continue"].invoke()
            warning.assert_called_once()
            self.assertTrue(dialog.winfo_exists())
            buttons["Use Local Mode"].invoke()
            self.assertEqual(app._settings().method, "neural")
        self.exercise_dialog(action)
