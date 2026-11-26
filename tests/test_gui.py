from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from enhancer import __version__
from enhancer.api import Result, run_job
from enhancer.core import EnhancerError, Settings, plan_image
from enhancer.gui import App
from test_api import response_body


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            root = tk.Tk()
            root.withdraw()
            root.destroy()
        except tk.TclError as exc:
            raise unittest.SkipTest("GUI tests require a desktop display or Xvfb") from exc

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gui test ")
        self.root = Path(self.temp.name)
        self.source = self.root / "texture.png"
        Image.new("RGBA", (64, 32), (10, 80, 150, 128)).save(self.source)
        self.dialogs = {}
        for function in ("showwarning", "showerror", "askyesno"):
            patcher = patch("enhancer.gui.messagebox." + function, return_value=True)
            self.dialogs[function] = patcher.start()
            self.addCleanup(patcher.stop)
        self.calls = []
        def runner(texture, settings, key, directory):
            self.calls.append((texture, settings, key, threading.get_ident()))
            return self.result_for(texture, settings)
        self.app = App(runner=runner, recovery_directory=self.root / "outputs", prompt_for_key=False)
        self.app.withdraw()
        # use a small output for offline workflow tests
        self.app.method.set("OpenAI enhancement")
        self.app.output_size.set("2× original")
        self.app.key.set("sk-test-gui-secret")

    def tearDown(self):
        if not self.app._closed:
            self.app.destroy()
        self.temp.cleanup()

    def result_for(self, texture, settings):
        plan = plan_image(texture.image.size, settings.scale)
        image = texture.image.resize(plan.target_size)
        return Result(image, texture.path, settings, plan, None, None, "req_gui", {}, ())

    def wait(self):
        deadline = time.monotonic() + 5
        while self.app._busy and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        self.assertFalse(self.app._busy, "UI did not receive worker completion")

    def test_launch_and_load(self):
        self.assertEqual(self.app.title(), f"Image Enhancer {__version__}")
        self.assertTrue(self.app.load_file(self.source))
        self.assertIn("128 x 64", self.app.dimensions.get())
        self.assertEqual(str(self.app.save_button["state"]), "disabled")

    def test_no_texture_no_request(self):
        self.app.process()
        self.dialogs["showwarning"].assert_called_once()
        self.assertEqual(self.calls, [])

    def test_no_key_no_request(self):
        self.app.load_file(self.source)
        self.app.key.set("")
        self.app.process()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.app._busy)

    def test_denied_cost_confirmation_no_request(self):
        self.app.load_file(self.source)
        self.dialogs["askyesno"].return_value = False
        self.app.process()
        self.assertEqual(self.calls, [])

    def test_success_and_control_locking(self):
        self.app.load_file(self.source)
        self.app.process()
        self.assertTrue(self.app._busy)
        self.assertEqual(str(self.app.open_button["state"]), "disabled")
        self.assertFalse(self.app.load_file(self.source))
        self.app.process()
        self.wait()
        self.assertEqual(len(self.calls), 1)
        self.assertNotEqual(self.calls[0][3], threading.get_ident())
        self.assertEqual(self.app.result.image.size, (128, 64))
        self.assertEqual(str(self.app.save_button["state"]), "normal")
        self.assertEqual(str(self.app.open_button["state"]), "normal")
        for widget, state in self.app._editable:
            self.assertEqual(str(widget["state"]), state)

    def test_snapshot_independent_of_later_control_values(self):
        self.app.load_file(self.source)
        self.app.process()
        self.wait()
        self.app.output_size.set("4× original")
        destination = self.root / "saved.png"
        with patch("enhancer.gui.filedialog.asksaveasfilename", return_value=str(destination)) as dialog:
            self.app.save_result()
        self.wait()
        self.assertEqual(dialog.call_args.kwargs["initialfile"], self.source.name)
        with Image.open(destination) as image:
            self.assertEqual(image.size, (128, 64))

    def test_failure_retains_previous_result(self):
        self.app.load_file(self.source)
        self.app.process()
        self.wait()
        previous = self.app.result
        def failure(*_):
            raise EnhancerError("simulated failure sk-test-gui-secret")
        self.app._runner = failure
        self.app.process()
        self.wait()
        self.assertIs(self.app.result, previous)
        self.assertEqual(str(self.app.save_button["state"]), "normal")
        self.assertNotIn("sk-test-gui-secret", str(self.dialogs["showerror"].call_args))

    def test_load_failure_preserves_current_source(self):
        self.app.load_file(self.source)
        previous = self.app.texture
        self.assertFalse(self.app.load_file(self.root / "absent.png"))
        self.assertIs(self.app.texture, previous)

    def test_switching_source_clears_old_result(self):
        self.app.load_file(self.source)
        self.app.process()
        self.wait()
        self.app.load_file(self.source)
        self.assertIsNone(self.app.result)
        self.assertEqual(str(self.app.save_button["state"]), "disabled")

    def test_save_original_refused(self):
        self.app.load_file(self.source)
        self.app.process()
        self.wait()
        original = self.source.read_bytes()
        with patch("enhancer.gui.filedialog.asksaveasfilename", return_value=str(self.source)):
            self.app.save_result()
        self.wait()
        self.assertEqual(self.source.read_bytes(), original)
        self.dialogs["showerror"].assert_called_once()

    def test_invalid_settings_no_request(self):
        self.app.load_file(self.source)
        self.app.output_size.set("invalid")
        self.app.process()
        self.assertEqual(self.calls, [])

    def test_fully_transparent_no_request(self):
        self.app.load_file(self.source)
        self.app.texture.image.putalpha(0)
        self.app.process()
        self.assertEqual(self.calls, [])

    def test_decline_discard_unsaved_result(self):
        self.app.load_file(self.source)
        self.app.result = self.result_for(self.app.texture, Settings())
        self.dialogs["askyesno"].return_value = False
        self.assertFalse(self.app.load_file(self.source))
        self.assertIsNotNone(self.app.result)

    def test_close_active_request_can_be_declined(self):
        self.app._set_busy(True)
        self.dialogs["askyesno"].return_value = False
        self.app.on_close()
        self.assertFalse(self.app._closed)
        self.app._set_busy(False)

    def test_close_clears_key(self):
        self.app.on_close()
        self.assertTrue(self.app._closed)
        self.assertEqual(self.app.key.get(), "")

    def test_worker_start_failure_restores_controls(self):
        self.app.load_file(self.source)
        with patch("enhancer.gui.threading.Thread.start", side_effect=RuntimeError("cannot start")):
            self.app.process()
        self.assertFalse(self.app._busy)
        self.assertEqual(self.calls, [])
        self.assertEqual(str(self.app.open_button["state"]), "normal")

    def test_real_pipeline_through_gui_with_mock_http(self):
        self.app.load_file(self.source)
        def runner(texture, settings, key, directory):
            plan = plan_image(texture.image.size, settings.scale)
            transport = httpx.MockTransport(lambda _: httpx.Response(200, content=response_body(plan.canvas_size)))
            return run_job(texture, settings, key, directory, transport=transport)
        self.app._runner = runner
        self.app.process()
        self.wait()
        self.assertTrue(self.app.result.saved_path.exists())
        self.assertIsNone(self.app.result.raw_path)
        self.assertEqual(self.app.result.image.size, (128, 64))
        self.dialogs["showerror"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
