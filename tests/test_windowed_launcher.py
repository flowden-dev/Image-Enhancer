from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tkinter as tk
import unittest
from unittest.mock import patch

import bootstrap


class WindowedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="launcher with spaces ")))
        shutil.copyfile(bootstrap.ROOT / "app.py", self.root / "app.py")
        package = self.root / "enhancer"
        package.mkdir()
        (package / "__init__.py").touch()
        self.gui = package / "gui.py"
        self.enterContext(patch.object(bootstrap, "ROOT", self.root))
        self.executable = Path(sys.executable)
        if os.name == "nt":
            self.executable = self.executable.with_name("pythonw.exe")

    def test_windows_selects_windowless_interpreter_from_same_environment(self):
        python = self.root / "Scripts" / "python.exe"
        with patch.object(bootstrap, "os") as system, patch.object(bootstrap, "launch_windowed", return_value=17) as run:
            system.name = "nt"
            self.assertEqual(bootstrap.launch_app(python), 17)
            run.assert_called_once_with(python.with_name("pythonw.exe"))

    def test_import_error_reaches_setup_console(self):
        self.gui.write_text("raise RuntimeError('startup error fixture')\n", encoding="utf-8")
        with redirect_stdout(StringIO()) as output:
            self.assertNotEqual(bootstrap.launch_windowed(self.executable), 0)
        self.assertIn("RuntimeError: startup error fixture", output.getvalue())

    def test_launcher_returns_with_main_window_mapped_and_process_alive(self):
        try:
            probe = tk.Tk()
            probe.destroy()
        except tk.TclError as exc:
            self.skipTest(f"GUI test needs a desktop display: {exc}")
        self.gui.write_text(textwrap.dedent("""\
            import ctypes
            import os
            from pathlib import Path
            import tkinter as tk

            def create_app():
                if os.name == "nt" and ctypes.windll.kernel32.GetConsoleWindow():
                    raise RuntimeError("GUI inherited a console")
                root = tk.Tk()
                root.withdraw()
                root.bind("<Map>", lambda event: Path("mapped").touch())
                root.after(150, root.deiconify)
                root.after(15000, root.destroy)
                return root
        """), encoding="utf-8")
        children = []
        popen = subprocess.Popen

        def start(*args, **kwargs):
            child = popen(*args, **kwargs)
            children.append(child)
            self.addCleanup(child.wait, timeout=10)
            self.addCleanup(child.terminate)
            return child

        with patch.object(bootstrap.subprocess, "Popen", side_effect=start):
            self.assertEqual(bootstrap.launch_windowed(self.executable), 0)
        self.assertTrue((self.root / "mapped").is_file())
        self.assertIsNone(children[0].poll())
        self.assertFalse(Path(children[0].args[-1]).exists())
