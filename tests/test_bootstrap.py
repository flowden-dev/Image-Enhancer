from __future__ import annotations

from pathlib import Path
import io
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import bootstrap


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        output = patch("sys.stdout", new_callable=io.StringIO)
        output.start()
        self.addCleanup(output.stop)
        self.temp = tempfile.TemporaryDirectory(prefix="bootstrap with spaces ")
        self.root = Path(self.temp.name)
        (self.root / "requirements.txt").write_text("Pillow==12.3.0\nhttpx==0.28.1\nnumpy==2.3.5\ntkinterdnd2==0.7.0\n")
        self.root_patcher = patch.object(bootstrap, "ROOT", self.root)
        self.root_patcher.start()
        self.addCleanup(self.root_patcher.stop)
        self.launch = self.enterContext(patch.object(bootstrap, "launch_app", return_value=0))
        env = self.root / ".venv" / "bin"
        env.mkdir(parents=True)
        (env / "python").touch()
        win_env = self.root / ".venv" / "Scripts"
        win_env.mkdir()
        (win_env / "python.exe").touch()

    def tearDown(self):
        self.temp.cleanup()

    def test_ready_environment_skips_pip(self):
        with patch("sys.argv", ["bootstrap.py"]), patch("bootstrap.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(bootstrap.main(), 0)
        run.assert_called_once()
        self.launch.assert_called_once_with(Path(run.call_args.args[0][0]))
        self.assertNotIn("pip", run.call_args.args[0])

    def test_test_option_runs_unittest(self):
        with patch("sys.argv", ["bootstrap.py", "--test"]), patch("bootstrap.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(bootstrap.main(), 0)
        self.assertIn("unittest", run.call_args_list[-1].args[0])
        self.launch.assert_not_called()

    def test_dependency_installation_path(self):
        responses = [subprocess.CompletedProcess([], 1)] + [subprocess.CompletedProcess([], 0)] * 2
        with patch("sys.argv", ["bootstrap.py"]), patch("bootstrap.subprocess.run", side_effect=responses) as run:
            self.assertEqual(bootstrap.main(), 0)
        self.assertIn("pip", run.call_args_list[1].args[0])
        self.assertIn("--require-virtualenv", run.call_args_list[1].args[0])

    def test_dependency_check_reads_current_requirement_pins(self):
        (self.root / "requirements.txt").write_text("Pillow==99.0.0\n")
        with patch("sys.argv", ["bootstrap.py"]), patch("bootstrap.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(bootstrap.main(), 0)
        self.assertIn("99.0.0", run.call_args_list[0].args[0][-1])

    def test_invalid_requirement_file_fails_without_installing(self):
        (self.root / "requirements.txt").write_text("Pillow>=12\n")
        with patch("sys.argv", ["bootstrap.py"]), patch("bootstrap.subprocess.run") as run:
            self.assertEqual(bootstrap.main(), 1)
        run.assert_not_called()

    def test_install_failure_reported(self):
        responses = [subprocess.CompletedProcess([], 1), subprocess.CalledProcessError(1, ["pip"])]
        with patch("sys.argv", ["bootstrap.py"]), patch("bootstrap.subprocess.run", side_effect=responses):
            self.assertEqual(bootstrap.main(), 1)

    def test_unsupported_python(self):
        with patch("sys.argv", ["bootstrap.py"]), patch("sys.version_info", (3, 10)):
            self.assertEqual(bootstrap.main(), 1)

    def test_future_python_without_pinned_wheels_is_rejected(self):
        with patch("sys.argv", ["bootstrap.py"]), patch("sys.version_info", (3, 15)):
            self.assertEqual(bootstrap.main(), 1)

    def test_unknown_option(self):
        with patch("sys.argv", ["bootstrap.py", "--wrong"]):
            self.assertEqual(bootstrap.main(), 1)

    def test_launch_failure_propagates(self):
        self.launch.return_value = 7
        with patch("sys.argv", ["bootstrap.py"]), patch("bootstrap.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
            self.assertEqual(bootstrap.main(), 7)


if __name__ == "__main__":
    unittest.main()
