"""OS-specific defaults and launcher configuration without external processes."""
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from enhancer.api import default_recovery_directory, private_recovery_directory
from enhancer.storage import default_data_directory


class PlatformTests(unittest.TestCase):
    def test_output_directories_are_platform_native(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            for platform, env, expected in (
                ("win32", {"LOCALAPPDATA": str(home / "Local")}, home / "Local" / "ImageEnhancer"),
                ("win32", {"LOCALAPPDATA": "relative"}, home / "AppData" / "Local" / "ImageEnhancer"),
                ("darwin", {}, home / "Library" / "Application Support" / "ImageEnhancer"),
                ("linux", {"XDG_DATA_HOME": str(home / "Data")}, home / "Data" / "image-enhancer"),
                ("linux", {"XDG_DATA_HOME": "relative"}, home / ".local" / "share" / "image-enhancer"),
            ):
                with self.subTest(platform=platform, env=env), patch("enhancer.storage.sys.platform", platform), \
                     patch.dict(os.environ, env, clear=True), patch("enhancer.storage.Path.home", return_value=home):
                    self.assertEqual(default_recovery_directory(), expected / "outputs")
                    self.assertEqual(private_recovery_directory(), expected / "recovery")

    def test_absolute_data_directory_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = {"IMAGE_ENHANCER_DATA_DIR": str(root), "XDG_DATA_HOME": str(root / "xdg")}
            with patch("enhancer.storage.sys.platform", "linux"), patch.dict(os.environ, env, clear=True):
                self.assertEqual(default_data_directory(), root)
                with patch.dict(os.environ, {"IMAGE_ENHANCER_DATA_DIR": "relative"}):
                    self.assertEqual(default_data_directory(), root / "xdg" / "image-enhancer")

    def test_launchers_and_distribution_entry_points_exist(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("run.cmd", "test.cmd", "run.sh", "Run.command", "enhancer/__main__.py"):
            self.assertTrue((root/name).is_file(), name)
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["name"], "image-enhancer")
        self.assertEqual(project["gui-scripts"], {"image-enhancer": "enhancer.gui:main"})
