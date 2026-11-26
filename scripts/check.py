"""Run the full offline suite; CI rejects skipped GUI checks."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
suite = unittest.defaultTestLoader.discover(str(root / "tests"))
with tempfile.TemporaryDirectory(prefix="image-enhancer-tests-") as directory:
    with patch.dict(os.environ, {"IMAGE_ENHANCER_DATA_DIR": directory}):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
if "--require-gui" in sys.argv and result.skipped:
    print("Full validation requires a working Tk desktop; skipped checks are not a pass.")
    raise SystemExit(1)
raise SystemExit(not result.wasSuccessful())
