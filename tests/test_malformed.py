"""Seeded, bounded malformed-input and container-metadata regressions."""
from __future__ import annotations

import io
from pathlib import Path
import random
import struct
import tempfile
import unittest

from PIL import Image

from enhancer.core import EnhancerError, load_texture
from enhancer.formats import png
from format_fixtures import dds_fixture, tga_fixture, png_fixture, png_chunk


class MalformedInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def decode(self, data, extension):
        path = self.root / ("input" + extension)
        path.write_bytes(data)
        try:
            return load_texture(path)
        except EnhancerError:
            return None

    def test_seeded_bounded_mutations_fail_cleanly(self):
        # mutation may legitimately leave a valid image; all other failures must be controlled
        stream = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(stream, format="JPEG")
        fixtures = [(dds_fixture(size=(8, 8)), ".dds"), (tga_fixture(size=(8, 8)), ".tga"),
                    (png_fixture((8, 8)), ".png"), (stream.getvalue(), ".jpg")]
        rng = random.Random(20261109)
        for original, extension in fixtures:
            for index in range(250):
                data = bytearray(original)
                if index % 3 == 0:
                    del data[rng.randrange(len(data)):]
                else:
                    for _ in range(1 + index % 4):
                        data[rng.randrange(len(data))] ^= rng.randrange(1, 256)
                with self.subTest(extension=extension, case=index):
                    self.decode(data, extension)

    def test_png_duplicate_critical_and_metadata_chunks_rejected(self):
        original = png_fixture()
        ihdr = original[8:33]
        invalid = [original[:33] + ihdr + original[33:]]
        gamma = png_chunk(b"gAMA", struct.pack(">I", 45455))
        invalid.append(original[:33] + gamma + gamma + original[33:])
        for data in invalid:
            with self.subTest(data=data[:60]), self.assertRaises(EnhancerError):
                png.inspect(data)

    def test_png_16bit_rejected_without_silent_precision_loss(self):
        stream = io.BytesIO()
        Image.new("I;16", (8, 8), 12345).save(stream, format="PNG")
        path = self.root / "sixteen.png"
        path.write_bytes(stream.getvalue())
        with self.assertRaisesRegex(EnhancerError, "16-bit"):
            load_texture(path)

    def test_random_short_non_image_inputs(self):
        rng = random.Random(1234)
        for extension in (".dds", ".tga", ".png", ".jpg"):
            for size in (0, 1, 2, 4, 16, 17, 31, 64, 127, 148, 512):
                with self.subTest(extension=extension, size=size):
                    self.assertIsNone(self.decode(rng.randbytes(size), extension))


if __name__ == "__main__":
    unittest.main()
