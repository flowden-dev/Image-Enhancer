"""Independent numeric references for channel and fixed-palette conversions."""
import unittest
import numpy as np
from PIL import Image

from enhancer.formats.common import palette_indices, premultiply, unpremultiply
from enhancer.formats.dds import _mip


class ConversionTests(unittest.TestCase):
    def test_srgb_mip_across_strip_boundary_matches_full_frame_reference(self):
        pixels = np.random.default_rng(33).integers(0, 256, (131, 43, 4), dtype=np.uint8)
        pixels[:, :, 3] = np.maximum(pixels[:, :, 3], 32)
        image = Image.fromarray(pixels)
        size = (21, 65)
        colors = pixels[:, :, :3].astype(np.float32) / 255
        alpha = pixels[:, :, 3].astype(np.float32) / 255
        linear = np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4)
        def resize(values):
            return np.asarray(Image.fromarray(values).resize(size, Image.Resampling.LANCZOS))
        a = resize(alpha).clip(0, 1)
        small = np.stack([resize(linear[:, :, c] * alpha) / a for c in range(3)], axis=2).clip(0, 1)
        srgb = np.where(small <= 0.0031308, small * 12.92, 1.055 * small ** (1 / 2.4) - 0.055)
        expected = np.rint(np.dstack((srgb, a)).clip(0, 1) * 255).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(_mip(image, size, True)), expected)

    def test_palette_chunk_boundary_matches_scalar_distance(self):
        palette = np.array([[0, 0, 0, 0], [255, 255, 255, 255], [40, 80, 120, 180],
                            [90, 50, 10, 20]], dtype=np.uint8)
        pixels = np.random.default_rng(78).integers(0, 256, (257, 257, 4), dtype=np.uint8)
        actual = palette_indices(Image.fromarray(pixels), palette)
        delta = pixels.astype(np.int32)[:, :, None, :] - palette.astype(np.int32)
        distance = delta**2
        distance[:, :, :, 3] *= 4
        np.testing.assert_array_equal(actual, distance.sum(axis=3).argmin(axis=2))

    def test_premultiplication_extremes_do_not_overflow(self):
        values = np.array([[[255, 254, 128, a] for a in (0, 1, 2, 127, 128, 254, 255)]], dtype=np.uint8)
        actual = np.array(premultiply(Image.fromarray(values)))
        for index, pixel in enumerate(values[0]):
            for channel in range(3):
                self.assertEqual(int(actual[0, index, channel]), (int(pixel[channel])*int(pixel[3])+127)//255)
        restored = np.array(unpremultiply(Image.fromarray(actual)))
        self.assertEqual(tuple(restored[0, 0]), (0, 0, 0, 0))
        self.assertEqual(tuple(restored[0, -1]), tuple(values[0, -1]))
