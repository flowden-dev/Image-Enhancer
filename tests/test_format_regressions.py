import io
from pathlib import Path
import struct
import tempfile
import unittest

from PIL import Image

from enhancer.api import run_job
from enhancer.core import EnhancerError, Settings, load_texture
from enhancer.formats import decode_source, preflight, encode_original
from format_fixtures import tga_fixture, png_chunk


class FormatRegressions(unittest.TestCase):
    def data_tga(self, alpha_type=2, descriptor_bits=8):
        raw = bytearray(tga_fixture((2, 1), extension=True))
        raw[17] = (raw[17] & ~15) | descriptor_bits
        start = 18 + raw[0]
        raw[start:start + 8] = bytes((0, 0, 200, 0, 0, 100, 0, 255))
        offset = struct.unpack_from('<I', raw, len(raw) - 26)[0]
        raw[offset + 494] = alpha_type
        return bytes(raw)

    def test_tga_undefined_attributes_do_not_blacken_rgb(self):
        for kind in (1, 2):
            for descriptor in (0, 8):
                with self.subTest(kind=kind, descriptor=descriptor), tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / 'data.tga'
                    source.write_bytes(self.data_tga(kind, descriptor))
                    texture = load_texture(source)
                    self.assertTrue(texture.profile.data_texture)
                    result = run_job(texture, Settings(method='lanczos'), '', Path(directory) / 'out')
                    self.assertGreater(result.image.getpixel((0, 0))[0], 150)
                    decoded, profile = decode_source(result.original_path.read_bytes(), '.tga')
                    self.assertGreater(decoded.getpixel((0, 0))[0], 150)
                    self.assertEqual(profile.details.extension[494], kind)
                    self.assertEqual(profile.details.header[17] & 15, descriptor)
                    self.assertLess(decoded.getpixel((0, 0))[3], 128)

    def test_tga_data_prevents_api_preflight(self):
        image, profile = decode_source(self.data_tga(), '.tga')
        with self.assertRaisesRegex(EnhancerError, 'data textures'):
            preflight(profile, image)
        preflight(profile, image, method='nearest')

    def test_contradictory_tga_alpha_is_rejected(self):
        with self.assertRaisesRegex(EnhancerError, 'no alpha'):
            decode_source(self.data_tga(0), '.tga')

    def test_indexed_tga_no_alpha_extension_ignores_palette_attribute(self):
        raw = bytearray(tga_fixture((4, 1), bits=8, indexed=True, extension=True))
        offset = struct.unpack_from('<I', raw, len(raw) - 26)[0]
        raw[offset + 494] = 0
        image, profile = decode_source(bytes(raw), '.tga')
        self.assertEqual(image.getchannel('A').getextrema(), (255, 255))
        saved = encode_original(image.resize((8, 2)), profile)
        again, output = decode_source(saved, '.tga')
        self.assertEqual(again.getchannel('A').getextrema(), (255, 255))
        self.assertEqual(output.details.palette_bytes, profile.details.palette_bytes)

    def test_custom_png_gamma_local_only_and_preserved(self):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), (80, 90, 100)).save(buffer, format='PNG')
        raw = buffer.getvalue()
        raw = raw[:33] + png_chunk(b'gAMA', struct.pack('>I', 100000)) + raw[33:]
        image, profile = decode_source(raw, '.png')
        with self.assertRaisesRegex(EnhancerError, 'calibration'):
            preflight(profile, image)
        preflight(profile, image, method='nearest')
        saved = encode_original(image.resize((16, 16)), profile)
        self.assertIn(png_chunk(b'gAMA', struct.pack('>I', 100000)), saved)

    def test_standard_png_gamma_can_use_api(self):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), (80, 90, 100)).save(buffer, format='PNG')
        raw = buffer.getvalue()
        raw = raw[:33] + png_chunk(b'gAMA', struct.pack('>I', 45455)) + raw[33:]
        image, profile = decode_source(raw, '.png')
        preflight(profile, image)

    def test_explicit_tga_gamma_local_only_and_preserved(self):
        raw = bytearray(tga_fixture((4, 4), extension=True))
        offset = struct.unpack_from('<I', raw, len(raw) - 26)[0]
        struct.pack_into('<HH', raw, offset + 478, 22, 10)
        image, profile = decode_source(bytes(raw), '.tga')
        with self.assertRaisesRegex(EnhancerError, 'gamma'):
            preflight(profile, image)
        preflight(profile, image, method='nearest')
        saved = encode_original(image.resize((8, 8)), profile)
        _, output = decode_source(saved, '.tga')
        self.assertEqual(output.details.extension[478:482], profile.details.extension[478:482])

    def test_hdr_png_metadata_is_not_dropped_silently(self):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8)).save(buffer, format='PNG')
        raw = buffer.getvalue()
        raw = raw[:33] + png_chunk(b'cICP', bytes((9, 16, 0, 1))) + raw[33:]
        with self.assertRaisesRegex(EnhancerError, 'HDR'):
            decode_source(raw, '.png')
