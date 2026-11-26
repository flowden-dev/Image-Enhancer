from __future__ import annotations

import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib

import httpx
from PIL import Image

from enhancer import api, core
from test_api import response_body


class AdditionalBoundaryTests(unittest.TestCase):
    def test_16bit_rgb_png_refused_before_silent_conversion(self):
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        png = b'\x89PNG\r\n\x1a\n'
        png += chunk(b'IHDR', struct.pack('>2I5B', 1, 1, 16, 2, 0, 0, 0))
        png += chunk(b'IDAT', zlib.compress(b'\0' + b'\xff\xff' * 3))
        png += chunk(b'IEND', b'')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rgb16.png'
            path.write_bytes(png)
            with self.assertRaisesRegex(core.EnhancerError, '16-bit'):
                core.load_texture(path)

    def test_instruction_invalid_unicode_and_control_characters(self):
        for instruction in ('invalid\x00text', 'invalid\ud800text', 'invalid\rtext'):
            with self.subTest(instruction=repr(instruction)), self.assertRaises(core.EnhancerError):
                core.Settings(instruction=instruction).validate()
        core.Settings(instruction='Translate\nPLAY\tto SPIELEN', source_protection='off').validate()

    def test_excessively_nested_json(self):
        with self.assertRaises(core.EnhancerError):
            api.decode_response(b'[' * 2000 + b']' * 2000)

    def test_invalid_upload_never_sent(self):
        calls = []
        transport = httpx.MockTransport(lambda request: calls.append(request))
        with self.assertRaises(core.EnhancerError):
            api.request_image('sk-test', b'not png', core.plan_image((64, 64), 2), core.Settings(), False, transport=transport)
        self.assertEqual(calls, [])

    def test_oversized_upload_never_sent(self):
        calls = []
        transport = httpx.MockTransport(lambda request: calls.append(request))
        with patch('enhancer.api.MAX_UPLOAD_BYTES', 5), self.assertRaises(core.EnhancerError):
            api.request_image('sk-test', b'\x89PNG\r\n\x1a\n', core.plan_image((64, 64), 2), core.Settings(), False, transport=transport)
        self.assertEqual(calls, [])

    def test_large_http_error_body_is_capped(self):
        plan = core.plan_image((64, 64), 2)
        png, alpha = core.make_canvas(Image.new('RGB', (64, 64)), plan)
        transport = httpx.MockTransport(lambda _: httpx.Response(500, content=b'x' * (300 * 1024)))
        with self.assertRaisesRegex(core.EnhancerError, 'HTTP 500'):
            api.request_image('sk-test', png, plan, core.Settings(), alpha, transport=transport)

    def test_non_json_http_error(self):
        plan = core.plan_image((64, 64), 2)
        png, alpha = core.make_canvas(Image.new('RGB', (64, 64)), plan)
        transport = httpx.MockTransport(lambda _: httpx.Response(502, content=b'<html>proxy error</html>'))
        with self.assertRaisesRegex(core.EnhancerError, 'HTTP 502'):
            api.request_image('sk-test', png, plan, core.Settings(), alpha, transport=transport)

    def test_postprocessing_failure_mentions_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'source.png'
            Image.new('RGB', (64, 64)).save(source)
            texture = core.load_texture(source)
            plan = core.plan_image(texture.image.size, 2)
            transport = httpx.MockTransport(lambda _: httpx.Response(200, content=response_body(plan.canvas_size)))
            with patch('enhancer.api.finish_image', side_effect=MemoryError()), self.assertRaisesRegex(core.EnhancerError, 'unmodified API canvas'):
                api.run_job(texture, core.Settings(), 'sk-test', folder / 'out', work_directory=folder / 'recovery', transport=transport)
            self.assertEqual(len(list((folder / 'recovery').glob('*_api-canvas.png'))), 1)

    def test_recovery_metadata_failure_does_not_lose_image(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'source.png'
            Image.new('RGB', (64, 64)).save(source)
            texture = core.load_texture(source)
            plan = core.plan_image(texture.image.size, 2)
            original_open = Path.open
            def open_file(path, *args, **kwargs):
                if path.suffix == '.json':
                    raise PermissionError('metadata write blocked')
                return original_open(path, *args, **kwargs)
            transport = httpx.MockTransport(lambda _: httpx.Response(200, content=response_body(plan.canvas_size)))
            with patch('enhancer.api.write_atomic', side_effect=PermissionError('metadata write blocked')):
                result = api.run_job(texture, core.Settings(keep_intermediates=True), 'sk-test', folder / 'out', transport=transport)
            self.assertTrue(result.saved_path.exists())
            self.assertTrue(result.raw_path.is_file())
            self.assertTrue(result.original_path.is_file())
            self.assertIn('Metadata', result.warnings[0])

    def test_atomic_publish_race_does_not_overwrite_other_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'out.png'
            real_link = os.link
            def race(src, dst):
                Path(dst).write_bytes(b'other output')
                real_link(src, dst)
            with patch('enhancer.storage.os.link', side_effect=race), self.assertRaises(core.EnhancerError):
                core.save_image_atomic(Image.new('RGB', (16, 16)), path)
            self.assertEqual(path.read_bytes(), b'other output')

    def test_windows_no_hardlink_fallback_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'out.png'
            proxy = Mock(wraps=os)
            proxy.name = 'nt'
            proxy.link.side_effect = OSError('filesystem has no hard links')
            with patch('enhancer.storage.os', proxy):
                core.save_image_atomic(Image.new('RGB', (16, 16)), path)
            proxy.rename.assert_called_once()
            self.assertTrue(path.is_file())

    def test_embedded_profile_is_warned_and_stripped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.png'
            Image.new('RGB', (16, 16)).save(path, icc_profile=b'test-profile')
            with self.assertRaisesRegex(core.EnhancerError, 'color profile'):
                core.load_texture(path)


if __name__ == '__main__':
    unittest.main()
