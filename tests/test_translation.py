"""Offline translation contract, region geometry, privacy and native pipeline tests."""
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from enhancer.api import ApiImage, request_image, run_job
from enhancer.batch import prepare_batch, run_batch
from enhancer.core import EnhancerError, Settings, build_prompt, load_texture, make_canvas, plan_settings
from enhancer.formats import verify_export
from enhancer.translation import (canvas_boxes, composite_translation, finish_translation,
                                   make_translation_mask, map_box, parse_replacements)
from format_fixtures import dds_fixture, png_fixture, raw_dds, tga_fixture
from test_api import fields_from_request, response_body
from test_native_pipeline import jpeg_fixture


def settings(**kwargs):
    return replace(Settings(task='translate', scale=1, text_regions=((3, 2, 20, 12),)), **kwargs)


class TranslationTests(unittest.TestCase):
    def test_original_size_and_all_resolution_plans(self):
        for source in ((64, 32), (1024, 768), (4096, 4096), (37, 11)):
            config = settings(translation_scope='all', text_regions=())
            self.assertEqual(plan_settings(source, config).target_size, source)
        self.assertEqual(plan_settings((64, 32), settings(scale=4)).target_size, (256, 128))
        self.assertEqual(plan_settings((64, 32), settings(target_edge=2048)).target_size, (2048, 1024))

    def test_invalid_translation_settings_rejected(self):
        for change in (dict(method='nearest'), dict(task='bad'), dict(source_language=''),
                       dict(target_language='Auto-detect'), dict(source_language='german'),
                       dict(target_language='a'*65), dict(target_language='a\nGerman'),
                       dict(target_language='\ud800'), dict(target_language=42),
                       dict(translation_scope='bad'), dict(text_regions=()),
                       dict(text_regions=((0, 0, 0, 2),)), dict(text_regions=((True, 0, 2, 2),)),
                       dict(text_regions=((1.5, 0, 3, 3),)), dict(text_regions=([1, 1, 2, 2],)),
                       dict(translation_scope='all'), dict(instruction='redesign logo'),
                       dict(keep_names='yes'), dict(protected_terms='x'*2001),
                       dict(text_regions=((0, 0, 8, 8), (7, 7, 10, 10)))):
            with self.subTest(change=change), self.assertRaises(EnhancerError):
                settings(**change).validate()
        settings(source_language='Chinese (Traditional)', target_language='Deutsch').validate()

    def test_region_limits_bounds_and_touching_edges(self):
        with self.assertRaises(EnhancerError):
            plan_settings((64, 32), settings(text_regions=((0, 0, 65, 1),)))
        with self.assertRaises(EnhancerError):
            settings(text_regions=tuple((i, 0, i+1, 1) for i in range(33))).validate()
        plan_settings((64, 32), settings(text_regions=tuple((i, 0, i+1, 1) for i in range(32))))

    def test_exact_wording_unicode_validation(self):
        self.assertEqual(parse_replacements(' PLAY => SPIELEN\n\nStart => 開始 '), (('PLAY', 'SPIELEN'), ('Start', '開始')))
        self.assertEqual(parse_replacements('PLAY => SPIELEN\r\nEXIT => BEENDEN\r\n'),
                         (('PLAY', 'SPIELEN'), ('EXIT', 'BEENDEN')))
        settings(protected_terms='Field Notes\r\nAtelier').validate()
        for value in ('PLAY', '=> Translation', 'PLAY =>', 'A => B\nA => C', 'A => B\x00', 'x'*4001, '\ud800 => x'):
            with self.subTest(value=repr(value)), self.assertRaises(EnhancerError):
                parse_replacements(value)

    def test_translation_prompt_does_not_contradict_text_change(self):
        config = settings(replacements='PLAY => SPIELEN', protected_terms='Field Notes')
        prompt = build_prompt(plan_settings((64, 32), config), config)
        for text in ('SPIELEN', 'German', 'Field Notes', 'baseline', 'shadows', 'transparency',
                     'ghost lettering', 'only where the original', 'transparent mask rectangles'):
            self.assertIn(text, prompt)
        self.assertNotIn('Do not re-typeset', prompt)
        self.assertNotIn('Preserve every silhouette', prompt)
        self.assertIn('Titles and logos may be translated', build_prompt(plan_settings((64, 32), config), replace(config, keep_names=False)))

    def test_mask_matches_canvas_and_only_selected_pixels_transparent(self):
        config = settings(text_regions=((1, 1, 2, 3), (20, 2, 30, 5)))
        plan = plan_settings((64, 16), config)
        with Image.open(io.BytesIO(make_translation_mask(plan, config))) as mask:
            self.assertEqual(mask.size, plan.canvas_size)
            self.assertEqual(mask.mode, 'RGBA')
            alpha = np.array(mask.getchannel('A'))
            expected = np.full((mask.height, mask.width), 255, dtype=np.uint8)
            for x0,y0,x1,y1 in canvas_boxes(plan, config):
                expected[y0:y1, x0:x1] = 0
            np.testing.assert_array_equal(alpha, expected)
            self.assertEqual(mask.getpixel((0, 0))[3], 255)
        self.assertIsNone(make_translation_mask(plan, Settings()))
        self.assertIsNone(make_translation_mask(plan, settings(translation_scope='all', text_regions=())))

    def test_mapping_rounds_outwards_at_odd_sizes(self):
        self.assertEqual(map_box((1, 1, 2, 2), (3, 3), (2, 2), (10, 20)), (10, 20, 12, 22))
        self.assertEqual(map_box((2, 2, 3, 3), (3, 3), (6, 6)), (4, 4, 6, 6))

    def test_outside_pixels_and_hidden_rgb_exact_at_original_size(self):
        rng = np.random.default_rng(84)
        original = rng.integers(0, 256, (32,64,4), dtype=np.uint8)
        source = Image.fromarray(original)
        edited = Image.new('RGBA', source.size, (255, 1, 2, 0))
        result = composite_translation(edited, source, settings())
        expected = original.copy()
        expected[2:12, 3:20] = (255, 1, 2, 0)
        np.testing.assert_array_equal(np.array(result), expected)
        np.testing.assert_array_equal(np.array(source), original)

    def test_new_transparent_glyphs_not_clipped_by_old_alpha(self):
        source = Image.new('RGBA', (64,32), (10,20,30,0))
        source.putpixel((4,4), (255,255,255,255))
        edited = Image.new('RGBA', source.size, (40,50,60,0))
        edited.putpixel((9,4), (255,0,0,255))
        result = composite_translation(edited, source, settings())
        self.assertEqual(result.getpixel((4,4))[3], 0)
        self.assertEqual(result.getpixel((9,4)), (255,0,0,255))
        self.assertEqual(result.getpixel((0,0)), source.getpixel((0,0)))

    def test_opaque_source_and_opaque_panel_do_not_gain_transparency(self):
        opaque = Image.new('RGBA', (64,32), (50,60,70,255))
        edited = Image.new('RGBA', opaque.size, (90,100,110,0))
        self.assertEqual(composite_translation(edited, opaque, settings()).mode, 'RGB')
        opaque.putpixel((0,0), (50,60,70,0))
        result = composite_translation(edited, opaque, settings())
        self.assertEqual(result.getpixel((4,4)), (90,100,110,255))
        self.assertEqual(result.getpixel((0,0))[3], 0)

    def test_resized_outside_uses_source_reference(self):
        source = Image.new('RGBA', (64,32), (10,20,30,50))
        edited = Image.new('RGBA', (128,64), (210,220,230,255))
        result = composite_translation(edited, source, settings(scale=2))
        expected = source.resize(edited.size, Image.Resampling.LANCZOS)
        expected.paste(edited.crop((6,4,40,24)), (6,4,40,24))
        self.assertEqual(result.tobytes(), expected.tobytes())

    def test_wrong_response_size_fails_instead_of_shifting_text(self):
        config = settings()
        source = Image.new('RGBA', (64,32))
        plan = plan_settings(source.size, config)
        with self.assertRaises(EnhancerError):
            finish_translation(Image.new('RGBA', (9,9)), source, plan, config)

    def test_mask_multipart_and_unrestricted_translation_no_color_clamp(self):
        config = settings()
        source = Image.new('RGBA', (64,32), (200,200,200,255))
        plan = plan_settings(source.size, config)
        png, alpha = make_canvas(source, plan)
        fields = []
        def handler(req):
            fields.append(fields_from_request(req))
            return httpx.Response(200, content=response_body(plan.canvas_size))
        request_image('sk-test', png, plan, config, alpha, transport=httpx.MockTransport(handler))
        self.assertEqual(len(fields), 1)
        self.assertIn('mask', fields[0])
        self.assertIn(b'German', fields[0]['prompt'])
        self.assertEqual(fields[0]['n'], b'1')
        with Image.open(io.BytesIO(fields[0]['mask'])) as mask:
            self.assertEqual(mask.size, plan.canvas_size)
        fields.clear()
        request_image('sk-test', png, plan, settings(translation_scope='all', text_regions=()), alpha,
                      transport=httpx.MockTransport(handler))
        self.assertNotIn('mask', fields[0])


class TranslationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root/'source.png'
        self.image = Image.new('RGBA', (64,32), (230,230,230,255))
        self.image.save(self.path)
        self.texture = load_texture(self.path)

    def test_guarded_output_and_private_wording_not_in_records(self):
        config = settings(replacements='PRIVATE => GEHEIM', protected_terms='CONFIDENTIAL', keep_intermediates=True)
        plan = plan_settings(self.image.size, config)
        raw = Image.new('RGBA', plan.canvas_size, (0,0,0,255))
        before = self.path.read_bytes()
        with patch('enhancer.api.request_image', return_value=ApiImage(raw,'mock',{})) as request:
            result = run_job(self.texture, config, 'sk-test', self.root/'out')
        request.assert_called_once()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(result.image.getpixel((4,4)), (0,0,0))
        self.assertEqual(result.image.getpixel((0,0)), (230,230,230))
        with Image.open(result.original_path) as native:
            self.assertEqual(native.size, (64,32))
            self.assertEqual(native.convert('RGB').getpixel((0,0)), (230,230,230))
        record = result.metadata_path.read_text()
        for private in ('PRIVATE', 'GEHEIM', 'CONFIDENTIAL', 'sk-test'):
            self.assertNotIn(private, record)
        metadata = json.loads(record)
        self.assertEqual(metadata['task'], 'translate')
        self.assertEqual(metadata['text_area_count'], 1)
        self.assertEqual(metadata['preserve_alpha'], 'automatic-text-areas')
        self.assertIsNone(metadata['source_protection'])
        self.assertTrue(result.raw_path.is_file())

    def test_invalid_region_before_any_paid_request(self):
        with patch('enhancer.api.request_image') as request, self.assertRaises(EnhancerError):
            run_job(self.texture, settings(text_regions=((0,0,65,2),)), 'sk-test', self.root/'out')
        request.assert_not_called()

    def test_translation_native_roundtrip_all_formats_and_one_request(self):
        fixtures = (('.dds', dds_fixture(levels=5)), ('.DDS', raw_dds()),
                    ('.tga', tga_fixture(rle=True, extension=True)),
                    ('.png', png_fixture(color=3, depth=4, interlace=1)),
                    ('.jpg', jpeg_fixture()))
        config = settings(text_regions=((1, 1, 6, 4),))
        for extension, data in fixtures:
            with self.subTest(extension=extension):
                source = self.root / ('native' + extension)
                source.write_bytes(data)
                texture = load_texture(source)
                plan = plan_settings(texture.image.size, config)
                requests = []
                def respond(request):
                    requests.append(fields_from_request(request))
                    return httpx.Response(200, content=response_body(plan.canvas_size))
                result = run_job(texture, config, 'sk-test', self.root/'native-output',
                                 transport=httpx.MockTransport(respond))
                self.assertEqual(len(requests), 1)
                self.assertIn('mask', requests[0])
                self.assertEqual(result.image.size, texture.image.size)
                self.assertEqual(result.warnings, ())
                self.assertEqual(source.read_bytes(), data)
                verify_export(result.original_path.read_bytes(), texture.profile, texture.image.size)
                actual = np.array(result.image.convert('RGBA'))
                original = np.array(texture.image.convert('RGBA'))
                outside = np.ones(actual.shape[:2], dtype=bool)
                outside[1:4, 1:6] = False
                np.testing.assert_array_equal(actual[outside], original[outside])

    def test_translation_native_export_failure_keeps_composited_png_and_raw(self):
        config = settings()
        plan = plan_settings(self.image.size, config)
        raw = Image.new('RGBA', plan.canvas_size, (0, 0, 0, 255))
        with patch('enhancer.api.request_image', return_value=ApiImage(raw, 'mock', {})) as request, \
             patch('enhancer.api.save_original_atomic', side_effect=EnhancerError('encoder failure')):
            result = run_job(self.texture, config, 'sk-test', self.root/'failed-export')
        request.assert_called_once()
        self.assertIsNone(result.original_path)
        self.assertEqual(result.saved_path, result.png_path)
        self.assertTrue(result.raw_path.is_file())
        with Image.open(result.png_path) as backup:
            self.assertEqual(backup.convert('RGB').getpixel((0, 0)), (230, 230, 230))
            self.assertEqual(backup.convert('RGB').getpixel((4, 4)), (0, 0, 0))

    def test_batch_rejects_source_specific_boxes(self):
        with self.assertRaisesRegex(EnhancerError, 'belong to one image'):
            prepare_batch((self.path,), settings(), threading.Event())
        with self.assertRaisesRegex(EnhancerError, 'belong to one image'):
            run_batch((), settings(), 'sk-test', self.root/'out', threading.Event())

    def test_batch_all_text_completes_and_records_operation(self):
        config = settings(translation_scope='all', text_regions=(), keep_intermediates=True)
        stop = threading.Event()
        items = prepare_batch((self.path,), config, stop)
        plan = plan_settings(self.image.size, config)
        with patch('enhancer.api.request_image', return_value=ApiImage(Image.new('RGBA', plan.canvas_size, (0,0,0,255)),'mock',{})):
            summary = run_batch(items, config, 'sk-test', self.root/'out', stop)
        self.assertEqual((summary.completed, summary.failed), (1,0))
        record = json.loads((summary.recovery_directory/'batch.json').read_text())
        self.assertEqual(record['task'], 'translate')
        self.assertEqual(record['target_language'], 'German')
