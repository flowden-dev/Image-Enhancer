"""End-to-end tests for native outputs, failure recovery and the Windows UI flow.

Every HTTP request uses MockTransport. No live credentials or API credit are used.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from enhancer.api import run_job
from enhancer.core import EnhancerError, Settings, load_texture, plan_image
from enhancer.formats import _jpeg_sof, decode_source, encode_original, verify_export
from enhancer.formats import dds
from enhancer.formats.common import Profile, from_srgb
from format_fixtures import dds_fixture, raw_dds, tga_fixture, png_fixture, pattern
from test_api import response_body, fields_from_request
import test_gui as gui_support


def jpeg_fixture():
    stream = io.BytesIO()
    pattern((17, 11), opaque=True).convert("RGB").save(stream, format="JPEG", quality=87, progressive=True, subsampling=1)
    return stream.getvalue()


class NativePipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="native flow ü ")
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.calls = []

    def source(self, data, extension):
        path = self.root / ("Resource ü with spaces" + extension)
        path.write_bytes(data)
        return load_texture(path)

    def transport(self, texture, settings=Settings()):
        plan = plan_image(texture.image.size, settings.scale)
        def respond(request):
            self.calls.append(request)
            return httpx.Response(200, headers={"x-request-id": "req_native_test"}, content=response_body(plan.canvas_size))
        return httpx.MockTransport(respond)

    def fixtures(self):
        return ((".DDS", dds_fixture(levels=5)), (".tga", tga_fixture(rle=True, extension=True)),
                (".png", png_fixture(color=3, depth=4, interlace=1)), (".JPEG", jpeg_fixture()))

    def test_api_full_roundtrip_all_four_formats(self):
        for ext, data in self.fixtures():
            with self.subTest(ext=ext):
                texture = self.source(data, ext)
                before = len(self.calls)
                result = run_job(texture, Settings(keep_intermediates=True), "sk-native-test", self.root / "output", transport=self.transport(texture))
                self.assertEqual(len(self.calls), before + 1)
                self.assertEqual(result.original_path.name, texture.path.name)
                self.assertEqual(result.saved_path, result.original_path)
                self.assertEqual(texture.path.read_bytes(), data)
                verify_export(result.original_path.read_bytes(), texture.profile, result.plan.target_size)
                self.assertTrue(result.png_path.is_file())
                self.assertTrue(result.raw_path.is_file())
                self.assertEqual(result.warnings, ())
                fields = fields_from_request(self.calls[-1])
                self.assertEqual(fields["quality"], b"max")
                self.assertEqual(fields["complexity"], b"high")
                self.assertEqual(fields["n"], b"1")

    def test_local_methods_four_formats_no_key_no_http(self):
        transport = httpx.MockTransport(lambda request: self.fail("Local mode must not access the API"))
        for ext, data in self.fixtures():
            for method in ("lanczos", "nearest", "neural"):
                with self.subTest(ext=ext, method=method):
                    texture = self.source(data, ext)
                    result = run_job(texture, Settings(method=method), "", self.root / "output", transport=transport)
                    verify_export(result.original_path.read_bytes(), texture.profile, result.plan.target_size)
                    self.assertIsNone(result.raw_path)
                    self.assertEqual(result.request_id, "")
                    self.assertEqual(result.warnings, ())
                    self.assertEqual(texture.path.read_bytes(), data)

    def test_native_export_failure_keeps_paid_result_and_png(self):
        texture = self.source(dds_fixture(), ".dds")
        with patch("enhancer.api.save_original_atomic", side_effect=EnhancerError("simulated encoder failure")):
            result = run_job(texture, Settings(keep_intermediates=True), "sk-native-test", self.root / "output", transport=self.transport(texture))
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(result.original_path)
        self.assertEqual(result.saved_path, result.png_path)
        self.assertTrue(result.png_path.is_file())
        self.assertTrue(result.raw_path.is_file())
        # the retained native profile can export the same paid image without a new request
        output = encode_original(result.image, result.profile)
        verify_export(output, texture.profile, result.plan.target_size)
        self.assertIn("Original-format export failed", " ".join(result.warnings))

    def test_metadata_failure_does_not_lose_result(self):
        texture = self.source(png_fixture(), ".png")
        with patch("enhancer.api.write_atomic", side_effect=EnhancerError("metadata collision")):
            result = run_job(texture, Settings(method="nearest", keep_intermediates=True), "", self.root / "output")
        self.assertTrue(result.original_path.is_file())
        self.assertIn("Metadata could not be saved", " ".join(result.warnings))

    def test_native_preflight_failure_before_http(self):
        texture = self.source(dds_fixture(), ".dds")
        with patch("enhancer.api.preflight", side_effect=EnhancerError("unsupported encoder")):
            with self.assertRaisesRegex(EnhancerError, "unsupported encoder"):
                run_job(texture, Settings(keep_intermediates=True), "sk-native-test", self.root / "output", transport=self.transport(texture))
        self.assertEqual(self.calls, [])

    def test_low_disk_space_stops_before_http(self):
        texture = self.source(png_fixture(), ".png")
        with patch("enhancer.storage.shutil.disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(EnhancerError, "insufficient free space"):
                run_job(texture, Settings(keep_intermediates=True), "sk-native-test", self.root / "output", transport=self.transport(texture))
        self.assertEqual(self.calls, [])

    def test_native_filename_failure_stops_before_http(self):
        texture = self.source(dds_fixture(), ".dds")
        original_open = Path.open
        def open_checked(path, mode="r", *args, **kwargs):
            if mode == "xb":
                raise OSError("simulated Windows filename limitation")
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, "open", new=open_checked):
            with self.assertRaisesRegex(EnhancerError, "native output filename"):
                run_job(texture, Settings(keep_intermediates=True), "sk-native-test", self.root / "output", transport=self.transport(texture))
        self.assertEqual(self.calls, [])

    def test_alpha_only_dds_local_only(self):
        texture = self.source(raw_dds(bits=8, masks=(0, 0, 0, 255), kind="alpha"), ".dds")
        with self.assertRaisesRegex(EnhancerError, "data textures"):
            run_job(texture, Settings(keep_intermediates=True), "sk-native-test", self.root / "output", transport=self.transport(texture))
        self.assertEqual(self.calls, [])
        result = run_job(texture, Settings(method="nearest"), "", self.root / "output")
        decoded = dds.decode(result.original_path.read_bytes(), dds.inspect(result.original_path.read_bytes()))
        expected = texture.image.getchannel("A").resize(result.plan.target_size, Image.Resampling.NEAREST)
        self.assertEqual(decoded.getchannel("A").tobytes(), expected.tobytes())

    def test_same_source_multiple_jobs_never_collide(self):
        texture = self.source(dds_fixture(), ".dds")
        settings = Settings(method="nearest")
        first = run_job(texture, settings, "", self.root / "output")
        second = run_job(texture, settings, "", self.root / "output")
        self.assertNotEqual(first.original_path, second.original_path)
        self.assertEqual(first.original_path.parent, second.original_path.parent)
        self.assertEqual(second.original_path.name, first.original_path.stem + " (2).dds")
        self.assertTrue(first.original_path.exists())
        self.assertTrue(second.original_path.exists())

    def test_audit_json_retains_source_hash_not_secrets_or_prompt(self):
        texture = self.source(png_fixture(), ".png")
        secret = "sk-native-secret-key"
        settings = Settings(instruction="private project instructions", source_protection="off", keep_intermediates=True)
        result = run_job(texture, settings, secret, self.root / "output", transport=self.transport(texture, settings))
        text = result.metadata_path.read_text()
        data = json.loads(text)
        self.assertNotIn(secret, text)
        self.assertNotIn(settings.instruction, text)
        self.assertEqual(data["source_sha256"], texture.sha256)
        self.assertTrue(data["original_saved"])

    def test_large_dds_local_target_is_exact_not_api_limited(self):
        texture = self.source(dds_fixture(size=(1024, 4)), ".dds")
        result = run_job(texture, Settings(scale=4, method="nearest"), "", self.root / "output")
        self.assertEqual(result.image.size, (4096, 16))
        verify_export(result.original_path.read_bytes(), texture.profile, (4096, 16))

    def test_extended_sequential_jpeg_marker_ignores_comment_bytes(self):
        comment = b"data containing a false SOF marker: \xff\xc0 x"
        stream = io.BytesIO()
        pattern(opaque=True).convert("RGB").save(stream, format="JPEG", comment=comment)
        raw = bytearray(stream.getvalue())
        _, offset = _jpeg_sof(raw, with_offset=True)
        raw[offset] = 0xC1
        texture = self.source(raw, ".jpg")
        output = encode_original(texture.image.resize((34, 22)), texture.profile)
        self.assertEqual(_jpeg_sof(output), 0xC1)
        with Image.open(io.BytesIO(output)) as image:
            self.assertEqual(image.info["comment"], comment)

    def test_contradictory_and_bump_dds_flags_rejected(self):
        for flags in (0x40 | 0x20000, 0x40 | 0x80000, 0x40 | 0x40000, 0x40 | 0x20):
            data = bytearray(raw_dds())
            struct.pack_into("<I", data, 80, flags)
            with self.subTest(flags=flags), self.assertRaises(EnhancerError):
                self.source(data, ".dds")

    def test_bc1_one_bit_alpha_remains_alpha(self):
        texture = self.source(dds_fixture("DXT1", (8, 8)), ".dds")
        image = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        image.paste((0, 0, 255, 0), (0, 0, 8, 8))
        output = encode_original(image, texture.profile)
        decoded = dds.decode(output, dds.inspect(output))
        self.assertEqual(decoded.getpixel((0, 0))[3], 0)
        self.assertEqual(decoded.getpixel((15, 15))[3], 255)

    def test_grayscale_icc_reverse_transform_retains_transparency(self):
        # isolate the alpha-handling boundary from platform-specific ICC fixtures
        profile = Profile("PNG", "L", (2, 2), icc=b"gray-profile")
        image = Image.new("RGBA", (2, 2), (10, 10, 10, 255))
        image.putpixel((0, 0), (50, 50, 50, 0))
        with patch("enhancer.formats.common.ImageCms.ImageCmsProfile"), \
             patch("enhancer.formats.common.ImageCms.profileToProfile", return_value=Image.new("L", (2, 2), 50)):
            restored = from_srgb(image, profile)
        self.assertEqual(restored.mode, "LA")
        self.assertEqual(restored.getchannel("A").tobytes(), image.getchannel("A").tobytes())


class NativeGuiTests(unittest.TestCase):
    setUpClass = classmethod(gui_support.GuiTests.setUpClass.__func__)
    setUp = gui_support.GuiTests.setUp
    tearDown = gui_support.GuiTests.tearDown
    result_for = gui_support.GuiTests.result_for
    wait = gui_support.GuiTests.wait

    def edited_png(self, size=(128, 64)):
        path = self.root / "edited.png"
        Image.new("RGBA", size, (20, 50, 80, 255)).save(path)
        return path

    def test_local_gui_requires_no_key_or_paid_confirmation(self):
        self.app.load_file(self.source)
        self.app.key.set("")
        self.app.method.set("Local nearest")
        self.assertEqual(self.app.run_button["text"], "Upscale Image")
        self.app.process()
        self.wait()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][2], "")
        self.dialogs["askyesno"].assert_not_called()

    def test_import_png_can_export_native_without_paid_request(self):
        source = self.root / "original.dds"
        source.write_bytes(dds_fixture(size=(64, 32), levels=7))
        self.app.load_file(source)
        with patch("enhancer.gui.filedialog.askopenfilename", return_value=str(self.edited_png())):
            self.app.import_edited_png()
        self.assertEqual(self.calls, [])
        self.assertIsNotNone(self.app.result)
        self.assertIsNone(self.app.result.saved_path)
        self.assertEqual(self.app.result.profile.format, "DDS")
        destination = self.root / "saved.dds"
        with patch("enhancer.gui.filedialog.asksaveasfilename", return_value=str(destination)):
            self.app.save_result()
        self.wait()
        self.assertTrue(destination.is_file())
        verify_export(destination.read_bytes(), self.app.texture.profile, (128, 64))
        self.assertEqual(self.calls, [])

    def test_import_wrong_size_rejected_and_previous_result_retained(self):
        self.app.load_file(self.source)
        previous = self.result_for(self.app.texture, Settings())
        self.app.result = previous
        with patch("enhancer.gui.filedialog.askopenfilename", return_value=str(self.edited_png((1024, 1024)))):
            self.app.import_edited_png()
        self.assertIs(self.app.result, previous)
        self.dialogs["showerror"].assert_called_once()
        self.assertEqual(self.calls, [])

    def test_import_requires_original(self):
        self.app.import_edited_png()
        self.dialogs["showwarning"].assert_called_once()
        self.assertIsNone(self.app.result)

    def test_imported_modified_alpha_is_unsaved_and_protected(self):
        self.app.load_file(self.source)
        with patch("enhancer.gui.filedialog.askopenfilename", return_value=str(self.edited_png())):
            self.app.import_edited_png()
        self.assertEqual(self.app.result.image.getchannel("A").getextrema(), (128, 128))
        self.assertIsNone(self.app.result.saved_path)
        self.dialogs["askyesno"].return_value = False
        self.assertFalse(self.app.load_file(self.source))
        self.assertIsNotNone(self.app.result)

    def test_png_export_completes_on_worker(self):
        self.app.load_file(self.source)
        self.app.result = self.result_for(self.app.texture, Settings())
        destination = self.root / "manual.png"
        with patch("enhancer.gui.filedialog.asksaveasfilename", return_value=str(destination)):
            self.app.export_png()
        self.wait()
        self.assertTrue(destination.is_file())
        self.assertEqual(self.app.result.png_path, destination)
        self.assertEqual(self.calls, [])

    def test_png_export_cannot_overwrite_original(self):
        self.app.load_file(self.source)
        original = self.source.read_bytes()
        self.app.result = self.result_for(self.app.texture, Settings())
        with patch("enhancer.gui.filedialog.asksaveasfilename", return_value=str(self.source)):
            self.app.export_png()
        self.wait()
        self.assertEqual(self.source.read_bytes(), original)
        self.dialogs["showerror"].assert_called_once()

    def test_local_close_warning_does_not_claim_a_paid_request(self):
        self.app._set_busy(True)
        self.dialogs["askyesno"].return_value = False
        self.app.on_close()
        message = self.dialogs["askyesno"].call_args.args[1]
        self.assertIn("Local processing", message)
        self.assertNotIn("charged", message)
        self.app._set_busy(False)

    def test_native_export_worker_start_failure_restores_controls(self):
        self.app.load_file(self.source)
        self.app.result = self.result_for(self.app.texture, Settings())
        with patch("enhancer.gui.filedialog.asksaveasfilename", return_value=str(self.root / "out.png")), \
             patch("enhancer.gui.threading.Thread.start", side_effect=RuntimeError("thread failure")):
            self.app.save_result()
        self.assertFalse(self.app._busy)
        self.assertFalse(self.app._active_paid)
        self.assertEqual(str(self.app.open_button["state"]), "normal")


if __name__ == "__main__":
    unittest.main()
