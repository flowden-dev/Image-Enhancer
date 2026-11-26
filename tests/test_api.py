from __future__ import annotations

import base64
import email
from email import policy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from enhancer import __version__
from enhancer.api import ENDPOINT, ApiImage, decode_response, redact, request_image, run_job, validate_key
from enhancer.core import EnhancerError, MODEL, Settings, load_texture, make_canvas, plan_image


def png_bytes(size=(1024, 1024), color=(120, 80, 50, 255)):
    stream = io.BytesIO()
    Image.new("RGBA", size, color).save(stream, format="PNG")
    return stream.getvalue()


def response_body(size=(1024, 1024), **extra):
    return json.dumps({"data": [{"b64_json": base64.b64encode(png_bytes(size)).decode("ascii")}], **extra}).encode()


def fields_from_request(request):
    message = email.message_from_bytes(
        b"Content-Type: " + request.headers["content-type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.read(),
        policy=policy.default,
    )
    return {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()}


class DecodeTests(unittest.TestCase):
    def test_success(self):
        result = decode_response(response_body(usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}), "req_test")
        self.assertEqual(result.image.size, (1024, 1024))
        self.assertEqual(result.request_id, "req_test")
        self.assertEqual(result.usage["total_tokens"], 30)

    def test_absent_usage(self):
        self.assertEqual(decode_response(response_body()).usage, {})

    def test_invalid_usage_is_not_trusted(self):
        result = decode_response(response_body(usage={"input_tokens": -10, "output_tokens": True, "total_tokens": "nan"}))
        self.assertEqual(result.usage, {})

    def test_non_json(self):
        with self.assertRaises(EnhancerError):
            decode_response(b"<html>error</html>")

    def test_non_object(self):
        for body in (b"[]", b"null", b"123"):
            with self.subTest(body=body), self.assertRaises(EnhancerError):
                decode_response(body)

    def test_absent_or_empty_images(self):
        for value in (None, [], {}, [None], [{}, {}]):
            with self.subTest(value=value), self.assertRaises(EnhancerError):
                decode_response(json.dumps({"data": value}).encode())

    def test_empty_or_missing_base64(self):
        for value in (None, "", 123):
            with self.subTest(value=value), self.assertRaises(EnhancerError):
                decode_response(json.dumps({"data": [{"b64_json": value}]}).encode())

    def test_url_not_followed(self):
        with self.assertRaisesRegex(EnhancerError, "URLs"):
            decode_response(b'{"data":[{"url":"https://example.com/image.png"}]}')

    def test_invalid_base64(self):
        for value in ("!!!!", "not base64", "\u20ac"):
            with self.subTest(value=value), self.assertRaises(EnhancerError):
                decode_response(json.dumps({"data": [{"b64_json": value}]}).encode())

    def test_invalid_image(self):
        body = json.dumps({"data": [{"b64_json": base64.b64encode(b"not png").decode()}]}).encode()
        with self.assertRaises(EnhancerError):
            decode_response(body)

    def test_jpeg_instead_of_png(self):
        stream = io.BytesIO()
        Image.new("RGB", (16, 16)).save(stream, format="JPEG")
        with self.assertRaises(EnhancerError):
            decode_response(json.dumps({"data": [{"b64_json": base64.b64encode(stream.getvalue()).decode()}]}).encode())

    def test_body_limit(self):
        with patch("enhancer.api.MAX_RESPONSE_BYTES", 100), self.assertRaises(EnhancerError):
            decode_response(response_body())

    def test_image_limit(self):
        with patch("enhancer.api.MAX_IMAGE_BYTES", 100), self.assertRaises(EnhancerError):
            decode_response(response_body())

    def test_pixel_limit(self):
        with patch("enhancer.api.MAX_SOURCE_PIXELS", 100), self.assertRaises(EnhancerError):
            decode_response(response_body())


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_image((64, 64), 2)
        self.png, self.alpha = make_canvas(Image.new("RGBA", (64, 64), (20, 30, 40, 128)), self.plan)
        self.key = "sk-proj-offline-test-secret"

    def test_request_contract(self):
        calls = []
        def handler(request):
            calls.append(request)
            self.assertEqual(str(request.url), ENDPOINT)
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.headers["user-agent"], f"Image-Enhancer/{__version__}")
            self.assertEqual(request.headers["authorization"], f"Bearer {self.key}")
            fields = fields_from_request(request)
            self.assertEqual(fields["model"], MODEL.encode())
            self.assertEqual(fields["quality"], b"max")
            self.assertEqual(fields["size"], self.plan.size_string.encode())
            self.assertEqual(fields["n"], b"1")
            self.assertEqual(fields["output_format"], b"png")
            self.assertEqual(fields["background"], b"transparent")
            self.assertEqual(fields["image[]"], self.png)
            self.assertNotIn("input_fidelity", fields)
            return httpx.Response(200, content=response_body(self.plan.canvas_size), headers={"x-request-id": "req_mock"})
        result = request_image(self.key, self.png, self.plan, Settings(), True, transport=httpx.MockTransport(handler))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.request_id, "req_mock")

    def test_opaque_background(self):
        def handler(request):
            self.assertEqual(fields_from_request(request)["background"], b"opaque")
            return httpx.Response(200, content=response_body(self.plan.canvas_size))
        request_image(self.key, self.png, self.plan, Settings(), False, transport=httpx.MockTransport(handler))

    def test_http_failures_no_retries(self):
        for status in (400, 401, 403, 404, 408, 413, 429, 500, 502, 503):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(status, json={"error": {"message": "simulated error"}}, headers={"x-request-id": "req_fail"})
            with self.subTest(status=status), self.assertRaisesRegex(EnhancerError, f"HTTP {status}"):
                request_image(self.key, self.png, self.plan, Settings(), True, transport=httpx.MockTransport(handler))
            self.assertEqual(len(calls), 1)

    def test_redirect_not_followed(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(307, headers={"location": "https://untrusted.invalid/"})
        with self.assertRaisesRegex(EnhancerError, "HTTP 307"):
            request_image(self.key, self.png, self.plan, Settings(), True, transport=httpx.MockTransport(handler))
        self.assertEqual(len(calls), 1)

    def test_timeout_no_retries(self):
        calls = []
        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout("simulated", request=request)
        with self.assertRaisesRegex(EnhancerError, "charged"):
            request_image(self.key, self.png, self.plan, Settings(), True, transport=httpx.MockTransport(handler))
        self.assertEqual(len(calls), 1)

    def test_connection_failure_no_retries(self):
        calls = []
        def handler(request):
            calls.append(request)
            raise httpx.ConnectError("simulated", request=request)
        with self.assertRaisesRegex(EnhancerError, "No automatic retry"):
            request_image(self.key, self.png, self.plan, Settings(), True, transport=httpx.MockTransport(handler))
        self.assertEqual(len(calls), 1)

    def test_response_stream_limit(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"X" * 200))
        with patch("enhancer.api.MAX_RESPONSE_BYTES", 100), self.assertRaises(EnhancerError):
            request_image(self.key, self.png, self.plan, Settings(), True, transport=transport)

    def test_key_redacted_from_api_error(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(401, json={"error": {"message": "Bad " + self.key}},
                                                                headers={"x-request-id": self.key}))
        with self.assertRaises(EnhancerError) as raised:
            request_image(self.key, self.png, self.plan, Settings(), True, transport=transport)
        self.assertNotIn(self.key, str(raised.exception))

    def test_no_key_no_request(self):
        calls = []
        transport = httpx.MockTransport(lambda request: calls.append(request))
        with self.assertRaises(EnhancerError):
            request_image("", self.png, self.plan, Settings(), True, transport=transport)
        self.assertEqual(calls, [])

    def test_invalid_setting_no_request(self):
        calls = []
        transport = httpx.MockTransport(lambda request: calls.append(request))
        with self.assertRaises(EnhancerError):
            request_image(self.key, self.png, self.plan, Settings(quality="unsupported"), True, transport=transport)
        self.assertEqual(calls, [])

    def test_environment_cannot_redirect_key(self):
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "http://untrusted.invalid", "HTTPS_PROXY": "http://untrusted.invalid"}):
            def handler(request):
                self.assertEqual(str(request.url), ENDPOINT)
                return httpx.Response(200, content=response_body(self.plan.canvas_size))
            request_image(self.key, self.png, self.plan, Settings(), True, transport=httpx.MockTransport(handler))

    def test_key_validation(self):
        self.assertEqual(validate_key("  sk-valid-test  "), "sk-valid-test")
        for key in ("", "x\ny", "x y", "x\ty", "é", "x" * 513, "x\x00y"):
            with self.subTest(key=repr(key)), self.assertRaises(EnhancerError):
                validate_key(key)

    def test_redaction(self):
        self.assertNotIn("sk-secret-value", redact("API sk-secret-value failed"))
        self.assertEqual(redact("A\x00B\nC"), "AB\nC")


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="job with spaces ")
        self.root = Path(self.temp.name)
        path = self.root / "button.tga"
        Image.new("RGBA", (64, 32), (30, 70, 150, 128)).save(path)
        self.texture = load_texture(path)
        self.settings = Settings(keep_intermediates=True)
        self.plan = plan_image(self.texture.image.size, self.settings.scale)
        self.key = "sk-proj-offline-secret"
        self.calls = []
        def handler(request):
            self.calls.append(request)
            return httpx.Response(200, content=response_body(self.plan.canvas_size), headers={"x-request-id": "req_job"})
        self.transport = httpx.MockTransport(handler)

    def tearDown(self):
        self.temp.cleanup()

    def job(self, **kwargs):
        return run_job(self.texture, self.settings, self.key, self.root / "outputs",
                       work_directory=self.root / "recovery", transport=kwargs.pop("transport", self.transport), **kwargs)

    def test_complete_round_trip_and_recovery(self):
        original = self.texture.path.read_bytes()
        result = self.job()
        self.assertEqual(result.image.size, (128, 64))
        self.assertTrue(result.saved_path.is_file())
        self.assertTrue(result.raw_path.is_file())
        self.assertEqual(self.texture.path.read_bytes(), original)
        self.assertEqual(list((self.root / "outputs").iterdir()), [result.original_path])
        metadata = result.metadata_path.read_text()
        self.assertNotIn(self.key, metadata)
        self.assertNotIn(str(self.root), metadata)
        self.assertEqual(json.loads(metadata)["scale"], 2)
        self.assertEqual(result.warnings, ())
        self.assertEqual(len(self.calls), 1)

    def test_unique_recovery_names(self):
        first = self.job()
        second = self.job(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=response_body(self.plan.canvas_size))))
        self.assertNotEqual(first.saved_path, second.saved_path)
        self.assertTrue(first.saved_path.exists())

    def test_invalid_recovery_folder_no_request(self):
        (self.root / "outputs").write_text("not a folder")
        with self.assertRaises(OSError):
            self.job()
        self.assertEqual(self.calls, [])

    def test_fully_transparent_no_request(self):
        self.texture.image.putalpha(0)
        with self.assertRaises(EnhancerError):
            self.job()
        self.assertEqual(self.calls, [])

    def test_wrong_dimensions_raw_saved(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, content=response_body((128, 128))))
        with self.assertRaisesRegex(EnhancerError, "unmodified API canvas"):
            self.job(transport=transport)
        self.assertEqual(len(list((self.root / "recovery").glob("*_api-canvas.png"))), 1)
        self.assertEqual(len(list((self.root / "recovery").glob("*_enhanced.png"))), 0)
        self.assertEqual(list((self.root / "outputs").iterdir()), [])

    def test_disk_failure_keeps_result_in_memory(self):
        with patch("enhancer.api.save_image_atomic", side_effect=OSError("disk full")), patch("enhancer.api.save_original_atomic", side_effect=OSError("disk full")):
            result = self.job()
        self.assertEqual(result.image.size, (128, 64))
        self.assertIsNone(result.saved_path)
        self.assertIsNone(result.raw_path)
        self.assertEqual(len(result.warnings), 3)

    def test_final_save_failure_keeps_raw(self):
        from enhancer.core import save_image_atomic
        def save(image, path):
            if path.name.endswith("_enhanced.png"):
                raise OSError("disk full")
            return save_image_atomic(image, path)
        with patch("enhancer.api.save_image_atomic", side_effect=save):
            result = self.job()
        self.assertTrue(result.raw_path.exists())
        self.assertIsNotNone(result.original_path)
        self.assertTrue(result.saved_path.exists())
        self.assertIsNone(result.png_path)
        self.assertTrue(result.warnings)


if __name__ == "__main__":
    unittest.main()
