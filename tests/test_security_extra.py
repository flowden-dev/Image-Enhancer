import gzip
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from enhancer.api import request_image
from enhancer.core import EnhancerError, Settings, make_canvas, plan_image


class ResponseSafetyTests(unittest.TestCase):
    def test_compressed_http_rejected_before_body_decode(self):
        plan = plan_image((16, 8), 2)
        png, alpha = make_canvas(Image.new("RGBA", (16, 8), "red"), plan)
        consumed = []
        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                consumed.append(True)
                yield gzip.compress(b"a" * 1_000_000)
        def handler(request):
            self.assertEqual(request.headers["accept-encoding"], "identity")
            return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=Stream())
        with self.assertRaisesRegex(EnhancerError, "compressed"):
            request_image("sk-test-secret", png, plan, Settings(), alpha, transport=httpx.MockTransport(handler))
        self.assertEqual(consumed, [])
