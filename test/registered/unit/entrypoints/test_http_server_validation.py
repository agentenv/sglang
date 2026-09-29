import asyncio
import json
import unittest
from types import SimpleNamespace

from fastapi.exceptions import RequestValidationError

from sglang.srt.entrypoints.http_server import validation_exception_handler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestHttpServerValidation(unittest.TestCase):
    def test_tensor_update_validation_does_not_reflect_payload(self):
        """A malformed /update_weights_from_tensor body must not be echoed back:
        Miles' per-rank CUDA-IPC payloads can be hundreds of MB of base64."""
        blob = "A" * 1_000_000
        exc = RequestValidationError(
            errors=[
                {
                    "type": "missing",
                    "loc": (
                        "body",
                        "function-after[build_struct(), "
                        "UpdateWeightsFromTensorReqInput]",
                        "serialized_named_tensors",
                    ),
                    "msg": "Field required",
                    "input": {
                        "load_format": "flattened_bucket",
                        "serialized_tensors": [blob],
                    },
                }
            ]
        )
        request = SimpleNamespace(
            url=SimpleNamespace(path="/update_weights_from_tensor")
        )

        response = asyncio.run(validation_exception_handler(request, exc))
        body = json.loads(response.body)

        self.assertEqual(response.status_code, 400)
        self.assertNotIn(blob[:100], body["message"])
        self.assertLess(len(response.body), 1_000)
        self.assertIn("serialized_named_tensors", body["message"])
        self.assertIn("Field required", body["message"])


if __name__ == "__main__":
    unittest.main()
