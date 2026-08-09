import asyncio
import json
import unittest
from types import SimpleNamespace

from fastapi.exceptions import RequestValidationError

from sglang.srt.entrypoints.http_server import validation_exception_handler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class TestHttpServerValidation(unittest.TestCase):
    def test_tensor_lora_validation_does_not_reflect_payload(self):
        blob = "A" * 1_000_000
        exc = RequestValidationError(
            errors=[
                {
                    "type": "missing",
                    "loc": (
                        "body",
                        "function-after[build_struct(), "
                        "LoadLoRAAdapterFromTensorsReqInput]",
                        "serialized_tensors",
                    ),
                    "msg": "Field required",
                    "input": {
                        "lora_name": "miles_lora",
                        "serialized_named_tensors": [blob],
                    },
                }
            ]
        )
        request = SimpleNamespace(
            url=SimpleNamespace(path="/load_lora_adapter_from_tensors")
        )

        response = asyncio.run(validation_exception_handler(request, exc))
        body = json.loads(response.body)

        self.assertEqual(response.status_code, 400)
        self.assertNotIn(blob[:100], body["message"])
        self.assertNotIn("miles_lora", body["message"])
        self.assertLess(len(response.body), 1_000)
        self.assertIn("serialized_tensors", body["message"])
        self.assertIn("Field required", body["message"])


if __name__ == "__main__":
    unittest.main()
