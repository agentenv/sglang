import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    dense_activation_cast_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

_SERVER_ARGS = "sglang.srt.true_on_policy.config._get_global_server_args"


def _server_args(on_policy: bool) -> SimpleNamespace:
    return SimpleNamespace(
        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1 if on_policy else None,
        tp_size=1,
    )


class TestDenseActivationCastDtype(unittest.TestCase):
    def _cast(self, on_policy, activation, weight):
        with mock.patch(_SERVER_ARGS, return_value=_server_args(on_policy)):
            return dense_activation_cast_dtype(activation, weight)

    def test_matching_bf16_is_not_cast(self):
        self.assertIsNone(self._cast(False, torch.bfloat16, torch.bfloat16))

    def test_fp32_activation_is_cast_to_bf16_weight(self):
        self.assertEqual(self._cast(False, torch.float32, torch.bfloat16), torch.bfloat16)

    def test_on_policy_forces_cast_to_weight_dtype(self):
        self.assertEqual(self._cast(True, torch.bfloat16, torch.bfloat16), torch.bfloat16)

    def test_fp8_weight_never_casts_activation(self):
        for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
            for activation in (torch.bfloat16, torch.float32):
                for on_policy in (False, True):
                    with self.subTest(fp8=fp8, activation=activation, on_policy=on_policy):
                        self.assertIsNone(self._cast(on_policy, activation, fp8))


if __name__ == "__main__":
    unittest.main()
