"""In-place reload of compressed-tensors W8A8 FP8 linear layers -- CPU-only.

``process_weights_after_loading`` stores TENSOR / CHANNEL weights transposed
as (K, N) Parameters without weight loaders, so ``update_weights_from_disk``
failed with "Attempted to load weight (N, K) into parameter (K, N)". The
``restore_weights_before_loading`` hook puts the (N, K) loader-facing
parameters back; the next post-process refills the kernel tensors in place.
TENSOR strategy requantizes with a CUDA kernel, so only CHANNEL runs here.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.model_loader.loader import postprocess_weight, restore_weight
from sglang.test.test_utils import CustomTestCase

N, K = 6, 4


def _loader(param, loaded_weight):
    assert param.shape == loaded_weight.shape, (
        f"Attempted to load weight {tuple(loaded_weight.shape)} "
        f"into parameter {tuple(param.shape)}"
    )
    param.data.copy_(loaded_weight)


class _Linear(torch.nn.Module):
    pass


def _model(strategy):
    scheme = CompressedTensorsW8A8Fp8(
        QuantizationArgs(num_bits=8, type="float", symmetric=True, strategy=strategy),
        is_static_input_scheme=False,
    )
    layer = _Linear()
    layer.scheme = scheme
    scheme.create_weights(
        layer,
        input_size_per_partition=K,
        output_partition_sizes=[N],
        input_size=K,
        output_size=N,
        params_dtype=torch.bfloat16,
        weight_loader=_loader,
    )
    layer.quant_method = CompressedTensorsLinearMethod.__new__(
        CompressedTensorsLinearMethod
    )
    model = torch.nn.Module()
    model.proj = layer
    return model


def _checkpoint(seed, strategy):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(N, K, generator=g).to(torch.float8_e4m3fn)
    rows = N if strategy == QuantizationStrategy.CHANNEL else 1
    scale = torch.rand(rows, generator=g) + 0.5
    if strategy == QuantizationStrategy.CHANNEL:
        scale = scale.view(N, 1)
    return weight, scale


def _load(model, weight, scale):
    for name, value in (("weight", weight), ("weight_scale", scale)):
        param = getattr(model.proj, name)
        param.weight_loader(param, value)


class TestCompressedTensorsFp8Reload(CustomTestCase):
    def _roundtrip(self, strategy):
        device = torch.device("cpu")
        model = _model(strategy)
        _load(model, *_checkpoint(0, strategy))
        postprocess_weight(model, device)
        layer = model.proj
        self.assertEqual(tuple(layer.weight.shape), (K, N))
        kernel = {n: getattr(layer, n) for n in ("weight", "weight_scale")}
        ptrs = {n: p.data_ptr() for n, p in kernel.items()}

        # Without the restore hook the reload hits the transposed parameter.
        with self.assertRaises(AttributeError):
            layer.weight.weight_loader

        restore_weight(model, device)
        self.assertEqual(tuple(layer.weight.shape), (N, K))
        weight, scale = _checkpoint(1, strategy)
        _load(model, weight, scale)
        postprocess_weight(model, device)

        for name in ("weight", "weight_scale"):
            self.assertIs(getattr(layer, name), kernel[name])
            self.assertEqual(getattr(layer, name).data_ptr(), ptrs[name])
        self.assertTrue(
            torch.equal(layer.weight.float(), weight.t().float()),
            "kernel weight must be the reloaded checkpoint, transposed",
        )
        expected_scale = (
            scale if strategy == QuantizationStrategy.CHANNEL else scale.max()
        )
        self.assertTrue(torch.equal(layer.weight_scale.data, expected_scale))

    def test_channel_reload_in_place(self):
        self._roundtrip(QuantizationStrategy.CHANNEL)

    def test_restore_is_noop_before_first_postprocess(self):
        model = _model(QuantizationStrategy.CHANNEL)
        weight = model.proj.weight
        restore_weight(model, torch.device("cpu"))
        self.assertIs(model.proj.weight, weight)


if __name__ == "__main__":
    unittest.main()
