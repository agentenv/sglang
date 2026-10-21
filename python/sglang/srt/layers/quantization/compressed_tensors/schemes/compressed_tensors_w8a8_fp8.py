# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Parameter

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    apply_fp8_ptpc_linear,
    deepgemm_w8a8_block_fp8_linear_with_fallback,
    dispatch_w8a8_block_fp8_linear,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_block_scale_ue8m0_for_deepgemm,
    validate_fp8_block_shape,
)
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.utils import get_bool_env_var, is_hip

__all__ = ["CompressedTensorsW8A8Fp8"]

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight


strategy_to_parameter_type = {
    QuantizationStrategy.BLOCK: BlockQuantScaleParameter,
    QuantizationStrategy.CHANNEL: ChannelQuantScaleParameter,
    QuantizationStrategy.TENSOR: PerTensorScaleParameter,
}


class CompressedTensorsW8A8Fp8(CompressedTensorsLinearScheme):
    def __init__(self, weight_quant: QuantizationArgs, is_static_input_scheme: bool):
        self.weight_quant = weight_quant
        self.strategy = self.weight_quant.strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = self.weight_quant.block_structure
        if self.weight_block_size is not None:
            self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = None
        layer.orig_dtype = params_dtype

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            # Validate block quantization shapes
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            weight_scale[:] = torch.finfo(torch.float32).min
        elif self.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            weight_scale[:] = torch.finfo(torch.float32).min
        elif self.strategy == QuantizationStrategy.BLOCK:
            assert layer.weight_block_size is not None
            block_n, block_k = layer.weight_block_size[0], layer.weight_block_size[1]
            output_size_per_partition = sum(output_partition_sizes)
            weight_scale = BlockQuantScaleParameter(
                data=torch.empty(
                    (output_size_per_partition + block_n - 1) // block_n,
                    (input_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            weight_scale.format_ue8m0 = False
            weight_scale[:] = torch.finfo(torch.float32).min

        layer.register_parameter("weight_scale", weight_scale)
        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    # Kernel-layout parameters replace the loader-facing ones for these
    # strategies (the weight is stored transposed as (K, N) for the FP8 GEMM),
    # so an in-place reload must swap the originals back in first.
    _RELOADABLE_PARAMS = ("weight", "weight_scale")

    def _supports_inplace_reload(self) -> bool:
        return (
            self.strategy in (QuantizationStrategy.TENSOR, QuantizationStrategy.CHANNEL)
            and not _use_aiter
            and not is_fp8_fnuz()
        )

    def restore_weights_before_loading(self, layer) -> None:
        """Make ``layer`` loadable again after ``process_weights_after_loading``.

        Puts the original (N, K) loader-facing parameters back so a checkpoint
        reload (``update_weights_from_disk``) writes through their weight
        loaders. The kernel-layout parameters are kept aside so the next
        ``process_weights_after_loading`` refills them in place and their
        device addresses (captured by CUDA graphs) do not change.
        """
        loadable = getattr(layer, "_fp8_loadable_params", None)
        if loadable is None:
            return
        layer._fp8_kernel_params = {
            name: getattr(layer, name) for name in self._RELOADABLE_PARAMS
        }
        for name, param in loadable.items():
            setattr(layer, name, param)

    def _finish_inplace_reload(self, layer) -> None:
        kernel_params = layer.__dict__.pop("_fp8_kernel_params", None)
        if kernel_params is None:
            return
        for name, old in kernel_params.items():
            new = getattr(layer, name)
            if old.shape != new.shape or old.stride() != new.stride():
                raise RuntimeError(
                    f"FP8 reload changed the kernel layout of {name}: "
                    f"{tuple(old.shape)}/{old.stride()} -> "
                    f"{tuple(new.shape)}/{new.stride()}"
                )
            if old.data_ptr() != new.data_ptr():
                old.data.copy_(new.data)
            setattr(layer, name, old)

    def process_weights_after_loading(self, layer) -> None:
        if self._supports_inplace_reload():
            layer._fp8_loadable_params = {
                name: getattr(layer, name) for name in self._RELOADABLE_PARAMS
            }
        self._process_weights_after_loading(layer)
        self._finish_inplace_reload(layer)

    def _process_weights_after_loading(self, layer) -> None:
        if self.strategy == QuantizationStrategy.TENSOR:
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )

            if is_fp8_fnuz():
                input_scale = getattr(layer, "input_scale", None)

                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight, weight_scale=max_w_scale, input_scale=input_scale
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight = layer.weight

            if is_fp8_fnuz():
                input_scale = getattr(layer, "input_scale", None)

                weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=layer.weight_scale,
                    input_scale=input_scale,
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)
            else:
                weight_scale = layer.weight_scale.data

            if _use_aiter:
                # keep the weight as (N, K)
                layer.weight = Parameter(
                    shuffle_weight(weight, (16, 16)), requires_grad=False
                )
            else:
                layer.weight = Parameter(weight.t(), requires_grad=False)

            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        elif self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            if is_fp8_fnuz():
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight, weight_scale=layer.weight_scale
                )
                layer.weight = Parameter(weight.data, requires_grad=False)
                layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
                layer.weight_scale.format_ue8m0 = False
            else:
                layer.weight.requires_grad_(False)
                layer.weight_scale.requires_grad_(False)

            # On Blackwell, block-FP8 dispatches to DeepGEMM, which needs the
            # weight scales UE8M0-packed to match its UE8M0 activation scales.
            use_deepgemm_runner = (
                self.w8a8_block_fp8_linear
                is deepgemm_w8a8_block_fp8_linear_with_fallback
            )
            requant_block_scale_ue8m0_for_deepgemm(
                layer.weight,
                layer.weight_scale,
                self.weight_block_size,
                use_deepgemm_runner=use_deepgemm_runner,
                output_dtype=getattr(layer, "orig_dtype", None),
                weight_shape=layer.weight.shape,
            )

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
        else:
            layer.input_scale = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            # Pre-quantized activation from a fused RMSNorm+FP8 quant kernel:
            # x = (fp8_input, per_tensor_input_scale[, orig_dtype]).
            # apply_fp8_linear detects the fp8 dtype and skips re-quantizing;
            # orig_dtype (when present) sets the GEMM output dtype.
            qx, x_scale = x[0], x[1]
            out_dtype = x[2] if len(x) > 2 else None
            return apply_fp8_linear(
                input=qx,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=x_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
                pre_quant_output_dtype=out_dtype,
            )
        if self.weight_block_size is not None:
            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.weight_block_size,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
            )

        if _use_aiter and self.strategy == QuantizationStrategy.CHANNEL:
            return apply_fp8_ptpc_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
            )
        else:
            return apply_fp8_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
            )
