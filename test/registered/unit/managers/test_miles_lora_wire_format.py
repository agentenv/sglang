"""Regression tests for the LoRA sync wire format upstream Miles sends.

Ported from agentenv/sglang@95d4d69 ("align tensor LoRA requests with Miles")
and realigned to radixark/miles@9e4260d. Miles no longer calls
``/load_lora_adapter_from_tensors``; it:

1. POSTs ``/register_lora_adapter`` with ``{lora_name, config_dict, pinned}``;
2. streams ``{lora_name}:{hf_key}``-prefixed tensors through the ordinary
   ``/update_weights_from_tensor`` endpoint as ``load_format="flattened_bucket"``
   with one base64 ``MultiprocessingSerializer`` payload per TP rank
   (``serialized_named_tensors[tp_rank]``), plus a ``selector`` field.

The original patch's intent — one serialized copy per TP rank, each rank
deserializing only its own — is what these tests pin down on that path.
"""

import unittest
from typing import Annotated

import torch
from fastapi import Body
from pydantic import TypeAdapter

from sglang.srt.managers.io_struct import (
    RegisterLoRAAdapterReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.common import normalize_serialized_named_tensor_payloads
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")

MILES_LORA_NAME = "miles_lora"  # miles.utils.lora.utils.LORA_ADAPTER_NAME


def _miles_rank_payload(rank: int) -> str:
    """Mirror miles cuda_ipc._send_to_colocated_engine for one rank/dtype."""
    named = [
        (
            f"{MILES_LORA_NAME}:model.layers.0.self_attn.q_proj.lora_A.weight",
            torch.full((2, 4), float(rank)),
        ),
        (
            f"{MILES_LORA_NAME}:model.layers.0.self_attn.q_proj.lora_B.weight",
            torch.full((4, 2), float(rank) + 0.5),
        ),
    ]
    bucket = FlattenedTensorBucket(named_tensors=named)
    return MultiprocessingSerializer.serialize(
        {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        },
        output_str=True,
    )


class TestMilesLoRAWireFormat(CustomTestCase):
    def test_register_lora_adapter_payload(self):
        payload = {
            "lora_name": MILES_LORA_NAME,
            "config_dict": {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
            "pinned": False,
        }
        req = TypeAdapter(
            Annotated[RegisterLoRAAdapterReqInput, Body()]
        ).validate_python(payload)
        self.assertEqual(req.lora_name, MILES_LORA_NAME)
        self.assertEqual(req.config_dict["r"], 8)
        self.assertIsNone(req.lora_path)  # stream-only adapter, not reloadable

    def test_update_weights_payload_is_per_rank(self):
        tp_size = 2
        payload = {
            "serialized_named_tensors": [_miles_rank_payload(r) for r in range(tp_size)],
            "load_format": "flattened_bucket",
            "flush_cache": False,
            "selector": "all",
        }
        req = TypeAdapter(
            Annotated[UpdateWeightsFromTensorReqInput, Body()]
        ).validate_python(payload)
        self.assertEqual(req.load_format, "flattened_bucket")
        self.assertEqual(len(req.serialized_named_tensors), tp_size)

        per_rank = normalize_serialized_named_tensor_payloads(
            req.serialized_named_tensors
        )
        for rank in range(tp_size):
            # Each rank deserializes only its own copy (tp_worker indexes by tp_rank).
            data = MultiprocessingSerializer.deserialize(per_rank[rank])
            tensors = dict(
                FlattenedTensorBucket(
                    flattened_tensor=data["flattened_tensor"],
                    metadata=data["metadata"],
                ).reconstruct_tensors()
            )
            self.assertTrue(
                all(n.startswith(f"{MILES_LORA_NAME}:") for n in tensors), tensors
            )
            a = tensors[
                f"{MILES_LORA_NAME}:model.layers.0.self_attn.q_proj.lora_A.weight"
            ]
            self.assertTrue(torch.equal(a, torch.full((2, 4), float(rank))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
