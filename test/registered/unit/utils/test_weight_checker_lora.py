"""CPU unit test for lora_pool_checksums (LoRA A/B read-back for the weight checker)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils.weight_checker import lora_pool_checksums


def _pool(slot_a_value: float = 1.0):
    a = [torch.zeros(2, 4, 8) for _ in range(2)]  # [num_loras, rank, dim] x 2 layers
    b = [torch.zeros(2, 8, 4) for _ in range(2)]
    for t in a:
        t[1].fill_(slot_a_value)
    for t in b:
        t[1].fill_(0.5)
    return SimpleNamespace(
        uid_to_buffer_id={None: 0, "uid1": 1}, A_buffer={"qkv_proj": a}, B_buffer={"qkv_proj": b}
    )


def _mgr(pool):
    return SimpleNamespace(
        memory_pool=pool, lora_refs={"uid1": SimpleNamespace(lora_name="policy")}
    )


class TestLoraPoolChecksums(unittest.TestCase):
    def test_keys_and_base_slot_skipped(self):
        out = lora_pool_checksums(_mgr(_pool()))
        self.assertEqual(
            sorted(out),
            sorted(
                f"lora:policy:qkv_proj:{i}:{t}" for i in (0, 1) for t in ("A", "B")
            ),
        )

    def test_perturbation_changes_only_that_tensor_kind(self):
        ref, bad = lora_pool_checksums(_mgr(_pool())), lora_pool_checksums(_mgr(_pool(1.001)))
        self.assertEqual(ref, lora_pool_checksums(_mgr(_pool())))
        self.assertNotEqual(ref["lora:policy:qkv_proj:0:A"], bad["lora:policy:qkv_proj:0:A"])
        self.assertEqual(ref["lora:policy:qkv_proj:0:B"], bad["lora:policy:qkv_proj:0:B"])

    def test_lora_off(self):
        self.assertEqual(lora_pool_checksums(None), {})
        self.assertEqual(lora_pool_checksums(SimpleNamespace(memory_pool=None)), {})


if __name__ == "__main__":
    unittest.main()
