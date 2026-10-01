"""Unit tests for Qwen4-Exp (Qwen3.8-Next) LoRA hooks.

`Qwen4ExpForConditionalGeneration` must expose `supported_lora_modules`,
`get_hidden_dim` and `should_apply_lora` like `Qwen3_5ForConditionalGeneration`;
otherwise the LoRA memory pool falls back to `get_default_hidden_dim`, which has
no branch for the GDN projections (`in_proj_qkvz`, `in_proj_ba`, `out_proj`).

Hermetic (CPU-only): the model classes are instantiated via `__new__` with just a
shrunk text config, no weights, CUDA or distributed groups.

Usage:
    python -m pytest test/registered/unit/lora/test_qwen4_exp_lora_dims_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
from sglang.srt.lora.utils import (
    _KNOWN_LORA_TARGET_MODULES,
    get_normalized_target_modules,
)
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration, Qwen4ExpModel

HIDDEN, HEAD_DIM, HEADS, KV_HEADS = 32, 8, 4, 2
LIN_K_HEADS, LIN_V_HEADS, LIN_HEAD_DIM = 2, 4, 8
MOE_INTER, SHARED_INTER = 16, 24


def _text_config():
    return Qwen4ExpTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=LIN_K_HEADS,
        linear_num_value_heads=LIN_V_HEADS,
        linear_key_head_dim=LIN_HEAD_DIM,
        linear_value_head_dim=LIN_HEAD_DIM,
        num_experts=8,
        moe_intermediate_size=MOE_INTER,
        shared_expert_intermediate_size=SHARED_INTER,
        ple_layer_ids=[2],
    )


def _make_fake_model():
    text = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(text)
    text.config = _text_config()
    # The model sets this at build time (qwen4_exp.py); the q projection carries the output gate.
    text.config.attn_output_gate = True
    model = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model = text
    return model


class TestQwen4ExpLoRAHooks(unittest.TestCase):
    def setUp(self):
        self.model = _make_fake_model()

    def test_supported_modules_match_qwen3_5(self):
        self.assertEqual(
            Qwen4ExpForConditionalGeneration.supported_lora_modules,
            Qwen3_5ForCausalLM.supported_lora_modules,
        )
        for name in (
            "in_proj_qkvz",
            "in_proj_ba",
            "out_proj",
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
        ):
            self.assertIn(name, Qwen4ExpForConditionalGeneration.supported_lora_modules)
            self.assertIn(name, _KNOWN_LORA_TARGET_MODULES)

    def test_attention_dims(self):
        self.assertEqual(
            self.model.get_hidden_dim("qkv_proj", 3),
            (HIDDEN, HEAD_DIM * (2 * HEADS + 2 * KV_HEADS)),
        )
        self.assertEqual(
            self.model.get_hidden_dim("o_proj", 3), (HEADS * HEAD_DIM, HIDDEN)
        )

    def test_gdn_dims(self):
        key_dim = LIN_K_HEADS * LIN_HEAD_DIM
        value_dim = LIN_V_HEADS * LIN_HEAD_DIM
        self.assertEqual(
            self.model.get_hidden_dim("in_proj_qkvz", 0),
            (HIDDEN, 2 * key_dim + 2 * value_dim),
        )
        self.assertEqual(
            self.model.get_hidden_dim("in_proj_ba", 0), (HIDDEN, 2 * LIN_V_HEADS)
        )
        self.assertEqual(self.model.get_hidden_dim("out_proj", 0), (value_dim, HIDDEN))

    def test_mlp_dims_use_shared_and_routed_intermediate_sizes(self):
        # qwen4_exp_text is a Qwen3.5-MoE text type: dense buffers size the shared expert.
        self.assertEqual(
            self.model.get_hidden_dim("gate_up_proj", 0), (HIDDEN, 2 * SHARED_INTER)
        )
        self.assertEqual(
            self.model.get_hidden_dim("down_proj", 0), (SHARED_INTER, HIDDEN)
        )
        self.assertEqual(
            self.model.get_hidden_dim("gate_up_proj_moe", 0), (HIDDEN, 2 * MOE_INTER)
        )
        self.assertEqual(
            self.model.get_hidden_dim("down_proj_moe", 0), (MOE_INTER, HIDDEN)
        )

    def test_should_apply_lora_only_to_language_layers(self):
        self.assertTrue(
            self.model.should_apply_lora("model.layers.0.linear_attn.in_proj_qkvz")
        )
        self.assertTrue(self.model.should_apply_lora("model.layers.3.mlp.experts"))
        self.assertFalse(self.model.should_apply_lora("visual.blocks.0.attn.qkv"))
        self.assertFalse(self.model.should_apply_lora("lm_head"))

    def test_trainer_target_leaves_normalize_to_pool_modules(self):
        """The Miles qwen4_exp adapter targets (per-expert layout) normalise to known modules."""
        miles_targets = [
            "model.language_model.layers.*.linear_attn.in_proj_qkv",
            "model.language_model.layers.*.linear_attn.in_proj_z",
            "model.language_model.layers.*.linear_attn.in_proj_b",
            "model.language_model.layers.*.linear_attn.in_proj_a",
            "model.language_model.layers.*.linear_attn.out_proj",
            "model.language_model.layers.*.self_attn.q_proj",
            "model.language_model.layers.*.self_attn.k_proj",
            "model.language_model.layers.*.self_attn.v_proj",
            "model.language_model.layers.*.self_attn.o_proj",
            "model.language_model.layers.*.mlp.experts.gate_up_proj",
            "model.language_model.layers.*.mlp.experts.down_proj",
            "model.language_model.layers.*.mlp.shared_expert.gate_proj",
            "model.language_model.layers.*.mlp.shared_expert.up_proj",
            "model.language_model.layers.*.mlp.shared_expert.down_proj",
        ]
        normalized = get_normalized_target_modules(
            target.rsplit(".", 1)[-1] for target in miles_targets
        )
        self.assertEqual(
            normalized,
            {
                "in_proj_qkvz",
                "in_proj_ba",
                "out_proj",
                "qkv_proj",
                "o_proj",
                "gate_up_proj",
                "down_proj",
            },
        )
        self.assertFalse(normalized - _KNOWN_LORA_TARGET_MODULES)


if __name__ == "__main__":
    unittest.main()
