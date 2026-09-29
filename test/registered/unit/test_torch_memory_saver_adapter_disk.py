from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.srt.utils import torch_memory_saver_adapter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def test_real_adapter_forwards_disk_backup(monkeypatch):
    region = Mock(return_value=nullcontext())
    monkeypatch.setattr(
        torch_memory_saver_adapter,
        "_memory_saver",
        SimpleNamespace(region=region),
        raising=False,  # torch_memory_saver is absent on CPU-only installs
    )

    adapter = torch_memory_saver_adapter._TorchMemorySaverAdapterReal()
    with adapter.region("weights", enable_disk_backup=True):
        pass

    region.assert_called_once_with(
        tag="weights",
        enable_cpu_backup=False,
        enable_disk_backup=True,
    )


def _load_main_weights(monkeypatch, *, cpu_backup, disk_backup, is_draft_worker):
    region = Mock(return_value=nullcontext())
    features = SimpleNamespace(
        enable_weights_cpu_backup=cpu_backup,
        enable_draft_weights_cpu_backup=True,
        enable_weights_disk_backup=disk_backup,
    )
    exec_ = SimpleNamespace(
        features=features,
        offload=SimpleNamespace(ple_offload_embedding=False),
    )
    monkeypatch.setattr(load_model_utils, "get_exec", lambda: exec_)
    monkeypatch.setattr(
        load_model_utils,
        "get_model",
        lambda: SimpleNamespace(weight_cache_mode="off"),
    )
    monkeypatch.setattr(load_model_utils, "monkey_patch_vllm_parallel_state", Mock())
    monkeypatch.setattr(
        load_model_utils,
        "get_model_loader",
        Mock(side_effect=RuntimeError("stop after entering the region")),
    )
    model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[]))

    with pytest.raises(RuntimeError, match="stop after entering the region"):
        load_model_utils.load_model_with_memory_saver(
            model_config=model_config,
            load_config=Mock(),
            device="cuda",
            gpu_id=0,
            memory_saver_adapter=SimpleNamespace(region=region),
            is_draft_worker=is_draft_worker,
        )
    return region


def test_model_load_marks_main_weights_for_disk_backup(monkeypatch):
    region = _load_main_weights(
        monkeypatch, cpu_backup=False, disk_backup=True, is_draft_worker=False
    )
    region.assert_called_once_with(
        "weights",
        enable_cpu_backup=False,
        enable_disk_backup=True,
    )


def test_draft_weights_never_use_disk_backup(monkeypatch):
    region = _load_main_weights(
        monkeypatch, cpu_backup=False, disk_backup=True, is_draft_worker=True
    )
    region.assert_called_once_with(
        "weights",
        enable_cpu_backup=True,
        enable_disk_backup=False,
    )


def test_cpu_and_disk_backup_are_mutually_exclusive(monkeypatch):
    with pytest.raises(AssertionError, match="mutually exclusive"):
        _load_main_weights(
            monkeypatch, cpu_backup=True, disk_backup=True, is_draft_worker=False
        )
