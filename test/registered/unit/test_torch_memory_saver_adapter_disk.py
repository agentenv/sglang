from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.srt.utils import torch_memory_saver_adapter


def test_real_adapter_forwards_disk_backup(monkeypatch):
    region = Mock(return_value=nullcontext())
    monkeypatch.setattr(
        torch_memory_saver_adapter,
        "_memory_saver",
        SimpleNamespace(region=region),
    )

    adapter = torch_memory_saver_adapter._TorchMemorySaverAdapterReal()
    with adapter.region("weights", enable_disk_backup=True):
        pass

    region.assert_called_once_with(
        tag="weights",
        enable_cpu_backup=False,
        enable_disk_backup=True,
    )


def test_model_load_marks_main_weights_for_disk_backup(monkeypatch):
    region = Mock(return_value=nullcontext())
    adapter = SimpleNamespace(region=region)
    server_args = SimpleNamespace(
        enable_weights_cpu_backup=False,
        enable_draft_weights_cpu_backup=True,
        enable_weights_disk_backup=True,
        remote_instance_weight_loader_seed_instance_ip=None,
    )

    monkeypatch.setattr(load_model_utils, "monkey_patch_vllm_parallel_state", Mock())
    monkeypatch.setattr(
        load_model_utils,
        "get_model_loader",
        Mock(side_effect=RuntimeError("stop after entering the region")),
    )

    try:
        load_model_utils.load_model_with_memory_saver(
            server_args=server_args,
            model_config=Mock(),
            load_config=Mock(),
            device="cuda",
            gpu_id=0,
            memory_saver_adapter=adapter,
            is_draft_worker=False,
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after entering the region"
    else:
        raise AssertionError("expected the loader sentinel")

    region.assert_called_once_with(
        "weights",
        enable_cpu_backup=False,
        enable_disk_backup=True,
    )
