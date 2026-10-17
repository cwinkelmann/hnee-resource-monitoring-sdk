"""The examples are part of the contract: they must stay importable and honest."""
from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_gpu_imports_without_torch_installed():
    """Importing must not need CUDA or torch: the import happens inside main()."""
    module = _load("train_gpu")
    assert callable(module.main)
    src = (EXAMPLES / "train_gpu.py").read_text()
    head = src.split("def ", 1)[0]
    assert "import torch" not in head


def test_train_gpu_parses_its_arguments():
    module = _load("train_gpu")
    args = module.build_parser().parse_args(["--memory", "19.5", "--hours", "2"])
    assert args.memory == 19.5
    assert args.hours == 2.0
    assert args.out


def test_train_gpu_uses_the_assigned_slot_contract():
    src = (EXAMPLES / "train_gpu.py").read_text()
    for needed in ("hrm.reserve(", "slot.check()", "notice_seconds", "auto_advance",
                   "Preempted", "hrm.capacity("):
        assert needed in src, needed
    assert "hrm.book(" not in src        # the old day-based verb is not this path


def test_train_gpu_checkpoints_and_resumes():
    src = (EXAMPLES / "train_gpu.py").read_text()
    assert "torch.save" in src and "torch.load" in src
