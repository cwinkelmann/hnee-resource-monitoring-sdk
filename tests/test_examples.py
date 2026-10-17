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


import json


def _notebook(name):
    return json.loads((EXAMPLES / (name + ".ipynb")).read_text())


def test_the_notebook_session_example_is_valid_and_uses_the_new_verbs():
    nb = _notebook("notebook_session")
    assert nb["nbformat"] >= 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")
    for needed in ("hrm.connect(", "hrm.capacity(", "hrm.reserve(", "slot.release()"):
        assert needed in code, needed
    assert "import hrm" in code


def test_the_notebook_ships_without_stored_output():
    nb = _notebook("notebook_session")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def test_every_example_is_listed_in_the_examples_readme():
    listing = (EXAMPLES / "README.md").read_text()
    for path in sorted(EXAMPLES.iterdir()):
        if path.suffix in (".py", ".ipynb"):
            assert path.name in listing, path.name
