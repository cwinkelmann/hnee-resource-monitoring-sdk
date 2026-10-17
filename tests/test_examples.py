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


# --- fixes from the whole-branch review ---------------------------------------------
def test_train_gpu_puts_the_work_on_the_booked_card():
    """CUDA_VISIBLE_DEVICES set after `import torch` does nothing -- the driver is
    already initialised and the device list cached -- so the job would train on
    physical GPU 0 whatever it booked. On a shared box that means running on someone
    else's card while the booking says otherwise."""
    src = (EXAMPLES / "train_gpu.py").read_text()
    assert 'os.environ["CUDA_VISIBLE_DEVICES"]' not in src
    assert "set_device(slot.gpu)" in src
    assert 'device="cuda:%d" % ' in src or "cuda:%d" in src
    assert 'device="cuda"' not in src          # never the implicit current device


def test_train_gpu_checkpoint_carries_the_model_shape():
    """Without --memory the layer count comes from live capacity, so a resume would
    load a checkpoint built for a different model and raise a size mismatch."""
    src = (EXAMPLES / "train_gpu.py").read_text()
    assert '"layers"' in src
    assert "layers_for" in src


def test_train_gpu_handles_a_refused_reservation():
    """reserve() can still 409: max_pending_per_user, or nothing fitting inside the
    14-day horizon. It sat outside the try, so the example died on a traceback."""
    src = (EXAMPLES / "train_gpu.py").read_text()
    body = src.split("def main(", 1)[1]
    assert "hrm.Conflict" in body


# --- the demo scripts and the walkthrough -------------------------------------------
DEMO = EXAMPLES / "demo"
DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_the_demo_scripts_are_valid_shell():
    import subprocess
    for script in sorted(DEMO.glob("*.sh")):
        done = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert done.returncode == 0, "%s: %s" % (script.name, done.stderr)


def test_every_demo_script_cleans_up_after_itself():
    """They book real slots on a real box. A demo that leaves bookings behind takes a
    card away from whoever comes next."""
    for script in sorted(DEMO.glob("[0-9]*.sh")):
        src = script.read_text()
        assert "trap " in src and "release_mine" in src, script.name
        assert "EXIT INT TERM" in src, script.name


def test_the_demos_never_signal_anything_but_their_own_child():
    for script in sorted(DEMO.glob("*.sh")):
        src = script.read_text()
        for line in src.splitlines():
            if "pkill" in line and not line.strip().startswith("#"):
                raise AssertionError("%s uses pkill: %s" % (script.name, line.strip()))
            if "kill " in line and not line.strip().startswith("#"):
                assert "JOB_PID" in line, "%s: %s" % (script.name, line.strip())


def test_the_walkthrough_links_images_that_exist():
    import re
    text = (DOCS / "walkthrough.md").read_text()
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert images, "the walkthrough has no screenshots"
    for rel in images:
        assert (DOCS / rel).exists(), rel


def test_the_walkthrough_points_at_scripts_that_exist():
    import re
    text = (DOCS / "walkthrough.md").read_text()
    for rel in re.findall(r"\]\((\.\./[^)]+)\)", text):
        assert (DOCS / rel).resolve().exists(), rel
