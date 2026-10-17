#!/usr/bin/env python3
"""Long-running GPU training that takes its turn on the shared box.

Reserve a slot, wait for it, train, checkpoint every epoch, and give the card
back. The booking never fails for lack of capacity: if the box is full you are
assigned a later window and this script waits for it, so you can start a hundred
of these and let the server decide the order.

    python examples/train_gpu.py --memory 19.5 --hours 1 --out runs/demo

Being preempted is a normal outcome, not a failure. The script books `lendable`
with `notice_seconds=0`, which says "an important booking may have this card at
once"; when that happens it writes a checkpoint and exits 0, and the same command
resumes from where it stopped.

Needs PyTorch (`pip install torch`) and a CUDA GPU; everything else is the hrm SDK.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import hrm

# A transformer block's parameters and activations scale with these; the defaults
# are sized so one block plus a batch fills roughly a GiB, and --memory picks how
# many blocks to stack.
D_MODEL = 1024
N_HEADS = 16
SEQ_LEN = 512
BATCH = 16
GIB_PER_LAYER = 0.55          # measured on an A6000 at these sizes, AMP on


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--memory", type=float, default=None,
                   help="GiB of VRAM to book (default: most of the card the server offers)")
    p.add_argument("--hours", type=float, default=1.0, help="how long to book for")
    p.add_argument("--gpu", type=int, default=None, help="pin to one GPU (default: any)")
    p.add_argument("--epochs", type=int, default=1000, help="epochs to run at most")
    p.add_argument("--steps", type=int, default=200, help="steps per epoch")
    p.add_argument("--out", type=Path, default=Path("runs/train_gpu"),
                   help="checkpoint directory")
    p.add_argument("--user", default=None, help="your account (default: HRM_USER or login)")
    p.add_argument("--url", default=None, help="booking server (default: HRM_URL)")
    p.add_argument("--important", action="store_true",
                   help="book as important: nobody can take this card from you")
    p.add_argument("--not-later-than", default=None,
                   help="give up if the earliest slot is after this ISO time")
    return p


def pick_memory(asked, gpu):
    """What to book: what was asked, or most of the biggest card the server has."""
    free = hrm.capacity(memory=1, hours=1, gpu=gpu)
    if asked is not None:
        if free.earliest is None:
            print("note: %.1f GiB does not fit anywhere in the next fortnight" % asked)
        return asked
    whole = max(free.free_gib, 1.0)
    memory = max(round(whole * 0.9, 1), 1.0)
    print("booking %.1f GiB of the %.1f GiB this box has free" % (memory, whole))
    return memory


def layers_for(memory_gib: float) -> int:
    """Stack enough blocks to use the booking. Honest about the arithmetic: this is
    a rule of thumb, and the point is to be busy, not to hit the number exactly."""
    return max(int(memory_gib / GIB_PER_LAYER), 2)


def make_scaler(torch):
    """AMP's scaler moved namespace in torch 2.4; accept both so the example runs
    on whatever is installed on the box."""
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def autocast(torch):
    try:
        return torch.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def build_model(torch, memory_gib: float):
    nn = torch.nn
    layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS,
                                       dim_feedforward=4 * D_MODEL,
                                       batch_first=True)
    model = nn.TransformerEncoder(layer, num_layers=layers_for(memory_gib))
    return nn.Sequential(model, nn.Linear(D_MODEL, D_MODEL))


def load_checkpoint(torch, path: Path, model, optimiser):
    """Resume if there is something to resume from; return the next epoch."""
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cuda")
    model.load_state_dict(state["model"])
    optimiser.load_state_dict(state["optimiser"])
    print("resuming from %s at epoch %d" % (path, state["epoch"] + 1))
    return state["epoch"] + 1


def save_checkpoint(torch, path: Path, model, optimiser, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                "epoch": epoch}, tmp)
    tmp.replace(path)          # atomic: a checkpoint is never half-written


def train_one_epoch(torch, model, optimiser, scaler, steps: int) -> float:
    model.train()
    total = 0.0
    for _ in range(steps):
        batch = torch.randn(BATCH, SEQ_LEN, D_MODEL, device="cuda")
        target = torch.randn(BATCH, SEQ_LEN, D_MODEL, device="cuda")
        optimiser.zero_grad(set_to_none=True)
        with autocast(torch):
            loss = torch.nn.functional.mse_loss(model(batch), target)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total += float(loss.detach())
    return total / max(steps, 1)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError:
        print("this example needs PyTorch: pip install torch", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("no CUDA GPU visible; run this on the box you booked", file=sys.stderr)
        return 1

    hrm.connect(user=args.user, url=args.url)
    memory = pick_memory(args.memory, args.gpu)

    slot = hrm.reserve(memory=memory, hours=args.hours, gpu=args.gpu,
                       important=args.important,
                       # lendable with no notice: a colleague who needs the card gets
                       # it at once, and this job resumes from its checkpoint later.
                       notice_seconds=None if args.important else 0,
                       # and if a card frees up before the assigned start, take it
                       auto_advance=True,
                       not_later_than=args.not_later_than,
                       note="train_gpu.py")
    if slot.assigned_later:
        print("the box is busy: assigned %s on GPU %d — waiting (it may move earlier)"
              % (slot.start.astimezone().strftime("%a %d %b %H:%M"), slot.gpu))
    else:
        print("starting now on GPU %d" % slot.gpu)

    checkpoint = args.out / "checkpoint.pt"
    try:
        with slot:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
            print("GPU %d is mine until %s"
                  % (slot.gpu, slot.end.astimezone().strftime("%H:%M")))
            model = build_model(torch, memory).cuda()
            optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = make_scaler(torch)
            first = load_checkpoint(torch, checkpoint, model, optimiser)
            held = torch.cuda.memory_reserved() / (1024 ** 3)
            print("%d transformer blocks, %.1f GiB reserved on the card"
                  % (layers_for(memory), held))

            for epoch in range(first, args.epochs):
                slot.check()        # the one safe place to stop: before the next epoch
                began = time.monotonic()
                loss = train_one_epoch(torch, model, optimiser, scaler, args.steps)
                save_checkpoint(torch, checkpoint, model, optimiser, epoch)
                print("epoch %d  loss %.4f  %.1fs" % (epoch, loss,
                                                      time.monotonic() - began),
                      flush=True)
                if slot.end <= slot.client.now():
                    print("the booked window is over; stopping at epoch %d" % epoch)
                    break
    except hrm.Preempted as taken:
        print("preempted: %s" % taken)
        print("checkpoint is at %s; run the same command to carry on" % checkpoint)
        return 0
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
