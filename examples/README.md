# Examples

| file | what it shows |
|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | the calendar verbs: usage, bookings, book whole days, cancel |
| [`notebook_session.ipynb`](notebook_session.ipynb) | hold a GPU while you work in a notebook, then give it back |
| [`train_gpu.py`](train_gpu.py) | a long training run that takes its turn: reserve, wait, train, checkpoint, release |

`train_gpu.py` is the one to copy for batch work. It books `lendable` with
`notice_seconds=0`, so a colleague who needs the card gets it at once and the job resumes
from its last checkpoint later — which is how thousands of one-hour jobs share a box
without anyone queueing by hand:

```bash
python examples/train_gpu.py --memory 19.5 --hours 1 --out runs/demo
```

It needs PyTorch and a CUDA GPU. Everything else here needs only the SDK.

Both notebooks and the script read `HRM_URL` and `HRM_USER` from the environment, so the
same code runs against carrot, olive or the ThinkPad:

```bash
export HRM_URL=http://10.188.1.1:8765
export HRM_USER=cwinkelmann
```
