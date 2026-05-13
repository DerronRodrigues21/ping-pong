# Training

Train from scratch:

```bash
python3 training/train.py
```

Resume from a checkpoint:

```bash
python3 training/train.py --resume training/checkpoints/ep_005000.pt
```

The trainer infers the next episode from checkpoint metadata. Older checkpoints
without metadata still work when they use the `ep_005000.pt` filename pattern.
Replay memory is not saved, so resumed runs collect fresh experience before
training updates restart.

Resume while increasing exploration:

```bash
python3 training/train.py --resume training/checkpoints/ep_005000.pt --epsilon 0.2
```

Faster resumed training:

```bash
python3 training/train.py --resume training/checkpoints/ep_005000.pt --epsilon 0.2 --no-export
```

By default, training now uses 3-point games and a 2,000-frame episode cap. That
keeps episodes short while still using the same state/action space and rewards.
Use `--win-score 11 --max-steps 10000` if you want the old full-match setup.

Export a checkpoint for the browser game:

```bash
python3 training/export.py training/checkpoints/ep_005000.pt
```

The browser loads the exported TensorFlow.js model from:

```text
model/model.json
model/group1-shard1of1.bin
```
