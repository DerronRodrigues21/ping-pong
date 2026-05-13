# Training

The agent is trained using a Deep Q-Network (DQN) implementation in PyTorch. The device selection logic automatically detects and utilizes the best available hardware accelerator: CUDA on Windows/Linux, MPS on Apple Silicon (Mac), and falling back to CPU when none are available.

## Train from scratch

```bash
python training/train.py
```

## Resume from a checkpoint

```bash
python training/train.py --resume training/checkpoints/ep_008000.pt
```

The trainer infers the next episode from checkpoint metadata. Older checkpoints without metadata still work when they use the `ep_008000.pt` filename pattern.
Replay memory is not saved, so resumed runs collect fresh experience before training updates restart.

## Resume while increasing exploration

```bash
python training/train.py --resume training/checkpoints/ep_008000.pt --epsilon 0.2
```

## Faster resumed training

```bash
python training/train.py --resume training/checkpoints/ep_008000.pt --epsilon 0.2 --no-export
```

By default, training uses 3-point games and a 2,000-frame episode cap. This keeps episodes short while still using the same state/action space and rewards.
Use `--win-score 11 --max-steps 10000` if you want the old full-match setup.

## Exporting for the browser game

The browser game supports dynamically loading different difficulty models. You can use our batch export script to populate all three difficulty options (`easy`, `medium`, `impossible`):

```bash
python training/export_all.py
```

If you train a new model and want to export it as a single file, you can run the standard export script:

```bash
python training/export.py training/checkpoints/ep_005000.pt
```

*Note: The browser now expects models to be placed in `model/<difficulty>/model.json` rather than the root `model/` folder.*
