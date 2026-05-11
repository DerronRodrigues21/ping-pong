# AI Ping Pong

A browser Pong game powered by a DQN agent trained offline in Python and exported
to TensorFlow.js for inference.

## Project Layout

```text
.
├── index.html              # Browser entrypoint
├── app/                    # Canvas game and TF.js inference code
├── assets/                 # CSS and static presentation assets
├── model/                  # Exported TensorFlow.js model used by the browser
├── training/               # Python DQN training and export pipeline
└── vercel.json             # Static hosting config
```

## Run The Game

```bash
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000
```

The status bar should show `DQN Model Active` when `model/model.json` loads.

## Train

Install Python dependencies:

```bash
pip3 install -r training/requirements.txt
```

Resume from the included checkpoint:

```bash
python3 training/train.py --resume training/checkpoints/ep_005000.pt --epsilon 0.2
```

Export a checkpoint for the browser:

```bash
python3 training/export.py training/checkpoints/ep_005000.pt
```

More training details are in [training/README.md](training/README.md).
