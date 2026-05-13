# AI Ping Pong

A browser Pong game powered by a DQN agent trained offline in Python and exported to TensorFlow.js for inference.

## Features

* **Difficulty Options:** Choose between Easy (5k episodes), Medium (8k episodes), and Impossible (Mathematically perfect tracking logic).
* **Serving System:** Proper table tennis serve rules (2 serves each, alternate every point at 10-10). Allows the player to aim their serve.
* **Keyboard Controls:** Use `ArrowUp` and `ArrowDown` to move your paddle, `Space` to launch a serve, and `K` to ready the AI's serve.
* **Cross-Platform Training:** Automatic device selection for PyTorch (`cuda` for Windows/Linux GPUs, `mps` for Mac Apple Silicon, falling back to `cpu`).

## Project Layout

```text
.
├── index.html              # Browser entrypoint
├── app/                    # Canvas game and TF.js inference code
├── assets/                 # CSS and static presentation assets
├── model/                  # Exported TensorFlow.js models (easy, medium, impossible)
├── training/               # Python DQN training and export pipeline
└── vercel.json             # Static hosting config
```

## Run The Game

Start a local web server from the project root:

```bash
python -m http.server 8080
```

Open your browser to:

```text
http://localhost:8080
```

## Train & Export

Install Python dependencies:

```bash
pip install -r training/requirements.txt
```

Train a new model from scratch:

```bash
python training/train.py
```

Export a checkpoint for the browser:
We use a script to export the different models into the specific difficulty directories.

```bash
python training/export_all.py
```

*(If you only want to export a single checkpoint, you can use `python training/export.py <checkpoint_path>` and manually move it into the appropriate difficulty folder).*

More training details are in [training/README.md](training/README.md).
