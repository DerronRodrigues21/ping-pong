"""
Export trained PyTorch DQN weights → Keras SavedModel → TensorFlow.js.

Architecture (must match browser app/agent.js exactly):
  Input(6) → Dense(256, relu) → Dense(256, relu) → Dense(128, relu) → Dense(3, linear)
Output shape: (None, 3)
"""

import os
import sys
import numpy as np
import torch

# Add parent dir so we can import agent
sys.path.insert(0, os.path.dirname(__file__))
from agent import QNetwork

# Paths
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_DIR = os.path.join(TRAINING_DIR, "saved_model")
TFJS_OUTPUT_DIR = os.path.join(TRAINING_DIR, "..", "model")


def export_to_tfjs(checkpoint_path=None):
    """
    Full pipeline: PyTorch checkpoint → Keras → TF.js model.json + weights.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(TRAINING_DIR, "checkpoints", "final.pt")

    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"[1/4] Loading PyTorch checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    pytorch_model = QNetwork()
    pytorch_model.load_state_dict(ckpt["policy_state_dict"])
    pytorch_model.eval()

    print("[2/4] Building identical Keras model...")
    keras_model = _build_keras_model()

    print("[3/4] Copying weights PyTorch → Keras...")
    _copy_weights(pytorch_model, keras_model)
    _verify_outputs(pytorch_model, keras_model)

    print("[4/4] Saving TensorFlow.js LayersModel...")
    os.makedirs(TFJS_OUTPUT_DIR, exist_ok=True)
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(keras_model, TFJS_OUTPUT_DIR)

    model_json = os.path.join(TFJS_OUTPUT_DIR, "model.json")
    if os.path.exists(model_json):
        print(f"\n✓ Export complete! model.json at: {model_json}")
    else:
        print("✗ Export failed — model.json not found")
        sys.exit(1)


def _build_keras_model():
    """Build a Keras model with the exact same architecture."""
    import tf_keras as keras

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(6,)),
        keras.layers.Dense(256, activation="relu", name="dense_0"),
        keras.layers.Dense(256, activation="relu", name="dense_1"),
        keras.layers.Dense(128, activation="relu", name="dense_2"),
        keras.layers.Dense(3, activation=None, name="output"),
    ])
    model.build((None, 6))
    return model


def _copy_weights(pytorch_model, keras_model):
    """
    Copy weights from PyTorch nn.Sequential to Keras Sequential layer by layer.
    PyTorch Linear stores (weight, bias); Keras Dense expects (kernel, bias).
    PyTorch weight shape is (out, in); Keras kernel shape is (in, out) → transpose.
    """
    # PyTorch layers: net.0 (Linear), net.2 (Linear), net.4 (Linear), net.6 (Linear)
    # Indices in nn.Sequential: 0, 2, 4, 6 (odd indices are ReLU)
    pt_layers = [pytorch_model.net[i] for i in (0, 2, 4, 6)]
    keras_layers = [l for l in keras_model.layers if hasattr(l, "kernel")]

    assert len(pt_layers) == len(keras_layers), (
        f"Layer count mismatch: PyTorch={len(pt_layers)}, Keras={len(keras_layers)}"
    )

    for pt_l, k_l in zip(pt_layers, keras_layers):
        w = pt_l.weight.detach().numpy().T  # (out, in) → (in, out)
        b = pt_l.bias.detach().numpy()
        k_l.set_weights([w, b])


def _verify_outputs(pytorch_model, keras_model):
    """Verify both models produce identical outputs for random inputs."""
    import tensorflow as tf

    test_input = np.random.rand(5, 6).astype(np.float32)

    with torch.no_grad():
        pt_out = pytorch_model(torch.tensor(test_input)).numpy()

    k_out = keras_model.predict(test_input, verbose=0)

    max_diff = np.max(np.abs(pt_out - k_out))
    print(f"  → Max weight-copy error: {max_diff:.2e}")
    if max_diff > 1e-5:
        print("  ⚠ Warning: outputs differ significantly!")
    else:
        print("  ✓ Outputs match within tolerance.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    export_to_tfjs(path)
