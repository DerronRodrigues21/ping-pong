/* ============================================================
   AIAgent — Inference-only controller for the AI paddle.

   Architecture:
     • Loads a pre-trained TensorFlow.js model (model.json + weights)
     • Runs forward-pass inference each frame to pick an action
     • Falls back to a deterministic heuristic when no model is found

   This module contains ZERO training logic. The model is trained
   offline in Python and exported to TF.js format.
   ============================================================ */

/**
 * @typedef {Object} GameState
 * @property {number} ballX       – normalized ball X  [0, 1]
 * @property {number} ballY       – normalized ball Y  [0, 1]
 * @property {number} ballVX      – normalized ball velocity X [-1, 1]
 * @property {number} ballVY      – normalized ball velocity Y [-1, 1]
 * @property {number} paddleY     – normalized AI paddle Y [0, 1]
 * @property {number} opponentY   – normalized player paddle Y [0, 1]
 */

/** Actions the agent can take. */
export const ACTION = Object.freeze({
  UP:   -1,
  STAY:  0,
  DOWN:  1,
});

/** Number of discrete actions (matches the model's output layer). */
export const NUM_ACTIONS = 3;

const PADDLE_HEIGHT_RATIO = 100 / 500;

export class AIAgent {
  constructor() {
    /** @type {any|null} TF.js LayersModel, null until loaded. */
    this._model = null;

    /** Whether the model loaded successfully. */
    this.modelLoaded = false;

    /** Human-readable status for the UI. */
    this.status = 'Initializing…';
  }

  /* ----------------------------------------------------------
     Model Loading
     ---------------------------------------------------------- */

  /**
   * Attempt to load the pre-trained model.
   * Gracefully falls back to heuristic mode if the model is missing.
   */
  async init(difficulty = 'medium') {
    // Check if TensorFlow.js is available
    if (typeof tf === 'undefined') {
      this.status = 'Heuristic — TF.js not loaded';
      console.warn('[AIAgent] TensorFlow.js not found. Running heuristic fallback.');
      return;
    }

    try {
      const modelPath = `./model/${difficulty}/model.json`;
      this.status = `Loading ${difficulty} model…`;
      this._model = await tf.loadLayersModel(modelPath);
      this.modelLoaded = true;
      this.status = 'DQN Model Active';
      console.info('[AIAgent] Pre-trained model loaded successfully.');

      // Warm up the model with a dummy inference to compile shaders / kernels
      const dummy = tf.zeros([1, 6]);
      this._model.predict(dummy).dispose();
      dummy.dispose();
    } catch (err) {
      this._model = null;
      this.modelLoaded = false;
      this.status = 'Heuristic — No model found';
      console.warn('[AIAgent] Could not load model, using heuristic fallback:', err.message);
    }
  }

  /* ----------------------------------------------------------
     State Normalization
     ---------------------------------------------------------- */

  /**
   * Build a normalized state vector from raw game values.
   * All values are scaled to roughly [0, 1] or [-1, 1] for the network.
   *
   * @param {Object} raw
   * @param {number} raw.ballX       – ball X position
   * @param {number} raw.ballY       – ball Y position
   * @param {number} raw.ballVX      – ball X velocity
   * @param {number} raw.ballVY      – ball Y velocity
   * @param {number} raw.aiPaddleY   – AI paddle top Y
   * @param {number} raw.playerPaddleY – player paddle top Y
   * @param {{ width: number, height: number }} raw.bounds – canvas dimensions
   * @returns {Float32Array} 6-element normalized state vector
   */
  static buildState(raw) {
    const { ballX, ballY, ballVX, ballVY, aiPaddleY, playerPaddleY, bounds } = raw;
    return new Float32Array([
      ballX / bounds.width,                         // [0, 1]
      ballY / bounds.height,                        // [0, 1]
      ballVX / 15,                                  // ~[-1, 1] (training BALL_MAX_SPEED)
      ballVY / 15,                                  // ~[-1, 1]
      aiPaddleY / bounds.height,                    // [0, 1]
      playerPaddleY / bounds.height,                // [0, 1]
    ]);
  }

  /* ----------------------------------------------------------
     Action Selection
     ---------------------------------------------------------- */

  /**
   * Choose an action given the current game state.
   * Uses the neural network if loaded, otherwise the heuristic.
   *
   * @param {Float32Array} state – 6-element normalized state vector
   * @returns {number} action from ACTION enum (-1, 0, +1)
   */
  selectAction(state) {
    if (this._model) {
      return this._inferAction(state);
    }
    return this._heuristicAction(state);
  }

  /**
   * Run a forward pass through the pre-trained DQN.
   * @param {Float32Array} state
   * @returns {number} action index mapped to ACTION enum
   */
  _inferAction(state) {
    // tf.tidy auto-disposes intermediate tensors to prevent leaks
    const actionIndex = tf.tidy(() => {
      const input = tf.tensor2d(state, [1, 6]);
      const qValues = /** @type {tf.Tensor} */ (this._model.predict(input));
      return qValues.argMax(-1).dataSync()[0];
    });

    // Training action map: 0 = UP, 1 = DOWN, 2 = STAY
    if (actionIndex === 0) return ACTION.UP;
    if (actionIndex === 1) return ACTION.DOWN;
    return ACTION.STAY;
  }

  /**
   * Simple deterministic fallback: track the ball's Y position.
   * Good enough to rally but easily beatable by a human.
   *
   * @param {Float32Array} state – normalized state [ballX, ballY, vx, vy, aiY, playerY]
   * @returns {number} action from ACTION enum
   */
  _heuristicAction(state) {
    const ballY = state[1];      // normalized ball Y
    const aiPaddleY = state[4] + PADDLE_HEIGHT_RATIO / 2;  // normalized AI paddle center Y

    const deadzone = 0.03; // ~15px at 500px height — prevents jitter
    const diff = ballY - aiPaddleY;

    if (diff > deadzone)  return ACTION.DOWN;
    if (diff < -deadzone) return ACTION.UP;
    return ACTION.STAY;
  }

  /* ----------------------------------------------------------
     Cleanup
     ---------------------------------------------------------- */

  /** Dispose TF.js model to free GPU/WebGL memory. */
  dispose() {
    if (this._model) {
      this._model.dispose();
      this._model = null;
      this.modelLoaded = false;
      this.status = 'Disposed';
    }
  }
}
