/* ============================================================
   AIAgent — Inference-only controller with Dynamic Loading
   ============================================================ */

export const ACTION = Object.freeze({ UP: -1, STAY: 0, DOWN: 1 });
const PADDLE_HEIGHT_RATIO = 100 / 500;

export class AIAgent {
  constructor() {
    this._model = null;
    this.modelLoaded = false;
    this.status = 'Initializing…';
  }

  /** 
   * Loads model based on difficulty. 
   * Note: 'hard' maps to your 'impossible' folder.
   */
  async init(difficulty = 'medium') {
    if (typeof tf === 'undefined') {
      this.status = 'Heuristic — TF.js not loaded';
      return;
    }
    try {
      const folder = difficulty === 'hard' ? 'impossible' : difficulty;
      this.status = `Loading ${folder.toUpperCase()}...`;
      
      // The path must be relative to index.html
      const modelPath = `./model/${folder}/model.json`; 
      this._model = await tf.loadLayersModel(modelPath);
      
      this.modelLoaded = true;
      this.status = `DQN ${folder.toUpperCase()} Active`;

      const dummy = tf.zeros([1, 6]);
      this._model.predict(dummy).dispose();
      dummy.dispose();
    } catch (err) {
      this._model = null;
      this.modelLoaded = false;
      this.status = 'Heuristic Fallback';
      console.error('[AIAgent] Model failed to load:', err);
    }
  }

  static buildState(raw) {
    const { ballX, ballY, ballVX, ballVY, aiPaddleY, playerPaddleY, bounds } = raw;
    return new Float32Array([
      ballX / bounds.width,
      ballY / bounds.height,
      ballVX / 15,
      ballVY / 15,
      aiPaddleY / bounds.height,
      playerPaddleY / bounds.height,
    ]);
  }

  selectAction(state) {
    if (this._model) {
      return tf.tidy(() => {
        const input = tf.tensor2d(state, [1, 6]);
        const qValues = this._model.predict(input);
        const actionIndex = qValues.argMax(-1).dataSync()[0];
        // Training action map: 0 = UP, 1 = DOWN, 2 = STAY
        return actionIndex === 0 ? ACTION.UP : actionIndex === 1 ? ACTION.DOWN : ACTION.STAY;
      });
    }
    return this._heuristicAction(state);
  }

  _heuristicAction(state) {
    const diff = state[1] - (state[4] + PADDLE_HEIGHT_RATIO / 2);
    if (diff > 0.03) return ACTION.DOWN;
    if (diff < -0.03) return ACTION.UP;
    return ACTION.STAY;
  }
}