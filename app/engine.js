/* ============================================================
   GameEngine — Core loop, rendering, physics orchestration.

   Responsible for:
     • High-DPI canvas setup
     • 60 fps game loop with delta-time correction
     • AABB collision detection (ball ↔ paddles)
     • Score tracking & UI sync
     • Court rendering (center line, etc.)
     • Delegating AI paddle control to AIAgent (inference only)

   This module is the **only** thing `index.html` imports.
   It boots itself on DOM-ready.
   ============================================================ */

import { Ball }    from './ball.js';
import { Paddle }  from './paddle.js';
import { AIAgent } from './agent.js';

/* ----------------------------------------------------------
   Constants
   ---------------------------------------------------------- */

/** Logical (CSS) dimensions of the playing field. */
const CANVAS_W = 800;
const CANVAS_H = 500;

/** Physics values mirrored from training/environment.py. */
const BALL_RADIUS = 6;
const BALL_BASE_SPEED = 6;
const BALL_MAX_SPEED = 15;
const BALL_HIT_MULTIPLIER = 1.05;
const PADDLE_W = 12;
const PADDLE_H = 100;
const PADDLE_OFFSET = 15;
const AI_SPEED = 5;
const PLAYER_SPEED = 4.5;
const MAX_RETURN_ANGLE = 75 * (Math.PI / 180);

/* ----------------------------------------------------------
   GameEngine
   ---------------------------------------------------------- */

class GameEngine {
  constructor() {
    /** @type {HTMLCanvasElement} */
    this.canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('game-canvas'));
    /** @type {CanvasRenderingContext2D} */
    this.ctx = /** @type {CanvasRenderingContext2D} */ (this.canvas.getContext('2d'));

    /** Logical bounds used by all game objects. */
    this.bounds = { width: CANVAS_W, height: CANVAS_H };

    // --- Entities ---
    this.ball = new Ball(CANVAS_W / 2, CANVAS_H / 2, BALL_RADIUS, BALL_BASE_SPEED);

    this.playerPaddle = new Paddle(
      PADDLE_OFFSET + PADDLE_W / 2,
      CANVAS_H / 2,
      PADDLE_W, PADDLE_H, PLAYER_SPEED,
    );

    this.aiPaddle = new Paddle(
      CANVAS_W - PADDLE_OFFSET - PADDLE_W / 2,
      CANVAS_H / 2,
      PADDLE_W, PADDLE_H, AI_SPEED,
    );

    // --- Timing ---
    this._lastTime = 0;
    this._frameCount = 0;
    this._fpsAccum = 0;
    this._fpsDisplay = 60;

    // --- DOM refs ---
    this._scorePlayerEl = document.getElementById('score-player');
    this._scoreAiEl     = document.getElementById('score-ai');
    this._fpsEl         = document.getElementById('fps-counter');
    this._statusEl      = document.getElementById('game-status');

    // --- AI Agent (inference only) ---
    this.agent = new AIAgent();

    // --- State ---
    this._running = false;
    this._serveDelay = 0;       // frames to wait before re-serving
    this._serveCooldown = 45;   // ~0.75 s at 60 fps

    /**
     * Serve tracking — proper ping pong rules.
     * 'player' = left paddle serves (ball launches right).
     * 'ai'     = right paddle serves (ball launches left).
     */
    this._serveSide = 'player'; // player always serves first
    this._totalPoints = 0;      // used for alternating every 2 points (ITTF rule)

    // --- Init ---
    this._setupHighDPI();
    this._bindInput();
    this._initAgent();
    this._serve();
    this._running = true;
    requestAnimationFrame((t) => this._loop(t));
  }

  /* ----------------------------------------------------------
     High-DPI / Retina Setup
     ---------------------------------------------------------- */

  _setupHighDPI() {
    const dpr = window.devicePixelRatio || 1;

    // Physical (backing store) size
    this.canvas.width  = CANVAS_W * dpr;
    this.canvas.height = CANVAS_H * dpr;

    // CSS (logical) size
    this.canvas.style.width  = `${CANVAS_W}px`;
    this.canvas.style.height = `${CANVAS_H}px`;

    // Scale the context so all draw calls use logical coordinates
    this.ctx.scale(dpr, dpr);
  }

  /* ----------------------------------------------------------
     AI Agent Init
     ---------------------------------------------------------- */

  async _initAgent() {
    await this.agent.init();
    this._statusEl.textContent = this.agent.status;
  }

  /* ----------------------------------------------------------
     Input
     ---------------------------------------------------------- */

  _bindInput() {
    // Track mouse Y relative to the canvas for the player paddle
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleY = CANVAS_H / rect.height;
      this.playerPaddle.targetY = (e.clientY - rect.top) * scaleY;
    });

    // Touch support (mobile)
    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const scaleY = CANVAS_H / rect.height;
      const touch = e.touches[0];
      this.playerPaddle.targetY = (touch.clientY - rect.top) * scaleY;
    }, { passive: false });
  }

  /* ----------------------------------------------------------
     Main Loop
     ---------------------------------------------------------- */

  /** @param {number} timestamp – from requestAnimationFrame */
  _loop(timestamp) {
    if (!this._running) return;

    // Delta-time (clamped to avoid spiral-of-death on tab switch)
    const rawDt = timestamp - this._lastTime;
    const dt = Math.min(rawDt, 33.33) / 16.667; // normalize to 60 fps
    this._lastTime = timestamp;

    // FPS counter (updates once per second)
    this._frameCount++;
    this._fpsAccum += rawDt;
    if (this._fpsAccum >= 1000) {
      this._fpsDisplay = Math.round((this._frameCount * 1000) / this._fpsAccum);
      this._fpsEl.textContent = `${this._fpsDisplay} FPS`;
      this._frameCount = 0;
      this._fpsAccum = 0;
    }

    this._update(dt);
    this._render();

    requestAnimationFrame((t) => this._loop(t));
  }

  /* ----------------------------------------------------------
     Update
     ---------------------------------------------------------- */

  _update(dt) {
    // Serve cooldown after a point is scored
    if (this._serveDelay > 0) {
      this._serveDelay -= dt;
      if (this._serveDelay <= 0) {
        this._serveDelay = 0;
        this._serve();
      }
      return; // freeze while waiting
    }

    // Move player paddle (mouse-driven)
    this.playerPaddle.update(this.bounds, dt);

    // AI paddle — agent selects action via model inference or heuristic
    const state = AIAgent.buildState({
      ballX:         this.ball.x,
      ballY:         this.ball.y,
      ballVX:        this.ball.vx,
      ballVY:        this.ball.vy,
      aiPaddleY:     this.aiPaddle.y - this.aiPaddle.height / 2,
      playerPaddleY: this.playerPaddle.y - this.playerPaddle.height / 2,
      bounds:        this.bounds,
    });
    const action = this.agent.selectAction(state);
    this.aiPaddle.applyAction(action, this.bounds, dt);

    this.ball.update(this.bounds, dt);

    // Collision detection
    this._checkPaddleCollision(this.playerPaddle, 1);
    this._checkPaddleCollision(this.aiPaddle, -1);

    // Scoring (ball exits left or right)
    if (this.ball.x - this.ball.radius <= 0) {
      this.aiPaddle.score++;
      this._onScore();
    } else if (this.ball.x + this.ball.radius >= CANVAS_W) {
      this.playerPaddle.score++;
      this._onScore();
    }
  }

  /* ----------------------------------------------------------
     Collision Detection (AABB)
     ---------------------------------------------------------- */

  /**
   * Test & resolve ball collision against a paddle.
   * @param {Paddle} paddle
   * @param {1 | -1} reflectDir – direction to bounce the ball (+1 = right, -1 = left)
   */
  _checkPaddleCollision(paddle, reflectDir) {
    const b = this.ball.aabb;
    const p = paddle.aabb;

    const overlaps =
      b.right  >= p.left  &&
      b.left   <= p.right &&
      b.bottom >= p.top   &&
      b.top    <= p.bottom;

    if (!overlaps) return;

    // Determine where on the paddle face the ball hit (−1 to +1)
    const hitPoint = (this.ball.y - paddle.y) / (paddle.height / 2);
    const clampedHit = Math.max(-1, Math.min(1, hitPoint));

    const angle = clampedHit * MAX_RETURN_ANGLE;

    const currentSpeed = Math.sqrt(this.ball.vx ** 2 + this.ball.vy ** 2);
    const newSpeed = Math.min(currentSpeed * BALL_HIT_MULTIPLIER, BALL_MAX_SPEED);

    this.ball.vx = newSpeed * Math.cos(angle) * reflectDir;
    this.ball.vy = newSpeed * Math.sin(angle);

    // Push ball outside the paddle to prevent repeat collisions
    if (reflectDir === 1) {
      this.ball.x = p.right + this.ball.radius;
    } else {
      this.ball.x = p.left - this.ball.radius;
    }
  }

  /* ----------------------------------------------------------
     Scoring
     ---------------------------------------------------------- */

  _onScore() {
    this._scorePlayerEl.textContent = this.playerPaddle.score;
    this._scoreAiEl.textContent     = this.aiPaddle.score;
    this._serveDelay = this._serveCooldown;
    this._totalPoints++;

    // Alternate serve every 2 points (ITTF rule)
    this._serveSide = (Math.floor(this._totalPoints / 2) % 2 === 0) ? 'player' : 'ai';

    // Park ball at the server's paddle while we wait
    const server = this._serveSide === 'player' ? this.playerPaddle : this.aiPaddle;
    this.ball.vx = 0;
    this.ball.vy = 0;
    this.ball.x = server.x;
    this.ball.y = server.y;
    this.ball._trail = [];
  }

  /**
   * Execute a serve from the current server's side.
   * Ball spawns at the serving paddle and launches toward the opponent.
   */
  _serve() {
    const direction = this._serveSide === 'player' ? 1 : -1;
    this.ball.launch(direction, CANVAS_W / 2, CANVAS_H / 2);
  }

  /* ----------------------------------------------------------
     Rendering
     ---------------------------------------------------------- */

  _render() {
    const ctx = this.ctx;

    // Clear
    ctx.fillStyle = 'hsl(230, 22%, 8%)';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Court decorations
    this._drawCourt(ctx);

    // Entities
    this.playerPaddle.draw(ctx, 'hsl(0, 0%, 82%)');
    this.aiPaddle.draw(ctx, 'hsl(165, 60%, 50%)');
    this.ball.draw(ctx);
  }

  /** Draw center line, center circle, and other court markings. */
  _drawCourt(ctx) {
    const centerX = CANVAS_W / 2;
    const centerY = CANVAS_H / 2;

    // Dashed center line
    ctx.setLineDash([8, 10]);
    ctx.strokeStyle = 'hsla(230, 15%, 28%, 0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, CANVAS_H);
    ctx.stroke();
    ctx.setLineDash([]);

    // Center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, 50, 0, Math.PI * 2);
    ctx.strokeStyle = 'hsla(230, 15%, 25%, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Small center dot
    ctx.beginPath();
    ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'hsla(230, 15%, 30%, 0.5)';
    ctx.fill();
  }
}

/* ----------------------------------------------------------
   Boot
   ---------------------------------------------------------- */

document.addEventListener('DOMContentLoaded', () => {
  window.__engine = new GameEngine();
});
