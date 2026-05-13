/* ============================================================
   GameEngine — Core loop, rendering, physics orchestration.
   ============================================================ */

import { Ball }    from './ball.js';
import { Paddle }  from './paddle.js';
import { AIAgent } from './agent.js';

const CANVAS_W = 800;
const CANVAS_H = 500;
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

class GameEngine {
  constructor() {
    this.canvas = document.getElementById('game-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.bounds = { width: CANVAS_W, height: CANVAS_H };

    this.ball = new Ball(CANVAS_W / 2, CANVAS_H / 2, BALL_RADIUS, BALL_BASE_SPEED);
    this.playerPaddle = new Paddle(PADDLE_OFFSET + PADDLE_W / 2, CANVAS_H / 2, PADDLE_W, PADDLE_H, PLAYER_SPEED);
    this.aiPaddle = new Paddle(CANVAS_W - PADDLE_OFFSET - PADDLE_W / 2, CANVAS_H / 2, PADDLE_W, PADDLE_H, AI_SPEED);

    this._lastTime = 0;
    this._frameCount = 0;
    this._fpsAccum = 0;
    this._fpsDisplay = 60;

    this._scorePlayerEl = document.getElementById('score-player');
    this._scoreAiEl     = document.getElementById('score-ai');
    this._fpsEl         = document.getElementById('fps-counter');
    this._statusEl      = document.getElementById('game-status');
    
    this._startScreen   = document.getElementById('start-screen');
    this._serveOverlay  = document.getElementById('serve-overlay');
    this._serveMessage  = document.getElementById('serve-message');
    this._readyBtn      = document.getElementById('ready-button');
    this._serveHint     = document.getElementById('serve-hint');

    this.agent = new AIAgent();

    this._running = false;
    this._serveSide = 'player';
    this._totalPoints = 0;
    
    this._waitingForServe = false;
    this.keys = {};

    this._setupHighDPI();
    this._bindInput();
    this._bindUI();
  }

  _bindUI() {
    const buttons = document.querySelectorAll('.difficulty-buttons button');
    buttons.forEach(btn => {
      btn.addEventListener('click', async () => {
        const diff = btn.getAttribute('data-level');
        this._startScreen.classList.add('hidden');
        await this._initAgent(diff);
        this._prepareServe();
        this._running = true;
        requestAnimationFrame((t) => this._loop(t));
      });
    });

    this._readyBtn.addEventListener('click', () => {
      if (this._waitingForServe && this._serveSide === 'ai') {
        this._executeServe();
      }
    });
  }

  async _initAgent(difficulty) {
    await this.agent.init(difficulty);
    this._statusEl.textContent = this.agent.status;
  }

  _setupHighDPI() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = CANVAS_W * dpr;
    this.canvas.height = CANVAS_H * dpr;
    this.canvas.style.width  = `${CANVAS_W}px`;
    this.canvas.style.height = `${CANVAS_H}px`;
    this.ctx.scale(dpr, dpr);
  }

  _bindInput() {
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleY = CANVAS_H / rect.height;
      this.playerPaddle.targetY = (e.clientY - rect.top) * scaleY;
    });

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const scaleY = CANVAS_H / rect.height;
      const touch = e.touches[0];
      this.playerPaddle.targetY = (touch.clientY - rect.top) * scaleY;
    }, { passive: false });

    window.addEventListener('keydown', (e) => {
      this.keys[e.code] = true;
      if (this._waitingForServe) {
        if (this._serveSide === 'player' && e.code === 'Space') {
          this._executeServe();
        } else if (this._serveSide === 'ai' && e.code === 'KeyK') {
          this._executeServe();
        }
      }
    });

    window.addEventListener('keyup', (e) => {
      this.keys[e.code] = false;
    });

    this.canvas.addEventListener('click', () => {
      if (this._waitingForServe && this._serveSide === 'player') {
        this._executeServe();
      }
    });
  }

  _loop(timestamp) {
    if (!this._running) return;
    const rawDt = timestamp - this._lastTime;
    const dt = Math.min(rawDt, 33.33) / 16.667;
    this._lastTime = timestamp;

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

  _update(dt) {
    if (this.keys['ArrowUp']) {
      this.playerPaddle.targetY -= PLAYER_SPEED * dt * 2; 
    }
    if (this.keys['ArrowDown']) {
      this.playerPaddle.targetY += PLAYER_SPEED * dt * 2;
    }
    
    this.playerPaddle.update(this.bounds, dt);

    if (this._waitingForServe) {
      const server = this._serveSide === 'player' ? this.playerPaddle : this.aiPaddle;
      this.ball.x = server.x + (this._serveSide === 'player' ? this.ball.radius + 2 : -this.ball.radius - 2);
      this.ball.y = server.y;
      
      if (this._serveSide === 'ai') {
        // AI tracks center when waiting
        this.aiPaddle.targetY = CANVAS_H / 2;
        this.aiPaddle.update(this.bounds, dt);
      }
      return;
    }

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

    this._checkPaddleCollision(this.playerPaddle, 1);
    this._checkPaddleCollision(this.aiPaddle, -1);

    if (this.ball.x - this.ball.radius <= 0) {
      this.aiPaddle.score++;
      this._onScore();
    } else if (this.ball.x + this.ball.radius >= CANVAS_W) {
      this.playerPaddle.score++;
      this._onScore();
    }
  }

  _checkPaddleCollision(paddle, reflectDir) {
    const b = this.ball.aabb;
    const p = paddle.aabb;

    const overlaps = b.right >= p.left && b.left <= p.right && b.bottom >= p.top && b.top <= p.bottom;
    if (!overlaps) return;

    const hitPoint = (this.ball.y - paddle.y) / (paddle.height / 2);
    const clampedHit = Math.max(-1, Math.min(1, hitPoint));
    const angle = clampedHit * MAX_RETURN_ANGLE;
    const currentSpeed = Math.sqrt(this.ball.vx ** 2 + this.ball.vy ** 2);
    const newSpeed = Math.min(currentSpeed * BALL_HIT_MULTIPLIER, BALL_MAX_SPEED);

    this.ball.vx = newSpeed * Math.cos(angle) * reflectDir;
    this.ball.vy = newSpeed * Math.sin(angle);

    if (reflectDir === 1) {
      this.ball.x = p.right + this.ball.radius;
    } else {
      this.ball.x = p.left - this.ball.radius;
    }
  }

  _onScore() {
    this._scorePlayerEl.textContent = this.playerPaddle.score;
    this._scoreAiEl.textContent     = this.aiPaddle.score;
    this._totalPoints++;

    // Serve switch logic
    if (this.playerPaddle.score >= 10 && this.aiPaddle.score >= 10) {
      // Alternate every point
      this._serveSide = this._totalPoints % 2 === 0 ? 'player' : 'ai';
    } else {
      // Alternate every 2 points
      this._serveSide = Math.floor(this._totalPoints / 2) % 2 === 0 ? 'player' : 'ai';
    }

    this._prepareServe();
  }

  _prepareServe() {
    this._waitingForServe = true;
    this.ball.vx = 0;
    this.ball.vy = 0;
    this.ball._trail = [];
    
    this._serveOverlay.classList.remove('hidden');
    if (this._serveSide === 'player') {
      this._serveMessage.textContent = "Player's Serve";
      this._serveHint.classList.remove('hidden');
      this._readyBtn.classList.add('hidden');
    } else {
      this._serveMessage.textContent = "AI's Serve";
      this._serveHint.classList.add('hidden');
      this._readyBtn.classList.remove('hidden');
    }
  }

  _executeServe() {
    this._waitingForServe = false;
    this._serveOverlay.classList.add('hidden');
    const direction = this._serveSide === 'player' ? 1 : -1;
    this.ball.launch(direction, this.ball.x, this.ball.y);
  }

  _render() {
    const ctx = this.ctx;
    ctx.fillStyle = 'hsl(230, 22%, 8%)';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    this._drawCourt(ctx);
    this.playerPaddle.draw(ctx, 'hsl(0, 0%, 82%)');
    this.aiPaddle.draw(ctx, 'hsl(165, 60%, 50%)');
    this.ball.draw(ctx);
  }

  _drawCourt(ctx) {
    const centerX = CANVAS_W / 2;
    const centerY = CANVAS_H / 2;
    ctx.setLineDash([8, 10]);
    ctx.strokeStyle = 'hsla(230, 15%, 28%, 0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, CANVAS_H);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(centerX, centerY, 50, 0, Math.PI * 2);
    ctx.strokeStyle = 'hsla(230, 15%, 25%, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'hsla(230, 15%, 30%, 0.5)';
    ctx.fill();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.__engine = new GameEngine();
});
