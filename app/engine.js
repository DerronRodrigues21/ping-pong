import { Ball } from './ball.js';
import { Paddle } from './paddle.js';
import { AIAgent } from './agent.js';

const CANVAS_W = 800;
const CANVAS_H = 500;
const BALL_RADIUS = 6;
const BALL_MAX_SPEED = 15;
const PADDLE_H = 100;

class GameEngine {
  constructor() {
    this.canvas = document.getElementById('game-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.bounds = { width: CANVAS_W, height: CANVAS_H };

    this.ball = new Ball(CANVAS_W / 2, CANVAS_H / 2, BALL_RADIUS);
    this.playerPaddle = new Paddle(27.5, CANVAS_H / 2, 12, PADDLE_H, 5);
    this.aiPaddle = new Paddle(CANVAS_W - 27.5, CANVAS_H / 2, 12, PADDLE_H, 5);

    this.agent = new AIAgent();
    this._running = false;
    this._isPaused = false;
    this._waitingForServe = false;
    this._lastTime = 0;
    this._serveSide = 'player';
    this._totalPoints = 0;

    this._setupHighDPI();
    this._bindInput();
    this._bindControls();
    
    // EXPOSE TO WINDOW: Allows index.html buttons to call this function
    window.startGame = (difficulty) => this.startGame(difficulty);
    
    requestAnimationFrame((t) => this._loop(t));
  }

  async startGame(difficulty) {
    document.getElementById('game-overlay').classList.add('hidden');
    document.getElementById('ingame-controls').classList.remove('hidden');
    
    await this.agent.init(difficulty);
    document.getElementById('game-status').textContent = this.agent.status;
    
    this._running = true;
    this._prepareServe();
  }

  _bindControls() {
    document.getElementById('btn-pause').onclick = () => {
      this._isPaused = !this._isPaused;
      document.getElementById('btn-pause').textContent = this._isPaused ? 'RESUME' : 'PAUSE';
    };
    document.getElementById('btn-exit').onclick = () => location.reload();

    // Serve triggers: Space bar or Mouse Click
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && this._waitingForServe) this._executeServe();
    });
    this.canvas.addEventListener('mousedown', () => {
      if (this._waitingForServe) this._executeServe();
    });
  }

  _prepareServe() {
    this._waitingForServe = true;
    document.getElementById('game-overlay').classList.remove('hidden');
    document.getElementById('menu-home').classList.add('hidden');
    document.getElementById('serve-prompt').classList.remove('hidden');
    
    this.ball.vx = 0;
    this.ball.vy = 0;
    this.ball.x = CANVAS_W / 2;
    this.ball.y = CANVAS_H / 2;
  }

  _executeServe() {
    this._waitingForServe = false;
    document.getElementById('game-overlay').classList.add('hidden');
    document.getElementById('serve-prompt').classList.add('hidden');
    this.ball.launch(this._serveSide === 'player' ? 1 : -1);
  }

  _loop(timestamp) {
    // FPS CODE START: Initialize _lastTime to prevent 0 FPS
    if (!this._lastTime) this._lastTime = timestamp;
    const dt = Math.min(timestamp - this._lastTime, 33.33) / 16.667;
    this._lastTime = timestamp;

    // Update FPS Counter in the UI
    if (Math.round(timestamp) % 30 === 0) {
      document.getElementById('fps-counter').textContent = `${Math.round(60 / dt)} FPS`;
    }
    // FPS CODE END

    if (this._running && !this._isPaused && !this._waitingForServe) {
      this._update(dt);
    }
    this._render();
    requestAnimationFrame((t) => this._loop(t));
  }

  _update(dt) {
    this.playerPaddle.update(this.bounds, dt);
    
    const state = AIAgent.buildState({
      ballX: this.ball.x, ballY: this.ball.y,
      ballVX: this.ball.vx, ballVY: this.ball.vy,
      aiPaddleY: this.aiPaddle.y - PADDLE_H / 2,
      playerPaddleY: this.playerPaddle.y - PADDLE_H / 2,
      bounds: this.bounds
    });

    const action = this.agent.selectAction(state);
    this.aiPaddle.applyAction(action, this.bounds, dt);
    this.ball.update(this.bounds, dt);

    this._checkCollision(this.playerPaddle, 1);
    this._checkCollision(this.aiPaddle, -1);

    if (this.ball.x < 0) { this.aiPaddle.score++; this._onScore(); }
    if (this.ball.x > CANVAS_W) { this.playerPaddle.score++; this._onScore(); }
  }

  /** Anti-Stuck Fix: Move ball outside paddle after collision */
  _checkCollision(paddle, reflect) {
    const b = this.ball.aabb;
    const p = paddle.aabb;

    if (b.right >= p.left && b.left <= p.right && b.bottom >= p.top && b.top <= p.bottom) {
      const hit = (this.ball.y - paddle.y) / (PADDLE_H / 2);
      const angle = hit * (75 * Math.PI / 180);
      const currentSpeed = Math.hypot(this.ball.vx, this.ball.vy);
      const speed = Math.min(currentSpeed * 1.05, BALL_MAX_SPEED);

      this.ball.vx = speed * Math.cos(angle) * reflect;
      this.ball.vy = speed * Math.sin(angle);

      // FORCE BALL OUT: Prevent "freeze" loop by moving it clear of the paddle
      this.ball.x = reflect === 1 ? p.right + BALL_RADIUS + 1 : p.left - BALL_RADIUS - 1;
    }
  }

  _onScore() {
    document.getElementById('score-player').textContent = this.playerPaddle.score;
    document.getElementById('score-ai').textContent = this.aiPaddle.score;
    this._totalPoints++;
    this._serveSide = (Math.floor(this._totalPoints / 2) % 2 === 0) ? 'player' : 'ai';
    this._prepareServe();
  }

  _setupHighDPI() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = CANVAS_W * dpr; this.canvas.height = CANVAS_H * dpr;
    this.canvas.style.width = `${CANVAS_W}px`; this.canvas.style.height = `${CANVAS_H}px`;
    this.ctx.scale(dpr, dpr);
  }

  _bindInput() {
    this.canvas.onmousemove = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.playerPaddle.targetY = (e.clientY - rect.top) * (CANVAS_H / rect.height);
    };
  }

  _render() {
    this.ctx.fillStyle = 'hsl(230, 22%, 8%)';
    this.ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    this._drawCourt();
    this.playerPaddle.draw(this.ctx, 'hsl(0, 0%, 82%)');
    this.aiPaddle.draw(this.ctx, 'hsl(165, 60%, 50%)');
    this.ball.draw(this.ctx);
  }

  _drawCourt() {
    this.ctx.strokeStyle = 'hsla(230, 15%, 28%, 0.6)';
    this.ctx.setLineDash([8, 10]);
    this.ctx.beginPath(); this.ctx.moveTo(CANVAS_W / 2, 0); this.ctx.lineTo(CANVAS_W / 2, CANVAS_H); this.ctx.stroke();
    this.ctx.setLineDash([]);
  }
}

document.addEventListener('DOMContentLoaded', () => { new GameEngine(); });