/* ============================================================
   Ball — Pure physics entity for the ping pong simulation.
   No rendering opinions; draw() receives a CanvasRenderingContext2D.
   ============================================================ */

/** @typedef {{ width: number, height: number }} Bounds */

export class Ball {
  /**
   * @param {number} x       – initial center X
   * @param {number} y       – initial center Y
   * @param {number} radius  – ball radius in logical pixels
   * @param {number} speed   – base scalar speed (px / frame at 60 fps)
   */
  constructor(x, y, radius = 8, speed = 6) {
    this.startX = x;
    this.startY = y;
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.baseSpeed = speed;

    /** Velocity components (px per tick) */
    this.vx = 0;
    this.vy = 0;

    /** Trail history for the glow effect */
    this._trail = [];
    this._maxTrail = 8;
  }

  /* ----------------------------------------------------------
     Lifecycle
     ---------------------------------------------------------- */

  /**
   * Launch the ball from a specific position toward a direction.
   * @param {1 | -1} direction – +1 = launch right, -1 = launch left
   * @param {number} [spawnX]  – starting X (defaults to center)
   * @param {number} [spawnY]  – starting Y (defaults to center)
   */
  launch(direction, spawnX = this.startX, spawnY = this.startY) {
    this.x = spawnX;
    this.y = spawnY;
    this._trail = [];

    const angle = (Math.random() * Math.PI / 3) - (Math.PI / 6); // ±30°
    this.vx = this.baseSpeed * Math.cos(angle) * direction;
    this.vy = this.baseSpeed * Math.sin(angle);
  }

  /**
   * Advance position by one tick, bouncing off top/bottom walls.
   * @param {Bounds} bounds – logical canvas dimensions
   * @param {number} dt     – delta-time multiplier (1.0 = perfect 60 fps frame)
   */
  update(bounds, dt = 1) {
    // Store trail position before moving
    this._trail.unshift({ x: this.x, y: this.y });
    if (this._trail.length > this._maxTrail) this._trail.pop();

    this.x += this.vx * dt;
    this.y += this.vy * dt;

    // Top / bottom wall bounce
    if (this.y - this.radius <= 0) {
      this.y = this.radius;
      this.vy = Math.abs(this.vy);
    } else if (this.y + this.radius >= bounds.height) {
      this.y = bounds.height - this.radius;
      this.vy = -Math.abs(this.vy);
    }
  }

  /* ----------------------------------------------------------
     Rendering
     ---------------------------------------------------------- */

  /**
   * Draw the ball with a subtle motion trail and glow.
   * @param {CanvasRenderingContext2D} ctx
   */
  draw(ctx) {
    // Motion trail
    for (let i = 0; i < this._trail.length; i++) {
      const t = this._trail[i];
      const alpha = 0.15 * (1 - i / this._trail.length);
      ctx.beginPath();
      ctx.arc(t.x, t.y, this.radius * 0.8, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(165, 80%, 55%, ${alpha})`;
      ctx.fill();
    }

    // Outer glow
    ctx.shadowColor = 'hsla(165, 80%, 48%, 0.6)';
    ctx.shadowBlur = 18;

    // Ball body
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fillStyle = 'hsl(165, 80%, 60%)';
    ctx.fill();

    // Inner highlight
    ctx.beginPath();
    ctx.arc(this.x - this.radius * 0.25, this.y - this.radius * 0.25, this.radius * 0.4, 0, Math.PI * 2);
    ctx.fillStyle = 'hsla(165, 90%, 85%, 0.35)';
    ctx.fill();

    // Reset shadow so it doesn't bleed into other draws
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }

  /* ----------------------------------------------------------
     Geometry helpers (used by collision detection in Engine)
     ---------------------------------------------------------- */

  /** Axis-Aligned Bounding Box of the ball. */
  get aabb() {
    return {
      left:   this.x - this.radius,
      right:  this.x + this.radius,
      top:    this.y - this.radius,
      bottom: this.y + this.radius,
    };
  }
}
