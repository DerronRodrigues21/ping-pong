/* ============================================================
   Paddle — Rectangular entity for player and AI paddles.
   ============================================================ */

/** @typedef {{ width: number, height: number }} Bounds */

export class Paddle {
  /**
   * @param {number} x      – center X position
   * @param {number} y      – center Y position
   * @param {number} width   – paddle width  (thickness)
   * @param {number} height  – paddle height (face length)
   * @param {number} speed   – max movement speed (px per tick)
   */
  constructor(x, y, width = 12, height = 80, speed = 7) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.speed = speed;

    /** Target Y for smooth interpolation (used by mouse control). */
    this.targetY = y;

    /** Score counter for this paddle's side. */
    this.score = 0;
  }

  /* ----------------------------------------------------------
     Movement
     ---------------------------------------------------------- */

  /**
   * Move paddle toward `targetY`, clamped within bounds.
   * @param {Bounds} bounds – logical canvas dimensions
   * @param {number} dt     – delta-time multiplier
   */
  update(bounds, dt = 1) {
    const dy = this.targetY - this.y;
    const step = Math.min(Math.abs(dy), this.speed * dt);
    this.y += Math.sign(dy) * step;

    // Clamp within canvas
    const halfH = this.height / 2;
    if (this.y - halfH < 0) this.y = halfH;
    if (this.y + halfH > bounds.height) this.y = bounds.height - halfH;
  }

  /**
   * Discrete move (for RL agent: -1 = up, 0 = stay, +1 = down).
   * @param {number} action – direction
   * @param {Bounds} bounds – logical canvas dimensions
   * @param {number} dt     – delta-time multiplier
   */
  applyAction(action, bounds, dt = 1) {
    this.y += action * this.speed * dt;

    const halfH = this.height / 2;
    if (this.y - halfH < 0) this.y = halfH;
    if (this.y + halfH > bounds.height) this.y = bounds.height - halfH;
  }

  /* ----------------------------------------------------------
     Rendering
     ---------------------------------------------------------- */

  /**
   * Draw the paddle with rounded corners and a subtle edge glow.
   * @param {CanvasRenderingContext2D} ctx
   * @param {string} [color='hsl(0, 0%, 85%)'] – base fill color
   */
  draw(ctx, color = 'hsl(0, 0%, 85%)') {
    const left = this.x - this.width / 2;
    const top  = this.y - this.height / 2;
    const r = this.width / 2; // corner radius = half the paddle thickness

    // Glow
    ctx.shadowColor = 'hsla(0, 0%, 100%, 0.15)';
    ctx.shadowBlur = 12;

    // Rounded rectangle
    ctx.beginPath();
    ctx.roundRect(left, top, this.width, this.height, r);
    ctx.fillStyle = color;
    ctx.fill();

    // Reset shadow
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // Subtle center line indicator
    ctx.beginPath();
    ctx.roundRect(left + 2, top + 2, this.width - 4, this.height - 4, Math.max(0, r - 2));
    ctx.strokeStyle = 'hsla(0, 0%, 100%, 0.08)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  /* ----------------------------------------------------------
     Geometry helpers
     ---------------------------------------------------------- */

  /** Axis-Aligned Bounding Box of the paddle. */
  get aabb() {
    return {
      left:   this.x - this.width / 2,
      right:  this.x + this.width / 2,
      top:    this.y - this.height / 2,
      bottom: this.y + this.height / 2,
    };
  }
}
