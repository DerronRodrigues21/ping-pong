"""
Headless Pong environment for DQN training.
Pure logic — no rendering, no pygame.
Canvas: 800x500, physics matched to browser exactly.
"""

import math
import random
import numpy as np

# --- Canvas ---
W, H = 800, 500

# --- Ball ---
BALL_RADIUS = 6
BALL_BASE_SPEED = 6.0
BALL_MAX_SPEED = BALL_BASE_SPEED * 2.5  # 15.0
BALL_HIT_MULTIPLIER = 1.05

# --- Paddles ---
PADDLE_W = 12
PADDLE_H = 100
PADDLE_OFFSET = 15
AI_SPEED = 5.0
OPP_BASE_SPEED = 4.5
OPP_NOISE_STD = 0.5
HALF_PADDLE = PADDLE_H / 2
MAX_RETURN_ANGLE = 75 * (math.pi / 180)  # 75 degrees in radians

# --- Scoring ---
WIN_SCORE = 11


class PongEnv:
    """OpenAI-gym-style interface: reset() → state, step(action) → (state, reward, done, info)."""

    def __init__(self):
        self.state_dim = 6
        self.action_dim = 3
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset to a new game. Returns initial state."""
        self.ball_x = W / 2
        self.ball_y = H / 2
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_speed = BALL_BASE_SPEED

        # Paddle y = top edge
        self.ai_y = (H - PADDLE_H) / 2
        self.opp_y = (H - PADDLE_H) / 2

        self.ai_score = 0
        self.opp_score = 0
        self.total_points = 0
        self.server = "opp"  # opponent serves first (ball goes right toward AI)

        self._serve()
        return self._get_state()

    def step(self, action):
        """
        Advance one frame.
        action: 0=up, 1=down, 2=stay
        Returns: (state, reward, done, info)
        """
        reward = -0.001  # living penalty

        # --- Move AI paddle ---
        if action == 0:
            self.ai_y -= AI_SPEED
        elif action == 1:
            self.ai_y += AI_SPEED
        self.ai_y = max(0, min(H - PADDLE_H, self.ai_y))

        # --- Move opponent (rule-based) ---
        self._move_opponent()

        # --- Ball physics ---
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Wall bounce
        if self.ball_y - BALL_RADIUS <= 0:
            self.ball_y = BALL_RADIUS
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y + BALL_RADIUS >= H:
            self.ball_y = H - BALL_RADIUS
            self.ball_vy = -abs(self.ball_vy)

        # --- Paddle collisions ---
        hit_ai = self._check_paddle_collision_ai()
        hit_opp = self._check_paddle_collision_opp()

        if hit_ai:
            reward += 1.0  # AI successfully returned the ball

        # --- Scoring ---
        done = False
        info = {}

        if self.ball_x - BALL_RADIUS <= 0:
            # Opponent missed — AI wins point
            self.ai_score += 1
            reward += 0.5
            done = self._check_game_over()
            info["point"] = "ai"
            if not done:
                self._new_point()
        elif self.ball_x + BALL_RADIUS >= W:
            # AI missed — loses point
            self.opp_score += 1
            reward -= 1.0
            done = self._check_game_over()
            info["point"] = "opp"
            if not done:
                self._new_point()

        if done:
            info["winner"] = "ai" if self.ai_score > self.opp_score else "opp"

        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_state(self):
        return np.array([
            self.ball_x / W,
            self.ball_y / H,
            self.ball_vx / BALL_MAX_SPEED,
            self.ball_vy / BALL_MAX_SPEED,
            self.ai_y / H,
            self.opp_y / H,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def _serve(self):
        self.ball_x = W / 2
        self.ball_y = H / 2
        self.ball_speed = BALL_BASE_SPEED

        # Serve toward receiver
        direction = 1 if self.server == "opp" else -1  # opp serves → ball goes right
        angle = random.uniform(-30, 30) * (math.pi / 180)
        self.ball_vx = self.ball_speed * math.cos(angle) * direction
        self.ball_vy = self.ball_speed * math.sin(angle)

    def _new_point(self):
        self.total_points += 1
        self._update_server()
        self._serve()

    def _update_server(self):
        is_deuce = self.ai_score >= 10 and self.opp_score >= 10
        if is_deuce:
            self.server = "opp" if self.total_points % 2 == 0 else "ai"
        else:
            self.server = "opp" if (self.total_points // 2) % 2 == 0 else "ai"

    def _check_game_over(self):
        hi = max(self.ai_score, self.opp_score)
        lo = min(self.ai_score, self.opp_score)
        if hi < WIN_SCORE:
            return False
        return (hi - lo >= 2) if (self.ai_score >= 10 and self.opp_score >= 10) else True

    # ------------------------------------------------------------------
    # Opponent AI (rule-based)
    # ------------------------------------------------------------------

    def _move_opponent(self):
        target = self.ball_y - HALF_PADDLE
        speed = OPP_BASE_SPEED + random.gauss(0, OPP_NOISE_STD)
        speed = max(0, speed)
        dy = target - self.opp_y
        step = min(abs(dy), speed)
        self.opp_y += math.copysign(step, dy)
        self.opp_y = max(0, min(H - PADDLE_H, self.opp_y))

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def _check_paddle_collision_ai(self):
        """AI paddle is on the right side."""
        px = W - PADDLE_OFFSET - PADDLE_W
        if (self.ball_x + BALL_RADIUS >= px and
            self.ball_x - BALL_RADIUS <= px + PADDLE_W and
            self.ball_y + BALL_RADIUS >= self.ai_y and
            self.ball_y - BALL_RADIUS <= self.ai_y + PADDLE_H and
            self.ball_vx > 0):

            hit_pos = self.ball_y - (self.ai_y + HALF_PADDLE)
            norm_hit = max(-1, min(1, hit_pos / HALF_PADDLE))
            angle = norm_hit * MAX_RETURN_ANGLE

            self.ball_speed = min(self.ball_speed * BALL_HIT_MULTIPLIER, BALL_MAX_SPEED)
            self.ball_vx = -self.ball_speed * math.cos(angle)
            self.ball_vy = self.ball_speed * math.sin(angle)
            self.ball_x = px - BALL_RADIUS  # prevent tunneling
            return True
        return False

    def _check_paddle_collision_opp(self):
        """Opponent paddle is on the left side."""
        px = PADDLE_OFFSET
        if (self.ball_x - BALL_RADIUS <= px + PADDLE_W and
            self.ball_x + BALL_RADIUS >= px and
            self.ball_y + BALL_RADIUS >= self.opp_y and
            self.ball_y - BALL_RADIUS <= self.opp_y + PADDLE_H and
            self.ball_vx < 0):

            hit_pos = self.ball_y - (self.opp_y + HALF_PADDLE)
            norm_hit = max(-1, min(1, hit_pos / HALF_PADDLE))
            angle = norm_hit * MAX_RETURN_ANGLE

            self.ball_speed = min(self.ball_speed * BALL_HIT_MULTIPLIER, BALL_MAX_SPEED)
            self.ball_vx = self.ball_speed * math.cos(angle)
            self.ball_vy = self.ball_speed * math.sin(angle)
            self.ball_x = px + PADDLE_W + BALL_RADIUS  # prevent tunneling
            return True
        return False
