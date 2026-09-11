# engine.py - Physics and Simulation Engine
import random
from config import GameConfig
from models import Vector2D, Paddle, Ball


class ArcadeEngine:
    def __init__(self, config: GameConfig = None):
        self.cfg = config or GameConfig()
        p_h = self.cfg.paddle_height
        init_py = self.cfg.height // 2 - p_h // 2
        self.paddle_left = Paddle(x=2, y=init_py, height=p_h)
        self.paddle_right = Paddle(x=self.cfg.width - 3, y=init_py, height=p_h)
        self.ball = Ball(
            pos=Vector2D(x=self.cfg.width // 2, y=self.cfg.height // 2),
            vel=Vector2D(x=random.choice([-1, 1]), y=random.choice([-1, 1]))
        )
        self.score_left = 0
        self.score_right = 0
        self.rally_count = 0

    def reset_ball(self):
        self.ball.pos.x = self.cfg.width // 2
        self.ball.pos.y = self.cfg.height // 2
        self.ball.vel.x = -self.ball.vel.x
        self.ball.vel.y = random.choice([-1, 1])
        self.rally_count = 0

    def step(self):
        # Autonomous AI Paddle tracking
        self.paddle_left.move_towards(self.ball.pos.y, 1, self.cfg.height - 1)
        self.paddle_right.move_towards(self.ball.pos.y, 1, self.cfg.height - 1)

        # Advance physics
        self.ball.update_position()

        # Top & Bottom Wall Collisions
        if self.ball.pos.y <= 1:
            self.ball.pos.y = 1
            self.ball.vel.y = -self.ball.vel.y
        elif self.ball.pos.y >= self.cfg.height - 2:
            self.ball.pos.y = self.cfg.height - 2
            self.ball.vel.y = -self.ball.vel.y

        # Left Paddle Collision
        if self.ball.pos.x <= 2:
            if self.paddle_left.y <= self.ball.pos.y < self.paddle_left.y + self.paddle_left.height:
                self.ball.pos.x = 3
                self.ball.vel.x = 1
                self.rally_count += 1
            elif self.ball.pos.x <= 0:
                self.score_right += 1
                self.reset_ball()

        # Right Paddle Collision
        elif self.ball.pos.x >= self.cfg.width - 3:
            if self.paddle_right.y <= self.ball.pos.y < self.paddle_right.y + self.paddle_right.height:
                self.ball.pos.x = self.cfg.width - 4
                self.ball.vel.x = -1
                self.rally_count += 1
            elif self.ball.pos.x >= self.cfg.width - 1:
                self.score_left += 1
                self.reset_ball()
