# models.py - Domain Entities and Vector Math
from dataclasses import dataclass


@dataclass
class Vector2D:
    x: int
    y: int


@dataclass
class Paddle:
    x: int
    y: int
    height: int

    def move_towards(self, target_y: int, min_y: int, max_y: int):
        center = self.y + self.height // 2
        if center < target_y and self.y + self.height < max_y:
            self.y += 1
        elif center > target_y and self.y > min_y:
            self.y -= 1


@dataclass
class Ball:
    pos: Vector2D
    vel: Vector2D

    def update_position(self):
        self.pos.x += self.vel.x
        self.pos.y += self.vel.y
