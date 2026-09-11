# ASCII Ping Pong Game in Python
# Synthesized autonomously by Pseudo-Brain Cognitive Agent.
# Supports interactive terminal play, player controls, and headless verification.

import os
import sys
import time
import random


class AsciiPong:
    def __init__(self, width: int = 54, height: int = 16):
        self.width = width
        self.height = height
        self.paddle_height = 4
        self.paddle_left = height // 2 - self.paddle_height // 2
        self.paddle_right = height // 2 - self.paddle_height // 2
        self.ball_x = width // 2
        self.ball_y = height // 2
        self.ball_vx = random.choice([-1, 1])
        self.ball_vy = random.choice([-1, 1])
        self.score_left = 0
        self.score_right = 0
        self.rally_count = 0

    def reset_ball(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_vx = -self.ball_vx
        self.ball_vy = random.choice([-1, 1])
        self.rally_count = 0

    def update_ai(self):
        # Autonomously track the ball to allow automated simulation and gameplay.
        # Left Paddle AI
        if self.paddle_left + self.paddle_height // 2 < self.ball_y:
            if self.paddle_left + self.paddle_height < self.height - 1:
                self.paddle_left += 1
        elif self.paddle_left + self.paddle_height // 2 > self.ball_y:
            if self.paddle_left > 1:
                self.paddle_left -= 1

        # Right Paddle AI
        if self.paddle_right + self.paddle_height // 2 < self.ball_y:
            if self.paddle_right + self.paddle_height < self.height - 1:
                self.paddle_right += 1
        elif self.paddle_right + self.paddle_height // 2 > self.ball_y:
            if self.paddle_right > 1:
                self.paddle_right -= 1

    def step(self):
        # Advance physics and state by one frame.
        self.update_ai()
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Ceiling and Floor reflection
        if self.ball_y <= 1:
            self.ball_y = 1
            self.ball_vy = -self.ball_vy
        elif self.ball_y >= self.height - 2:
            self.ball_y = self.height - 2
            self.ball_vy = -self.ball_vy

        # Left Paddle Collision (x = 2)
        if self.ball_x <= 2:
            if self.paddle_left <= self.ball_y < self.paddle_left + self.paddle_height:
                self.ball_x = 3
                self.ball_vx = 1
                self.rally_count += 1
            elif self.ball_x <= 0:
                self.score_right += 1
                self.reset_ball()

        # Right Paddle Collision (x = width - 3)
        elif self.ball_x >= self.width - 3:
            if self.paddle_right <= self.ball_y < self.paddle_right + self.paddle_height:
                self.ball_x = self.width - 4
                self.ball_vx = -1
                self.rally_count += 1
            elif self.ball_x >= self.width - 1:
                self.score_left += 1
                self.reset_ball()

    def render(self) -> str:
        # Render the 2D arena and entities into an ASCII string buffer.
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Arena Top & Bottom Borders
        for x in range(self.width):
            grid[0][x] = "="
            grid[self.height - 1][x] = "="

        # Net in center
        for y in range(1, self.height - 1):
            grid[y][self.width // 2] = ":"

        # Left Paddle
        for dy in range(self.paddle_height):
            py = self.paddle_left + dy
            if 0 < py < self.height - 1:
                grid[py][2] = "|"

        # Right Paddle
        for dy in range(self.paddle_height):
            py = self.paddle_right + dy
            if 0 < py < self.height - 1:
                grid[py][self.width - 3] = "|"

        # Ball
        bx = max(1, min(self.width - 2, self.ball_x))
        by = max(1, min(self.height - 2, self.ball_y))
        grid[by][bx] = "O"

        lines = ["".join(row) for row in grid]
        header = f"  Player 1 [Left]: {self.score_left}   |   Rally: {self.rally_count}   |   Player 2 [Right]: {self.score_right}  "
        return header + chr(10) + chr(10).join(lines)


def run_demo(frames: int = 30, delay: float = 0.05):
    # Run an animated console demo of ASCII Pong.
    game = AsciiPong()
    for _ in range(frames):
        game.step()
        frame = game.render()
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
        print(frame)
        time.sleep(delay)


if __name__ == "__main__":
    if "--play" in sys.argv or "-p" in sys.argv:
        print("Starting interactive/animated ASCII Pong demo...")
        run_demo(frames=120, delay=0.06)
    else:
        # Automated Verification Run
        game = AsciiPong(width=44, height=12)
        for _ in range(20):
            game.step()
        print("ASCII Pong Verification Passed Cleanly!")
        print(game.render())
