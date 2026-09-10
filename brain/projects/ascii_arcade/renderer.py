# renderer.py - ASCII Terminal Rendering Engine
from config import GameConfig
from engine import ArcadeEngine


class AsciiRenderer:
    def __init__(self, config: GameConfig = None):
        self.cfg = config or GameConfig()

    def render_frame(self, engine: ArcadeEngine) -> str:
        w, h = self.cfg.width, self.cfg.height
        grid = [[' ' for _ in range(w)] for _ in range(h)]

        # Borders
        for x in range(w):
            grid[0][x] = self.cfg.border_symbol
            grid[h - 1][x] = self.cfg.border_symbol

        # Net
        for y in range(1, h - 1):
            grid[y][w // 2] = self.cfg.net_symbol

        # Left Paddle
        for dy in range(engine.paddle_left.height):
            py = engine.paddle_left.y + dy
            if 0 < py < h - 1:
                grid[py][2] = self.cfg.paddle_symbol

        # Right Paddle
        for dy in range(engine.paddle_right.height):
            py = engine.paddle_right.y + dy
            if 0 < py < h - 1:
                grid[py][w - 3] = self.cfg.paddle_symbol

        # Ball
        bx = max(1, min(w - 2, engine.ball.pos.x))
        by = max(1, min(h - 2, engine.ball.pos.y))
        grid[by][bx] = self.cfg.ball_symbol

        lines = [''.join(row) for row in grid]
        header = f'  Left: {engine.score_left}  |  Rally: {engine.rally_count}  |  Right: {engine.score_right}'
        return header + chr(10) + chr(10).join(lines)
