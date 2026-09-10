# tests/test_arcade.py - Cross-Module Integration Unit Tests
import sys
from pathlib import Path
root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from config import GameConfig
from models import Vector2D, Paddle, Ball
from engine import ArcadeEngine
from renderer import AsciiRenderer


def test_config_defaults():
    cfg = GameConfig()
    assert cfg.width > 0 and cfg.height > 0
    assert cfg.paddle_symbol == '|'
    assert cfg.ball_symbol == 'O'


def test_engine_physics_and_reflection():
    cfg = GameConfig(width=40, height=12)
    engine = ArcadeEngine(cfg)
    for _ in range(25):
        engine.step()
    assert 1 <= engine.ball.pos.x <= cfg.width - 2
    assert 1 <= engine.ball.pos.y <= cfg.height - 2
    assert engine.score_left >= 0 and engine.score_right >= 0


def test_renderer_buffer():
    cfg = GameConfig(width=30, height=8)
    engine = ArcadeEngine(cfg)
    renderer = AsciiRenderer(cfg)
    buffer = renderer.render_frame(engine)
    assert '=' in buffer
    assert ':' in buffer
    assert 'Left: 0' in buffer


if __name__ == '__main__':
    test_config_defaults()
    test_engine_physics_and_reflection()
    test_renderer_buffer()
    print('ALL CROSS-MODULE TESTS PASSED CLEANLY (3/3)!')
