# main.py - CLI Entry Point
import os
import sys
import time
from config import GameConfig
from engine import ArcadeEngine
from renderer import AsciiRenderer


def run_interactive(frames: int = 100):
    cfg = GameConfig()
    engine = ArcadeEngine(cfg)
    renderer = AsciiRenderer(cfg)
    for _ in range(frames):
        engine.step()
        frame = renderer.render_frame(engine)
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
        print(frame)
        time.sleep(cfg.fps_delay)


def verify_smoke() -> bool:
    cfg = GameConfig(width=32, height=10)
    engine = ArcadeEngine(cfg)
    renderer = AsciiRenderer(cfg)
    for _ in range(10):
        engine.step()
    frame = renderer.render_frame(engine)
    assert 'Left:' in frame and 'Right:' in frame
    print('ARCADE VERIFICATION SMOKE TEST: PASSED CLEANLY!')
    print(frame)
    return True


if __name__ == '__main__':
    if '--play' in sys.argv or '-p' in sys.argv:
        run_interactive()
    else:
        verify_smoke()
