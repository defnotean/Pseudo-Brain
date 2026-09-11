# config.py - Game Configuration and Constants
from dataclasses import dataclass


@dataclass
class GameConfig:
    width: int = 54
    height: int = 16
    paddle_height: int = 4
    paddle_symbol: str = '|'
    ball_symbol: str = 'O'
    net_symbol: str = ':'
    border_symbol: str = '='
    fps_delay: float = 0.05
