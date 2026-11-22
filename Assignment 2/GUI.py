"""High-Resolution Anti-Aliased Pygame GUI for Connect 4 (Human vs. AI).

Features:
- Supersampled (3x -> smoothscale) anti-aliased 3D discs, recessed board cutouts
  with realistic shadows/bevels, and layered UI cards (zero blocky edges).
- Full PDF compliance: player color selection (Red or Yellow), configurable grid
  dimensions (rows >= 6, cols >= 7), depth K parameter, and algorithm selection
  (Minimax, Alpha-Beta, Expectiminimax).
- Bonus feature: Interactive, scrollable in-GUI Minimax / Expectiminimax Search
  Tree Viewer displaying node types, branch probabilities, pruning cutoffs, and
  evaluated heuristic values.
- Clean separation of concerns via `GameEngineAdapter` and `AIAgentAdapter`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
import sys
import threading
from typing import Any, Optional

import pygame

try:
    from board import board as ExternalBoard
except ImportError as exc:
    raise ImportError(
        "Could not import `board` from `board.py`. Ensure `board.py` is in the "
        "same directory as `gui.py`."
    ) from exc

try:
    import Algos as external_algos
except ImportError as exc:
    raise ImportError(
        "Could not import `Algos.py`. Ensure `Algos.py` is in the same "
        "directory as `gui.py`."
    ) from exc


# =============================================================================
# Constants & Visual Theme
# =============================================================================

WINDOW_WIDTH = 1220
WINDOW_HEIGHT = 820
WINDOW_TITLE = "Connect 4 — Adversarial Search AI"
FPS = 60

MIN_ROWS = 6
MAX_ROWS = 10
DEFAULT_ROWS = 6

MIN_COLS = 7
MAX_COLS = 11
DEFAULT_COLS = 7

HUMAN_PLAYER = 1
AI_PLAYER = 2
EMPTY_CELL = 0

ALGORITHM_OPTIONS = [
    ("minimax", "Minimax"),
    ("alpha_beta", "Alpha-Beta"),
    ("expectiminimax", "Expectiminimax"),
]

MIN_DEPTH_K = 1
MAX_DEPTH_K = 6
DEFAULT_DEPTH_K = 3

# Modern Slate & Studio Lighting Color Palette
COLOR_BG_TOP = (11, 17, 32)  # #0B1120
COLOR_BG_BOTTOM = (20, 30, 52)  # #141E34
COLOR_PANEL = (26, 36, 56)  # #1A2438 Card Surface
COLOR_PANEL_INSET = (17, 25, 42)  # #11192A Recessed Card
COLOR_BORDER = (51, 68, 99)  # #334463 Crisp Slate Border
COLOR_BORDER_BRIGHT = (76, 102, 148)

COLOR_TEXT_PRIMARY = (248, 250, 252)  # #F8FAFC
COLOR_TEXT_SECONDARY = (148, 163, 184)  # #94A3B8
COLOR_TEXT_ACCENT = (56, 189, 248)  # #38BDF8 Sky Cyan
COLOR_TEXT_SUCCESS = (74, 222, 128)  # #4ADE80 Emerald
COLOR_TEXT_WARN = (251, 191, 36)  # #FBBF24 Amber
COLOR_TEXT_DANGER = (248, 113, 113)  # #F87171 Coral

# Board Cabinet Colors
COLOR_BOARD_TOP = (37, 99, 235)  # #2563EB Royal Blue
COLOR_BOARD_BOTTOM = (23, 62, 168)  # #173EA8 Deep Sapphire
COLOR_BOARD_RIM = (96, 165, 250)  # #60A5FA Specular Edge
COLOR_BOARD_SHADOW = (6, 10, 20)
COLOR_SLOT_BACKING = (10, 15, 28)  # Dark backing behind transparent holes

# Disc Color Palettes ("RED" and "YELLOW")
DISC_THEMES = {
    "RED": {
        "name": "Red",
        "outer": (185, 28, 28),
        "mid": (239, 68, 68),
        "inner": (248, 113, 113),
        "rim_dark": (127, 29, 29),
        "rim_light": (254, 202, 202),
        "glow": (239, 68, 68),
    },
    "YELLOW": {
        "name": "Yellow",
        "outer": (202, 138, 4),
        "mid": (250, 204, 21),
        "inner": (253, 224, 71),
        "rim_dark": (133, 77, 14),
        "rim_light": (254, 249, 195),
        "glow": (250, 204, 21),
    },
}

# Interactive Button Palette
COLOR_BTN_PRIMARY = (37, 99, 235)
COLOR_BTN_PRIMARY_HOVER = (59, 130, 246)
COLOR_BTN_SECONDARY = (41, 55, 82)
COLOR_BTN_SECONDARY_HOVER = (58, 77, 112)
COLOR_BTN_DANGER = (190, 34, 34)
COLOR_BTN_DANGER_HOVER = (225, 48, 48)
COLOR_BTN_SELECTED = (14, 142, 212)
COLOR_BTN_DISABLED = (24, 33, 50)

# Physics Constants
GRAVITY_ACCEL = 3400.0
BOUNCE_FACTOR = 0.20


# =============================================================================
# Game States & Data Structures
# =============================================================================


class GameState(Enum):
    MENU = auto()
    PLAYING = auto()
    AI_THINKING = auto()
    CONFIRM_RESTART = auto()
    TREE_VIEW = auto()
    GAME_OVER = auto()


@dataclass
class AIDecision:
    """Normalized output returned by `AIAgentAdapter`."""

    column: Optional[int]
    heuristic_value: float = 0.0
    nodes_expanded: int = 0
    nodes_visited: int = 0
    leaves_evaluated: int = 0
    pruned_branches: int = 0
    time_taken: float = 0.0
    tree_trace: str = ""
    error: Optional[str] = None


@dataclass
class DiscAnimation:
    """Smooth gravity-driven disc drop animation with subtle bounce."""

    col: int
    target_row: int
    player: int
    x: float
    y: float
    target_y: float
    vy: float = 0.0
    bounced: bool = False
    finished: bool = False

    def update(self, dt: float) -> None:
        if self.finished:
            return
        self.vy += GRAVITY_ACCEL * dt
        self.y += self.vy * dt
        if self.y >= self.target_y:
            self.y = self.target_y
            if not self.bounced and self.vy > 280.0:
                self.vy = -self.vy * BOUNCE_FACTOR
                self.bounced = True
            else:
                self.vy = 0.0
                self.finished = True


# =============================================================================
# High-Resolution Supersampled Asset Generator (Eliminates Blocky Edges)
# =============================================================================


class AssetFactory:
    """Generates smooth, anti-aliased surfaces via 3x supersampling."""

    def __init__(self) -> None:
        self._disc_cache: dict[tuple[Any, ...], pygame.Surface] = {}
        self._board_cache: dict[tuple[int, ...], pygame.Surface] = {}
        self._bg_cache: Optional[pygame.Surface] = None

    def get_background(self, width: int, height: int) -> pygame.Surface:
        if self._bg_cache is not None and self._bg_cache.get_size() == (width, height):
            return self._bg_cache
        surf = pygame.Surface((width, height))
        for y in range(height):
            t = y / max(1, height - 1)
            r = int(COLOR_BG_TOP[0] + (COLOR_BG_BOTTOM[0] - COLOR_BG_TOP[0]) * t)
            g = int(COLOR_BG_TOP[1] + (COLOR_BG_BOTTOM[1] - COLOR_BG_TOP[1]) * t)
            b = int(COLOR_BG_TOP[2] + (COLOR_BG_BOTTOM[2] - COLOR_BG_TOP[2]) * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (width, y))
        self._bg_cache = surf
        return surf

    def get_disc(
        self,
        radius: int,
        color_key: str,
        ghost: bool = False,
        last_move: bool = False,
    ) -> pygame.Surface:
        """Render an anti-aliased, glossy 3D Connect 4 disc at 3x resolution."""
        key = (radius, color_key, ghost, last_move)
        if key in self._disc_cache:
            return self._disc_cache[key]

        scale = 3
        pad = 6 * scale
        hr = radius * scale
        size = hr * 2 + pad * 2
        hi = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = size // 2, size // 2
        theme = DISC_THEMES[color_key]

        if ghost:
            pygame.draw.circle(hi, (*theme["mid"], 52), (cx, cy), hr)
            pygame.draw.circle(
                hi, (*theme["rim_light"], 145), (cx, cy), hr, width=3 * scale
            )
            pygame.draw.circle(
                hi, (*theme["rim_light"], 90), (cx, cy), int(hr * 0.72), width=2 * scale
            )
            smooth = pygame.transform.smoothscale(hi, (size // scale, size // scale))
            self._disc_cache[key] = smooth
            return smooth

        # 1. Soft drop shadow beneath the disc
        pygame.draw.circle(hi, (4, 8, 18, 115), (cx, cy + 3 * scale), hr)

        # 2. Outer metallic rim
        pygame.draw.circle(hi, theme["rim_dark"], (cx, cy), hr)

        # 3. Radial gradient body (concentric layered shading)
        steps = 18
        for i in range(steps):
            t = i / (steps - 1)
            r_cur = int(hr * (0.95 - 0.55 * t))
            if t < 0.5:
                u = t / 0.5
                col = tuple(
                    int(theme["outer"][c] + (theme["mid"][c] - theme["outer"][c]) * u)
                    for c in range(3)
                )
            else:
                u = (t - 0.5) / 0.5
                col = tuple(
                    int(theme["mid"][c] + (theme["inner"][c] - theme["mid"][c]) * u)
                    for c in range(3)
                )
            offset_x = int(-hr * 0.14 * t)
            offset_y = int(-hr * 0.14 * t)
            pygame.draw.circle(hi, col, (cx + offset_x, cy + offset_y), max(1, r_cur))

        # 4. Recessed inner coin groove
        groove_r = int(hr * 0.72)
        pygame.draw.circle(hi, theme["rim_dark"], (cx, cy), groove_r, width=3 * scale)
        pygame.draw.circle(
            hi,
            (*theme["rim_light"], 110),
            (cx, cy + scale),
            groove_r - scale,
            width=scale,
        )

        # 5. Glossy specular crescent in upper-left quadrant
        spec_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        spec_rect = pygame.Rect(
            cx - int(hr * 0.62),
            cy - int(hr * 0.68),
            int(hr * 1.15),
            int(hr * 0.78),
        )
        pygame.draw.ellipse(spec_surf, (255, 255, 255, 72), spec_rect)
        # Mask specular highlight inside the disc circle
        mask = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), (cx, cy), int(hr * 0.88))
        spec_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        hi.blit(spec_surf, (0, 0))

        # 6. Optional crisp indicator ring for the last move played
        if last_move:
            pygame.draw.circle(
                hi, (255, 255, 255, 235), (cx, cy), int(hr * 0.34), width=3 * scale
            )
            pygame.draw.circle(hi, (15, 23, 42, 180), (cx, cy), int(hr * 0.25))

        smooth = pygame.transform.smoothscale(hi, (size // scale, size // scale))
        self._disc_cache[key] = smooth
        return smooth

    def get_board_face(
        self,
        board_w: int,
        board_h: int,
        rows: int,
        cols: int,
        cell_size: int,
        hole_radius: int,
    ) -> pygame.Surface:
        """Render a smooth, anti-aliased blue Connect 4 board with true transparent cutouts."""
        key = (board_w, board_h, rows, cols, cell_size, hole_radius)
        if key in self._board_cache:
            return self._board_cache[key]

        scale = 2
        pad = 12 * scale
        sw = board_w * scale + pad * 2
        sh = board_h * scale + pad * 2
        hi = pygame.Surface((sw, sh), pygame.SRCALPHA)

        body_rect = pygame.Rect(pad, pad, board_w * scale, board_h * scale)
        corner_r = 18 * scale

        # 1. Outer ambient drop shadow
        shadow_rect = body_rect.move(0, 6 * scale)
        pygame.draw.rect(
            hi, (*COLOR_BOARD_SHADOW, 175), shadow_rect, border_radius=corner_r
        )

        # 2. Vertical blue gradient cabinet body masked to rounded rectangle
        grad = pygame.Surface((sw, sh), pygame.SRCALPHA)
        for y in range(body_rect.top, body_rect.bottom):
            t = (y - body_rect.top) / max(1, body_rect.height - 1)
            col = tuple(
                int(
                    COLOR_BOARD_TOP[c]
                    + (COLOR_BOARD_BOTTOM[c] - COLOR_BOARD_TOP[c]) * t
                )
                for c in range(3)
            )
            pygame.draw.line(
                grad, (*col, 255), (body_rect.left, y), (body_rect.right, y)
            )

        round_mask = pygame.Surface((sw, sh), pygame.SRCALPHA)
        pygame.draw.rect(
            round_mask, (255, 255, 255, 255), body_rect, border_radius=corner_r
        )
        grad.blit(round_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        hi.blit(grad, (0, 0))

        # 3. Crisp top/outer cabinet bevel
        pygame.draw.rect(
            hi,
            (*COLOR_BOARD_RIM, 150),
            body_rect,
            width=2 * scale,
            border_radius=corner_r,
        )

        # 4. Punch out transparent circular slots + recessed bevel rings
        cutout_mask = pygame.Surface((sw, sh), pygame.SRCALPHA)
        cutout_mask.fill((255, 255, 255, 255))

        hr = hole_radius * scale
        cs = cell_size * scale
        for r in range(rows):
            for c in range(cols):
                cx = pad + int((c + 0.5) * cs)
                cy = pad + int((r + 0.5) * cs)
                # Transparent hole cutout
                pygame.draw.circle(cutout_mask, (255, 255, 255, 0), (cx, cy), hr)

        hi.blit(cutout_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

        # 5. Add subtle 3D inner socket rim around each hole after cutout
        for r in range(rows):
            for c in range(cols):
                cx = pad + int((c + 0.5) * cs)
                cy = pad + int((r + 0.5) * cs)
                pygame.draw.circle(
                    hi, (12, 26, 74, 210), (cx, cy), hr + scale, width=3 * scale
                )
                pygame.draw.circle(
                    hi,
                    (*COLOR_BOARD_RIM, 125),
                    (cx, cy + scale),
                    hr + 2 * scale,
                    width=scale,
                )

        smooth = pygame.transform.smoothscale(hi, (sw // scale, sh // scale))
        self._board_cache[key] = smooth
        return smooth


# =============================================================================
# Engine & AI Integration Adapters
# =============================================================================


class GameEngineAdapter:
    """Clean interface isolating the GUI from the external `board.py` engine."""

    def __init__(self, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> None:
        self.rows = max(MIN_ROWS, rows)
        self.cols = max(MIN_COLS, cols)
        self._board = ExternalBoard(l=self.rows, w=self.cols)

    def reset_board(
        self, rows: Optional[int] = None, cols: Optional[int] = None
    ) -> None:
        if rows is not None:
            self.rows = max(MIN_ROWS, rows)
        if cols is not None:
            self.cols = max(MIN_COLS, cols)
        self._board = ExternalBoard(l=self.rows, w=self.cols)

    def get_grid(self) -> list[list[int]]:
        return self._board.grid

    def get_valid_moves(self) -> list[int]:
        if hasattr(self._board, "get_valid_columns"):
            return sorted(self._board.get_valid_columns())
        if hasattr(self._board, "moves"):
            return [c for c, r in enumerate(self._board.moves()) if r != -1]
        return [c for c in range(self.cols) if self._board.grid[0][c] == EMPTY_CELL]

    def get_landing_row(self, column: int) -> int:
        if not (0 <= column < self.cols):
            return -1
        if hasattr(self._board, "moves"):
            return self._board.moves()[column]
        for r in range(self.rows - 1, -1, -1):
            if self._board.grid[r][column] == EMPTY_CELL:
                return r
        return -1

    def make_move(self, column: int, player: int) -> bool:
        if column not in self.get_valid_moves():
            return False
        if hasattr(self._board, "play_move"):
            return bool(self._board.play_move(player, column))
        if hasattr(self._board, "drop_piece"):
            self._board.drop_piece(column, player)
            return True
        raise AttributeError("Board engine must implement `play_move` or `drop_piece`.")

    def is_board_full(self) -> bool:
        if hasattr(self._board, "isFull"):
            return bool(self._board.isFull())
        if hasattr(self._board, "is_full"):
            return bool(self._board.is_full())
        return len(self.get_valid_moves()) == 0

    def get_scores(self) -> tuple[int, int]:
        if hasattr(self._board, "count_connect_fours"):
            return (
                int(self._board.count_connect_fours(HUMAN_PLAYER)),
                int(self._board.count_connect_fours(AI_PLAYER)),
            )
        return (0, 0)

    def check_winner(self) -> Optional[int]:
        if not self.is_board_full():
            return None
        human_score, ai_score = self.get_scores()
        if human_score > ai_score:
            return HUMAN_PLAYER
        if ai_score > human_score:
            return AI_PLAYER
        return 0

    def evaluate_board(self) -> float:
        if hasattr(external_algos, "evaluate"):
            try:
                return float(external_algos.evaluate(self._board))
            except Exception:
                pass
        if hasattr(external_algos, "default_evaluate"):
            return float(external_algos.default_evaluate(self._board))
        human_score, ai_score = self.get_scores()
        return float(ai_score - human_score)

    def get_heuristic_breakdown(self) -> dict[str, float]:
        """Return component metrics if available on the board engine."""
        fours_diff = 0
        threats_diff = 0.0
        forks_diff = 0
        if hasattr(self._board, "count_connect_fours"):
            fours_diff = self._board.count_connect_fours(
                AI_PLAYER
            ) - self._board.count_connect_fours(HUMAN_PLAYER)
        if hasattr(self._board, "count_threats"):
            threats_diff = self._board.count_threats(
                AI_PLAYER
            ) - 1.1 * self._board.count_threats(HUMAN_PLAYER)
        if hasattr(self._board, "count_forks"):
            forks_diff = self._board.count_forks(AI_PLAYER) - self._board.count_forks(
                HUMAN_PLAYER
            )
        return {
            "total": self.evaluate_board(),
            "fours": fours_diff,
            "threats": round(threats_diff, 2),
            "forks": forks_diff,
        }

    def resolve_ai_landing_column(self, chosen_col: int, algorithm: str) -> int:
        if algorithm == "expectiminimax" and hasattr(
            external_algos, "sample_landing_column"
        ):
            return int(external_algos.sample_landing_column(self._board, chosen_col))
        return chosen_col

    def clone_raw_board(self) -> Any:
        return deepcopy(self._board)


class AIAgentAdapter:
    """Clean interface isolating the GUI from the external AI search module."""

    def __init__(self, evaluate_fn: Optional[Any] = None) -> None:
        self.evaluate_fn = evaluate_fn

    def get_best_move(
        self, board_snapshot: Any, algorithm: str, depth_k: int
    ) -> AIDecision:
        try:
            if not hasattr(external_algos, "get_best_move"):
                raise AttributeError("`Algos.py` does not define `get_best_move`.")

            kwargs: dict[str, Any] = {
                "print_tree": True,
                "collect_trace": True,
                "stochastic_players": (AI_PLAYER,),
            }
            if self.evaluate_fn is not None:
                kwargs["evaluate_fn"] = self.evaluate_fn

            try:
                raw_result = external_algos.get_best_move(
                    board_snapshot, algorithm, depth_k, **kwargs
                )
            except TypeError:
                raw_result = external_algos.get_best_move(board_snapshot, depth_k)

            if isinstance(raw_result, dict):
                return AIDecision(
                    column=raw_result.get("best_col"),
                    heuristic_value=float(raw_result.get("heuristic_value", 0.0)),
                    nodes_expanded=int(raw_result.get("nodes_expanded", 0)),
                    nodes_visited=int(raw_result.get("nodes_visited", 0)),
                    leaves_evaluated=int(raw_result.get("leaves_evaluated", 0)),
                    pruned_branches=int(raw_result.get("pruned_branches", 0)),
                    time_taken=float(raw_result.get("time_taken", 0.0)),
                    tree_trace=str(raw_result.get("tree_trace", "")),
                )
            return AIDecision(column=int(raw_result))

        except Exception as exc:
            return AIDecision(column=None, error=str(exc))


# =============================================================================
# Reusable UI Widgets
# =============================================================================


def draw_smooth_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fill: tuple[int, int, int],
    border: tuple[int, int, int],
    radius: int = 14,
    shadow: bool = True,
) -> None:
    """Draw a rounded card with subtle elevation shadow and crisp border."""
    if shadow:
        sh_surf = pygame.Surface((rect.width + 12, rect.height + 12), pygame.SRCALPHA)
        pygame.draw.rect(
            sh_surf,
            (4, 8, 18, 120),
            pygame.Rect(6, 9, rect.width, rect.height),
            border_radius=radius,
        )
        surface.blit(sh_surf, (rect.x - 6, rect.y - 6))
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, width=1, border_radius=radius)


class Button:
    """Modern rounded interactive button with subtle elevation and hover states."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font,
        base_color: tuple[int, int, int] = COLOR_BTN_PRIMARY,
        hover_color: tuple[int, int, int] = COLOR_BTN_PRIMARY_HOVER,
        text_color: tuple[int, int, int] = COLOR_TEXT_PRIMARY,
        selected_color: tuple[int, int, int] = COLOR_BTN_SELECTED,
        border_radius: int = 10,
    ) -> None:
        self.rect = rect
        self.text = text
        self.font = font
        self.base_color = base_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.selected_color = selected_color
        self.border_radius = border_radius
        self.enabled = True
        self.selected = False

    def is_hovered(self, mouse_pos: tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(mouse_pos)

    def handle_click(self, mouse_pos: tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(mouse_pos)

    def draw(self, surface: pygame.Surface, mouse_pos: tuple[int, int]) -> None:
        hovered = self.is_hovered(mouse_pos)
        if not self.enabled:
            fill = COLOR_BTN_DISABLED
            label_color = COLOR_TEXT_SECONDARY
            border_col = COLOR_BORDER
        elif self.selected:
            fill = self.selected_color
            label_color = self.text_color
            border_col = COLOR_TEXT_ACCENT
        elif hovered:
            fill = self.hover_color
            label_color = self.text_color
            border_col = COLOR_BORDER_BRIGHT
        else:
            fill = self.base_color
            label_color = self.text_color
            border_col = COLOR_BORDER

        draw_rect = self.rect.move(0, -1 if hovered else 0)
        pygame.draw.rect(surface, fill, draw_rect, border_radius=self.border_radius)
        pygame.draw.rect(
            surface,
            border_col,
            draw_rect,
            width=2 if self.selected else 1,
            border_radius=self.border_radius,
        )

        text_surf = self.font.render(self.text, True, label_color)
        surface.blit(text_surf, text_surf.get_rect(center=draw_rect.center))


# =============================================================================
# Main GUI Application Controller
# =============================================================================


class Connect4GUI:
    """Manages Pygame rendering, event routing, animations, and state transitions."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.assets = AssetFactory()

        # Typography (clean sans-serif stack)
        font_family = pygame.font.match_font(
            "segoeui,inter,sfprodisplay,arial,helvetica"
        )
        self.font_title = pygame.font.Font(font_family, 52)
        self.font_title.set_bold(True)
        self.font_subtitle = pygame.font.Font(font_family, 19)
        self.font_heading = pygame.font.Font(font_family, 23)
        self.font_heading.set_bold(True)
        self.font_body = pygame.font.Font(font_family, 17)
        self.font_body_bold = pygame.font.Font(font_family, 17)
        self.font_body_bold.set_bold(True)
        self.font_small = pygame.font.Font(font_family, 14)
        self.font_btn = pygame.font.Font(font_family, 17)
        self.font_btn.set_bold(True)

        mono_family = pygame.font.match_font(
            "consolas,cascadiamono,couriernew,monospace"
        )
        self.font_mono = pygame.font.Font(mono_family, 14)

        # Match Configuration (Main Menu)
        self.selected_algo_idx = 1  # Default: Alpha-Beta
        self.human_color_key = "RED"
        self.config_rows = DEFAULT_ROWS
        self.config_cols = DEFAULT_COLS
        self.depth_k = DEFAULT_DEPTH_K

        # Engine & AI Adapters
        self.engine = GameEngineAdapter(rows=self.config_rows, cols=self.config_cols)
        eval_fn = getattr(external_algos, "evaluate", None)
        self.ai_agent = AIAgentAdapter(evaluate_fn=eval_fn)

        # Runtime State
        self.state = GameState.MENU
        self.previous_state = GameState.PLAYING
        self.current_turn = HUMAN_PLAYER
        self.hovered_col: Optional[int] = None
        self.active_animation: Optional[DiscAnimation] = None
        self.last_move_cell: Optional[tuple[int, int]] = None
        self.moves_played = 0
        self.status_message = "Click any column to drop your disc."

        # Telemetry & Heuristic Display
        self.heuristic_info = self.engine.get_heuristic_breakdown()
        self.last_ai_decision = AIDecision(column=None)
        self.winner: Optional[int] = None

        # Tree Viewer Scroll State (Bonus Feature)
        self.tree_scroll_y = 0
        self.tree_scroll_x = 0

        # Background AI Thread State
        self._ai_lock = threading.Lock()
        self._ai_thread: Optional[threading.Thread] = None
        self._pending_ai_decision: Optional[AIDecision] = None

        # Board Geometry Cache
        self.board_rect = pygame.Rect(42, 120, 680, 620)
        self.cell_size = 90
        self.disc_radius = 38
        self._recompute_board_geometry()

        # Build UI Controls
        self._init_menu_widgets()
        self._init_hud_widgets()
        self._init_modal_widgets()

    @property
    def ai_color_key(self) -> str:
        return "YELLOW" if self.human_color_key == "RED" else "RED"

    @property
    def current_algorithm_key(self) -> str:
        return ALGORITHM_OPTIONS[self.selected_algo_idx][0]

    @property
    def current_algorithm_label(self) -> str:
        return ALGORITHM_OPTIONS[self.selected_algo_idx][1]

    # -------------------------------------------------------------------------
    # Layout & Widget Setup
    # -------------------------------------------------------------------------

    def _recompute_board_geometry(self) -> None:
        max_board_w = 740
        max_board_h = 640
        origin_x = 34
        origin_y = 112

        cell_w = max_board_w // self.engine.cols
        cell_h = max_board_h // (self.engine.rows + 1)
        self.cell_size = max(44, min(cell_w, cell_h))
        self.disc_radius = int(self.cell_size * 0.40)

        total_w = self.cell_size * self.engine.cols
        total_h = self.cell_size * self.engine.rows
        board_x = origin_x + (max_board_w - total_w) // 2
        board_y = (
            origin_y + self.cell_size + (max_board_h - self.cell_size - total_h) // 2
        )
        self.board_rect = pygame.Rect(board_x, board_y, total_w, total_h)

    def _init_menu_widgets(self) -> None:
        cx = WINDOW_WIDTH // 2

        # 1. Player Color Picker (PDF Sec 1: "players first choose a color")
        self.btn_color_red = Button(
            pygame.Rect(cx - 210, 240, 195, 44),
            "Play as Red",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )
        self.btn_color_yellow = Button(
            pygame.Rect(cx + 15, 240, 195, 44),
            "Play as Yellow",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )

        # 2. Algorithm Selector
        self.menu_algo_buttons: list[Button] = []
        btn_w, btn_h, gap = 185, 44, 14
        total_w = len(ALGORITHM_OPTIONS) * btn_w + (len(ALGORITHM_OPTIONS) - 1) * gap
        start_x = cx - total_w // 2
        for idx, (_, label) in enumerate(ALGORITHM_OPTIONS):
            rect = pygame.Rect(start_x + idx * (btn_w + gap), 336, btn_w, btn_h)
            self.menu_algo_buttons.append(
                Button(
                    rect,
                    label,
                    self.font_btn,
                    base_color=COLOR_BTN_SECONDARY,
                    hover_color=COLOR_BTN_SECONDARY_HOVER,
                )
            )

        # 3. Rows (>=6), Columns (>=7), and Depth K Steppers
        self.btn_rows_minus = Button(
            pygame.Rect(cx - 265, 442, 40, 40),
            "-",
            self.font_heading,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )
        self.btn_rows_plus = Button(
            pygame.Rect(cx - 155, 442, 40, 40),
            "+",
            self.font_heading,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )

        self.btn_cols_minus = Button(
            pygame.Rect(cx - 75, 442, 40, 40),
            "-",
            self.font_heading,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )
        self.btn_cols_plus = Button(
            pygame.Rect(cx + 35, 442, 40, 40),
            "+",
            self.font_heading,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )

        self.btn_depth_minus = Button(
            pygame.Rect(cx + 115, 442, 40, 40),
            "-",
            self.font_heading,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )
        self.btn_depth_plus = Button(
            pygame.Rect(cx + 225, 442, 40, 40),
            "+",
            self.font_heading,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )

        # 4. Primary Menu Actions
        self.btn_start_game = Button(
            pygame.Rect(cx - 165, 535, 330, 54),
            "Start Game",
            self.font_btn,
            base_color=COLOR_BTN_PRIMARY,
            hover_color=COLOR_BTN_PRIMARY_HOVER,
        )
        self.btn_exit_game = Button(
            pygame.Rect(cx - 165, 604, 330, 46),
            "Exit",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_DANGER_HOVER,
        )

    def _init_hud_widgets(self) -> None:
        panel_x = 804
        panel_w = WINDOW_WIDTH - panel_x - 32
        self.hud_panel_rect = pygame.Rect(panel_x, 28, panel_w, WINDOW_HEIGHT - 56)

        btn_y = WINDOW_HEIGHT - 196
        self.btn_view_tree = Button(
            pygame.Rect(panel_x + 20, btn_y, panel_w - 40, 44),
            "View Search Tree (Bonus)",
            self.font_btn,
            base_color=COLOR_BTN_PRIMARY,
            hover_color=COLOR_BTN_PRIMARY_HOVER,
        )
        self.btn_restart = Button(
            pygame.Rect(panel_x + 20, btn_y + 54, panel_w - 40, 42),
            "Restart Game",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )
        self.btn_main_menu = Button(
            pygame.Rect(panel_x + 20, btn_y + 106, panel_w - 40, 42),
            "Main Menu",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_DANGER_HOVER,
        )

    def _init_modal_widgets(self) -> None:
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        self.modal_rect = pygame.Rect(cx - 240, cy - 130, 480, 260)
        self.btn_confirm_yes = Button(
            pygame.Rect(cx - 190, cy + 48, 175, 46),
            "Yes, Reset",
            self.font_btn,
            base_color=COLOR_BTN_DANGER,
            hover_color=COLOR_BTN_DANGER_HOVER,
        )
        self.btn_confirm_no = Button(
            pygame.Rect(cx + 15, cy + 48, 175, 46),
            "Cancel",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )

        # Tree Viewer Modal (Bonus)
        self.tree_modal_rect = pygame.Rect(
            70, 52, WINDOW_WIDTH - 140, WINDOW_HEIGHT - 104
        )
        self.btn_close_tree = Button(
            pygame.Rect(
                self.tree_modal_rect.right - 140, self.tree_modal_rect.y + 18, 116, 38
            ),
            "Close (Esc)",
            self.font_small,
            base_color=COLOR_BTN_DANGER,
            hover_color=COLOR_BTN_DANGER_HOVER,
        )

        # End-Game Overlay
        self.gameover_card_rect = pygame.Rect(cx - 260, cy - 175, 520, 350)
        self.btn_play_again = Button(
            pygame.Rect(cx - 215, cy + 98, 200, 48),
            "Play Again",
            self.font_btn,
            base_color=COLOR_BTN_PRIMARY,
            hover_color=COLOR_BTN_PRIMARY_HOVER,
        )
        self.btn_gameover_menu = Button(
            pygame.Rect(cx + 15, cy + 98, 200, 48),
            "Main Menu",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )

    # -------------------------------------------------------------------------
    # Game Lifecycle & Turn Management
    # -------------------------------------------------------------------------

    def _start_new_game(self) -> None:
        self.engine.reset_board(rows=self.config_rows, cols=self.config_cols)
        self._recompute_board_geometry()

        with self._ai_lock:
            self._pending_ai_decision = None

        self.state = GameState.PLAYING
        self.current_turn = HUMAN_PLAYER
        self.active_animation = None
        self.last_move_cell = None
        self.moves_played = 0
        self.winner = None
        self.heuristic_info = self.engine.get_heuristic_breakdown()
        self.last_ai_decision = AIDecision(column=None)
        color_name = DISC_THEMES[self.human_color_key]["name"]
        self.status_message = (
            f"Your turn! Click a column to drop your {color_name} disc."
        )

    def _slot_center(self, row: int, col: int) -> tuple[float, float]:
        cx = self.board_rect.x + col * self.cell_size + self.cell_size / 2.0
        cy = self.board_rect.y + row * self.cell_size + self.cell_size / 2.0
        return cx, cy

    def _column_from_mouse(self, mouse_pos: tuple[int, int]) -> Optional[int]:
        mx, my = mouse_pos
        hover_top = self.board_rect.y - self.cell_size
        if not (self.board_rect.left <= mx < self.board_rect.right):
            return None
        if not (hover_top <= my <= self.board_rect.bottom):
            return None
        col = (mx - self.board_rect.left) // self.cell_size
        if 0 <= col < self.engine.cols:
            return int(col)
        return None

    def _begin_disc_drop(self, col: int, player: int) -> bool:
        target_row = self.engine.get_landing_row(col)
        if target_row == -1:
            return False
        cx, target_y = self._slot_center(target_row, col)
        start_y = self.board_rect.y - self.cell_size * 0.55
        self.active_animation = DiscAnimation(
            col=col,
            target_row=target_row,
            player=player,
            x=cx,
            y=start_y,
            target_y=target_y,
        )
        return True

    def _handle_human_column_click(self, col: int) -> None:
        if self.state != GameState.PLAYING:
            return
        if self.current_turn != HUMAN_PLAYER or self.active_animation is not None:
            return

        if col not in self.engine.get_valid_moves():
            self.status_message = f"Column {col + 1} is full! Select another column."
            return

        if self._begin_disc_drop(col, HUMAN_PLAYER):
            color_name = DISC_THEMES[self.human_color_key]["name"]
            self.status_message = f"Dropping {color_name} disc into Column {col + 1}..."

    def _commit_animated_move(self, anim: DiscAnimation) -> None:
        applied = self.engine.make_move(anim.col, anim.player)
        if not applied:
            self.status_message = f"Move rejected in Column {anim.col + 1}."
            return

        self.last_move_cell = (anim.target_row, anim.col)
        self.moves_played += 1
        self.heuristic_info = self.engine.get_heuristic_breakdown()

        outcome = self.engine.check_winner()
        if outcome is not None:
            self.winner = outcome
            self.state = GameState.GAME_OVER
            h_score, a_score = self.engine.get_scores()
            if outcome == HUMAN_PLAYER:
                self.status_message = f"Victory! You won {h_score} to {a_score}."
            elif outcome == AI_PLAYER:
                self.status_message = f"AI wins {a_score} to {h_score}!"
            else:
                self.status_message = f"Draw game! Both players scored {h_score}."
            return

        if anim.player == HUMAN_PLAYER:
            self.current_turn = AI_PLAYER
            self._start_ai_turn()
        else:
            self.current_turn = HUMAN_PLAYER
            self.state = GameState.PLAYING
            self.status_message = "Your turn! Choose a column."

    def _start_ai_turn(self) -> None:
        self.state = GameState.AI_THINKING
        self.status_message = (
            f"AI is evaluating ({self.current_algorithm_label}, K={self.depth_k})..."
        )

        snapshot = self.engine.clone_raw_board()
        algorithm = self.current_algorithm_key
        depth = self.depth_k

        with self._ai_lock:
            self._pending_ai_decision = None

        def worker() -> None:
            decision = self.ai_agent.get_best_move(snapshot, algorithm, depth)
            with self._ai_lock:
                self._pending_ai_decision = decision

        self._ai_thread = threading.Thread(target=worker, daemon=True)
        self._ai_thread.start()

    def _poll_ai_result(self) -> None:
        if self.state != GameState.AI_THINKING or self.active_animation is not None:
            return

        with self._ai_lock:
            decision = self._pending_ai_decision
            if decision is None:
                return
            self._pending_ai_decision = None

        self.last_ai_decision = decision
        valid_moves = self.engine.get_valid_moves()
        if not valid_moves:
            self.winner = self.engine.check_winner()
            self.state = GameState.GAME_OVER
            return

        chosen_col = decision.column
        if decision.error or chosen_col not in valid_moves:
            chosen_col = valid_moves[0]
            err_note = f" ({decision.error})" if decision.error else ""
            self.status_message = f"AI fallback to Column {chosen_col + 1}{err_note}."

        actual_col = self.engine.resolve_ai_landing_column(
            chosen_col, self.current_algorithm_key
        )
        if actual_col not in valid_moves:
            actual_col = chosen_col

        ai_color_name = DISC_THEMES[self.ai_color_key]["name"]
        if actual_col != chosen_col:
            self.status_message = (
                f"Expectiminimax: AI aimed at Col {chosen_col + 1}, "
                f"slipped into Col {actual_col + 1}!"
            )
        else:
            self.status_message = (
                f"AI drops {ai_color_name} disc in Column {actual_col + 1}."
            )

        self._begin_disc_drop(actual_col, AI_PLAYER)

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def handle_events(self) -> None:
        mouse_pos = pygame.mouse.get_pos()

        if self.state in (GameState.PLAYING, GameState.AI_THINKING):
            self.hovered_col = self._column_from_mouse(mouse_pos)
        else:
            self.hovered_col = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.MOUSEWHEEL and self.state == GameState.TREE_VIEW:
                self.tree_scroll_y = max(0, self.tree_scroll_y - event.y * 36)
                self.tree_scroll_x = max(0, self.tree_scroll_x - event.x * 36)

            if event.type == pygame.KEYDOWN:
                if self.state == GameState.TREE_VIEW:
                    if event.key in (pygame.K_ESCAPE, pygame.K_t):
                        self.state = self.previous_state
                    elif event.key == pygame.K_DOWN:
                        self.tree_scroll_y += 40
                    elif event.key == pygame.K_UP:
                        self.tree_scroll_y = max(0, self.tree_scroll_y - 40)
                    elif event.key == pygame.K_RIGHT:
                        self.tree_scroll_x += 40
                    elif event.key == pygame.K_LEFT:
                        self.tree_scroll_x = max(0, self.tree_scroll_x - 40)
                elif event.key == pygame.K_ESCAPE:
                    if self.state == GameState.CONFIRM_RESTART:
                        self.state = self.previous_state
                    elif self.state in (GameState.PLAYING, GameState.GAME_OVER):
                        self._request_navigation(GameState.MENU)
                elif event.key == pygame.K_t and self.state in (
                    GameState.PLAYING,
                    GameState.GAME_OVER,
                ):
                    if self.last_ai_decision.tree_trace:
                        self._open_tree_viewer()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_click(event.pos)

    def _open_tree_viewer(self) -> None:
        self.previous_state = self.state
        self.tree_scroll_y = 0
        self.tree_scroll_x = 0
        self.state = GameState.TREE_VIEW

    def _request_navigation(self, target_action: str | GameState) -> None:
        if self.state == GameState.AI_THINKING or self.active_animation is not None:
            self.status_message = (
                "Wait for the current turn to finish before resetting."
            )
            return

        if self.moves_played > 0 and self.state == GameState.PLAYING:
            self.previous_state = self.state
            self._pending_nav_action = target_action
            self.state = GameState.CONFIRM_RESTART
        else:
            if target_action == GameState.MENU:
                self.state = GameState.MENU
            else:
                self._start_new_game()

    def _handle_mouse_click(self, pos: tuple[int, int]) -> None:
        if self.state == GameState.MENU:
            if self.btn_color_red.handle_click(pos):
                self.human_color_key = "RED"
                return
            if self.btn_color_yellow.handle_click(pos):
                self.human_color_key = "YELLOW"
                return
            for idx, btn in enumerate(self.menu_algo_buttons):
                if btn.handle_click(pos):
                    self.selected_algo_idx = idx
                    return
            if self.btn_rows_minus.handle_click(pos):
                self.config_rows = max(MIN_ROWS, self.config_rows - 1)
                return
            if self.btn_rows_plus.handle_click(pos):
                self.config_rows = min(MAX_ROWS, self.config_rows + 1)
                return
            if self.btn_cols_minus.handle_click(pos):
                self.config_cols = max(MIN_COLS, self.config_cols - 1)
                return
            if self.btn_cols_plus.handle_click(pos):
                self.config_cols = min(MAX_COLS, self.config_cols + 1)
                return
            if self.btn_depth_minus.handle_click(pos):
                self.depth_k = max(MIN_DEPTH_K, self.depth_k - 1)
                return
            if self.btn_depth_plus.handle_click(pos):
                self.depth_k = min(MAX_DEPTH_K, self.depth_k + 1)
                return
            if self.btn_start_game.handle_click(pos):
                self._start_new_game()
                return
            if self.btn_exit_game.handle_click(pos):
                self.running = False
                return

        elif self.state == GameState.PLAYING:
            if self.btn_view_tree.handle_click(pos):
                self._open_tree_viewer()
                return
            if self.btn_restart.handle_click(pos):
                self._request_navigation("RESTART")
                return
            if self.btn_main_menu.handle_click(pos):
                self._request_navigation(GameState.MENU)
                return

            col = self._column_from_mouse(pos)
            if col is not None:
                self._handle_human_column_click(col)

        elif self.state == GameState.AI_THINKING:
            self.status_message = "AI is thinking — please wait..."

        elif self.state == GameState.TREE_VIEW:
            if self.btn_close_tree.handle_click(
                pos
            ) or not self.tree_modal_rect.collidepoint(pos):
                self.state = self.previous_state

        elif self.state == GameState.CONFIRM_RESTART:
            if self.btn_confirm_yes.handle_click(pos):
                action = getattr(self, "_pending_nav_action", "RESTART")
                if action == GameState.MENU:
                    self.state = GameState.MENU
                else:
                    self._start_new_game()
            elif self.btn_confirm_no.handle_click(pos):
                self.state = self.previous_state

        elif self.state == GameState.GAME_OVER:
            if self.btn_view_tree.handle_click(pos):
                self._open_tree_viewer()
            elif self.btn_play_again.handle_click(pos) or self.btn_restart.handle_click(
                pos
            ):
                self._start_new_game()
            elif self.btn_gameover_menu.handle_click(
                pos
            ) or self.btn_main_menu.handle_click(pos):
                self.state = GameState.MENU

    # -------------------------------------------------------------------------
    # Frame Update
    # -------------------------------------------------------------------------

    def update(self, dt: float) -> None:
        if self.active_animation is not None:
            self.active_animation.update(dt)
            if self.active_animation.finished:
                finished_anim = self.active_animation
                self.active_animation = None
                self._commit_animated_move(finished_anim)
            return

        if self.state == GameState.AI_THINKING:
            self._poll_ai_result()

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def render(self) -> None:
        bg = self.assets.get_background(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.screen.blit(bg, (0, 0))
        mouse_pos = pygame.mouse.get_pos()

        if self.state == GameState.MENU:
            self._render_menu(mouse_pos)
        else:
            self._render_game_screen(mouse_pos)
            if self.state == GameState.CONFIRM_RESTART:
                self._render_confirm_modal(mouse_pos)
            elif self.state == GameState.TREE_VIEW:
                self._render_tree_viewer(mouse_pos)
            elif self.state == GameState.GAME_OVER:
                self._render_gameover_overlay(mouse_pos)

        pygame.display.flip()

    def _blit_disc(
        self,
        cx: float,
        cy: float,
        radius: int,
        player: int,
        ghost: bool = False,
        last_move: bool = False,
    ) -> None:
        color_key = (
            self.human_color_key if player == HUMAN_PLAYER else self.ai_color_key
        )
        surf = self.assets.get_disc(radius, color_key, ghost=ghost, last_move=last_move)
        rect = surf.get_rect(center=(int(cx), int(cy)))
        self.screen.blit(surf, rect)

    def _render_menu(self, mouse_pos: tuple[int, int]) -> None:
        cx = WINDOW_WIDTH // 2
        card = pygame.Rect(cx - 350, 48, 700, 724)
        draw_smooth_card(self.screen, card, COLOR_PANEL, COLOR_BORDER, radius=20)

        # Anti-aliased header discs
        self._blit_disc(cx - 220, 114, 24, HUMAN_PLAYER)
        self._blit_disc(cx + 220, 114, 24, AI_PLAYER)

        title_surf = self.font_title.render("CONNECT 4", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title_surf, title_surf.get_rect(center=(cx, 114)))

        sub_surf = self.font_subtitle.render(
            "Human vs. AI  •  Full-Board Connected-4 Scoring",
            True,
            COLOR_TEXT_SECONDARY,
        )
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(cx, 162)))

        # 1. Color Selection
        c_lbl = self.font_body_bold.render(
            "1. Choose Your Disc Color", True, COLOR_TEXT_ACCENT
        )
        self.screen.blit(c_lbl, c_lbl.get_rect(center=(cx, 214)))

        self.btn_color_red.selected = self.human_color_key == "RED"
        self.btn_color_yellow.selected = self.human_color_key == "YELLOW"
        self.btn_color_red.draw(self.screen, mouse_pos)
        self.btn_color_yellow.draw(self.screen, mouse_pos)
        self._blit_disc(
            self.btn_color_red.rect.left + 26,
            self.btn_color_red.rect.centery,
            11,
            HUMAN_PLAYER if self.human_color_key == "RED" else AI_PLAYER,
        )
        self._blit_disc(
            self.btn_color_yellow.rect.left + 26,
            self.btn_color_yellow.rect.centery,
            11,
            HUMAN_PLAYER if self.human_color_key == "YELLOW" else AI_PLAYER,
        )

        # 2. Algorithm Selection
        a_lbl = self.font_body_bold.render(
            "2. Select AI Search Algorithm", True, COLOR_TEXT_ACCENT
        )
        self.screen.blit(a_lbl, a_lbl.get_rect(center=(cx, 310)))
        for idx, btn in enumerate(self.menu_algo_buttons):
            btn.selected = idx == self.selected_algo_idx
            btn.draw(self.screen, mouse_pos)

        # 3. Grid Dimensions & Search Depth K
        g_lbl = self.font_body_bold.render(
            "3. Configure Board Dimensions & Depth (K)", True, COLOR_TEXT_ACCENT
        )
        self.screen.blit(g_lbl, g_lbl.get_rect(center=(cx, 408)))

        self.btn_rows_minus.enabled = self.config_rows > MIN_ROWS
        self.btn_rows_plus.enabled = self.config_rows < MAX_ROWS
        self.btn_cols_minus.enabled = self.config_cols > MIN_COLS
        self.btn_cols_plus.enabled = self.config_cols < MAX_COLS
        self.btn_depth_minus.enabled = self.depth_k > MIN_DEPTH_K
        self.btn_depth_plus.enabled = self.depth_k < MAX_DEPTH_K

        stepper_groups = [
            (
                "Rows (>=6)",
                self.config_rows,
                cx - 190,
                self.btn_rows_minus,
                self.btn_rows_plus,
            ),
            (
                "Cols (>=7)",
                self.config_cols,
                cx,
                self.btn_cols_minus,
                self.btn_cols_plus,
            ),
            (
                "Depth K",
                self.depth_k,
                cx + 190,
                self.btn_depth_minus,
                self.btn_depth_plus,
            ),
        ]
        for label, val, group_cx, b_minus, b_plus in stepper_groups:
            cap = self.font_small.render(label, True, COLOR_TEXT_SECONDARY)
            self.screen.blit(cap, cap.get_rect(center=(group_cx, 428)))
            b_minus.draw(self.screen, mouse_pos)
            b_plus.draw(self.screen, mouse_pos)
            val_box = pygame.Rect(group_cx - 32, 442, 64, 40)
            pygame.draw.rect(self.screen, COLOR_PANEL_INSET, val_box, border_radius=8)
            pygame.draw.rect(
                self.screen, COLOR_BORDER, val_box, width=1, border_radius=8
            )
            vs = self.font_heading.render(str(val), True, COLOR_TEXT_PRIMARY)
            self.screen.blit(vs, vs.get_rect(center=val_box.center))

        # 4. Action Buttons
        self.btn_start_game.draw(self.screen, mouse_pos)
        self.btn_exit_game.draw(self.screen, mouse_pos)

    def _render_game_screen(self, mouse_pos: tuple[int, int]) -> None:
        # Header Bar
        title_s = self.font_heading.render("CONNECT 4", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title_s, (self.board_rect.x, 28))

        sub_s = self.font_small.render(
            f"Grid: {self.engine.rows}x{self.engine.cols}  •  Board plays until full; highest Connected-4s wins",
            True,
            COLOR_TEXT_SECONDARY,
        )
        self.screen.blit(sub_s, (self.board_rect.x, 58))

        # Dark backing plate behind the transparent cutouts
        backing_rect = self.board_rect.inflate(-6, -6)
        pygame.draw.rect(
            self.screen, COLOR_SLOT_BACKING, backing_rect, border_radius=14
        )

        # Hover Column Highlight + Floating Disc + Ghost Target Preview
        valid_moves = self.engine.get_valid_moves()
        if (
            self.state == GameState.PLAYING
            and self.current_turn == HUMAN_PLAYER
            and self.active_animation is None
            and self.hovered_col is not None
        ):
            col_x = self.board_rect.x + self.hovered_col * self.cell_size
            beam = pygame.Surface(
                (self.cell_size, self.board_rect.height), pygame.SRCALPHA
            )
            beam.fill((255, 255, 255, 18))
            self.screen.blit(beam, (col_x, self.board_rect.y))

            preview_cx = col_x + self.cell_size / 2.0
            preview_cy = self.board_rect.y - self.cell_size * 0.52
            is_valid = self.hovered_col in valid_moves
            self._blit_disc(
                preview_cx,
                preview_cy,
                self.disc_radius,
                HUMAN_PLAYER,
                ghost=not is_valid,
            )
            if is_valid:
                landing_r = self.engine.get_landing_row(self.hovered_col)
                if landing_r != -1:
                    gx, gy = self._slot_center(landing_r, self.hovered_col)
                    self._blit_disc(gx, gy, self.disc_radius, HUMAN_PLAYER, ghost=True)

        # Committed Discs
        grid = self.engine.get_grid()
        for r in range(self.engine.rows):
            for c in range(self.engine.cols):
                cell_val = grid[r][c]
                if cell_val != EMPTY_CELL:
                    cx, cy = self._slot_center(r, c)
                    is_last = self.last_move_cell == (r, c)
                    self._blit_disc(
                        cx, cy, self.disc_radius, cell_val, last_move=is_last
                    )

        # Active Falling Disc (rendered behind the blue cabinet cutouts!)
        if self.active_animation is not None:
            self._blit_disc(
                self.active_animation.x,
                self.active_animation.y,
                self.disc_radius,
                self.active_animation.player,
            )

        # Smooth Anti-Aliased Blue Board Overlay with True Circular Cutouts
        board_face = self.assets.get_board_face(
            self.board_rect.width,
            self.board_rect.height,
            self.engine.rows,
            self.engine.cols,
            self.cell_size,
            self.disc_radius,
        )
        self.screen.blit(board_face, (self.board_rect.x - 12, self.board_rect.y - 12))

        # Column Numbers
        for c in range(self.engine.cols):
            cx, _ = self._slot_center(self.engine.rows - 1, c)
            lbl = self.font_small.render(str(c + 1), True, COLOR_TEXT_SECONDARY)
            self.screen.blit(
                lbl, lbl.get_rect(center=(int(cx), self.board_rect.bottom + 24))
            )

        # Right-Side Telemetry & Control HUD
        self._render_hud_panel(mouse_pos)

    def _render_hud_panel(self, mouse_pos: tuple[int, int]) -> None:
        draw_smooth_card(
            self.screen, self.hud_panel_rect, COLOR_PANEL, COLOR_BORDER, radius=18
        )

        px = self.hud_panel_rect.x + 20
        pw = self.hud_panel_rect.width - 40
        y = self.hud_panel_rect.y + 20

        # 1. Turn Indicator
        turn_box = pygame.Rect(px, y, pw, 64)
        draw_smooth_card(
            self.screen,
            turn_box,
            COLOR_PANEL_INSET,
            COLOR_BORDER,
            radius=12,
            shadow=False,
        )

        if self.state == GameState.GAME_OVER:
            turn_text = "MATCH COMPLETE"
            turn_col = COLOR_TEXT_ACCENT
            badge_player = (
                self.winner
                if self.winner in (HUMAN_PLAYER, AI_PLAYER)
                else HUMAN_PLAYER
            )
        elif self.state == GameState.AI_THINKING:
            dots = "." * ((pygame.time.get_ticks() // 350) % 4)
            turn_text = f"AI THINKING{dots}"
            turn_col = COLOR_TEXT_WARN
            badge_player = AI_PLAYER
        elif self.current_turn == HUMAN_PLAYER:
            turn_text = "YOUR TURN (HUMAN)"
            turn_col = COLOR_TEXT_SUCCESS
            badge_player = HUMAN_PLAYER
        else:
            turn_text = "AI TURN"
            turn_col = COLOR_TEXT_WARN
            badge_player = AI_PLAYER

        self._blit_disc(px + 28, y + 32, 15, badge_player)
        turn_s = self.font_body_bold.render(turn_text, True, turn_col)
        self.screen.blit(turn_s, (px + 54, y + 21))
        y += 78

        # 2. Connected-4s Scoreboard
        human_fours, ai_fours = self.engine.get_scores()
        score_box = pygame.Rect(px, y, pw, 94)
        draw_smooth_card(
            self.screen,
            score_box,
            COLOR_PANEL_INSET,
            COLOR_BORDER,
            radius=12,
            shadow=False,
        )

        s_hdr = self.font_small.render("CONNECTED-4s SCORE", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(s_hdr, (px + 14, y + 10))

        h_col_name = DISC_THEMES[self.human_color_key]["name"]
        a_col_name = DISC_THEMES[self.ai_color_key]["name"]

        self._blit_disc(px + 25, y + 46, 10, HUMAN_PLAYER)
        h_lbl = self.font_body_bold.render(
            f"Human ({h_col_name}):", True, COLOR_TEXT_PRIMARY
        )
        h_val = self.font_heading.render(str(human_fours), True, COLOR_TEXT_PRIMARY)
        self.screen.blit(h_lbl, (px + 44, y + 35))
        self.screen.blit(h_val, (px + pw - h_val.get_width() - 16, y + 31))

        self._blit_disc(px + 25, y + 74, 10, AI_PLAYER)
        a_lbl = self.font_body_bold.render(
            f"AI ({a_col_name}):", True, COLOR_TEXT_PRIMARY
        )
        a_val = self.font_heading.render(str(ai_fours), True, COLOR_TEXT_PRIMARY)
        self.screen.blit(a_lbl, (px + 44, y + 63))
        self.screen.blit(a_val, (px + pw - a_val.get_width() - 16, y + 59))
        y += 108

        # 3. Heuristic & Search Telemetry
        info_box = pygame.Rect(px, y, pw, 212)
        draw_smooth_card(
            self.screen,
            info_box,
            COLOR_PANEL_INSET,
            COLOR_BORDER,
            radius=12,
            shadow=False,
        )

        i_hdr = self.font_small.render(
            "HEURISTIC & SEARCH TELEMETRY", True, COLOR_TEXT_SECONDARY
        )
        self.screen.blit(i_hdr, (px + 14, y + 10))

        telemetry_rows = [
            ("Algorithm:", f"{self.current_algorithm_label} (K={self.depth_k})"),
            ("Board Heuristic:", f"{self.heuristic_info['total']:+.2f}"),
            (
                "Fours / Threats:",
                f"{self.heuristic_info['fours']:+d} / {self.heuristic_info['threats']:+.1f}",
            ),
            ("AI Search Value:", f"{self.last_ai_decision.heuristic_value:+.2f}"),
            ("Nodes Expanded:", f"{self.last_ai_decision.nodes_expanded:,}"),
            ("Think Time:", f"{self.last_ai_decision.time_taken:.4f} s"),
        ]
        ry = y + 36
        for k_txt, v_txt in telemetry_rows:
            ks = self.font_small.render(k_txt, True, COLOR_TEXT_SECONDARY)
            val_col = COLOR_TEXT_ACCENT if "Heuristic" in k_txt else COLOR_TEXT_PRIMARY
            vs = self.font_body_bold.render(v_txt, True, val_col)
            self.screen.blit(ks, (px + 14, ry + 2))
            self.screen.blit(vs, (px + pw - vs.get_width() - 14, ry))
            ry += 28
        y += 226

        # 4. Game Status Message
        status_box = pygame.Rect(px, y, pw, 88)
        draw_smooth_card(
            self.screen,
            status_box,
            COLOR_PANEL_INSET,
            COLOR_BORDER,
            radius=12,
            shadow=False,
        )
        st_hdr = self.font_small.render("STATUS", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(st_hdr, (px + 14, y + 9))
        self._draw_wrapped_text(
            self.status_message,
            pygame.Rect(px + 14, y + 31, pw - 28, 50),
            self.font_small,
            COLOR_TEXT_PRIMARY,
        )

        # 5. Buttons
        controls_enabled = self.state in (GameState.PLAYING, GameState.GAME_OVER)
        self.btn_view_tree.enabled = controls_enabled and bool(
            self.last_ai_decision.tree_trace
        )
        self.btn_restart.enabled = controls_enabled
        self.btn_main_menu.enabled = controls_enabled

        self.btn_view_tree.draw(self.screen, mouse_pos)
        self.btn_restart.draw(self.screen, mouse_pos)
        self.btn_main_menu.draw(self.screen, mouse_pos)

    def _render_tree_viewer(self, mouse_pos: tuple[int, int]) -> None:
        """Bonus Feature: Scrollable in-GUI Search Tree Viewer with heuristic values."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 9, 18, 200))
        self.screen.blit(overlay, (0, 0))

        draw_smooth_card(
            self.screen, self.tree_modal_rect, COLOR_PANEL, COLOR_TEXT_ACCENT, radius=18
        )

        r = self.tree_modal_rect
        title = self.font_heading.render(
            f"AI Search Tree ({self.current_algorithm_label}, K={self.depth_k}) — "
            f"Expanded: {self.last_ai_decision.nodes_expanded} | "
            f"Root Value: {self.last_ai_decision.heuristic_value:+.2f}",
            True,
            COLOR_TEXT_PRIMARY,
        )
        self.screen.blit(title, (r.x + 24, r.y + 22))
        self.btn_close_tree.draw(self.screen, mouse_pos)

        hint = self.font_small.render(
            "Scroll with Mouse Wheel or Arrow Keys  •  Shows MAX, MIN, CHANCE, PRUNED & Heuristic Leaf Values",
            True,
            COLOR_TEXT_SECONDARY,
        )
        self.screen.blit(hint, (r.x + 24, r.y + 56))

        viewport = pygame.Rect(r.x + 22, r.y + 86, r.width - 44, r.height - 108)
        pygame.draw.rect(self.screen, COLOR_PANEL_INSET, viewport, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_BORDER, viewport, width=1, border_radius=12)

        lines = (
            self.last_ai_decision.tree_trace or "No tree recorded yet."
        ).splitlines()
        line_h = self.font_mono.get_linesize() + 3
        max_scroll_y = max(0, len(lines) * line_h - viewport.height + 24)
        self.tree_scroll_y = min(self.tree_scroll_y, max_scroll_y)

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(viewport.inflate(-8, -8))

        start_idx = max(0, self.tree_scroll_y // line_h)
        visible_count = (viewport.height // line_h) + 3
        for idx in range(start_idx, min(len(lines), start_idx + visible_count)):
            line = lines[idx]
            if "PRUNED" in line:
                col = COLOR_TEXT_DANGER
            elif "MAX " in line:
                col = COLOR_TEXT_SUCCESS
            elif "MIN " in line:
                col = COLOR_TEXT_ACCENT
            elif "CHANCE " in line:
                col = COLOR_TEXT_WARN
            else:
                col = COLOR_TEXT_PRIMARY

            surf = self.font_mono.render(line, True, col)
            draw_x = viewport.x + 14 - self.tree_scroll_x
            draw_y = viewport.y + 12 + idx * line_h - self.tree_scroll_y
            self.screen.blit(surf, (draw_x, draw_y))

        self.screen.set_clip(prev_clip)

    def _render_confirm_modal(self, mouse_pos: tuple[int, int]) -> None:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 9, 18, 195))
        self.screen.blit(overlay, (0, 0))

        draw_smooth_card(
            self.screen, self.modal_rect, COLOR_PANEL, COLOR_TEXT_ACCENT, radius=18
        )

        cx = self.modal_rect.centerx
        title = self.font_heading.render("Match In Progress", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title, title.get_rect(center=(cx, self.modal_rect.y + 52)))

        msg = self.font_body.render(
            "Abandon the current game and reset the board?", True, COLOR_TEXT_SECONDARY
        )
        self.screen.blit(msg, msg.get_rect(center=(cx, self.modal_rect.y + 104)))

        self.btn_confirm_yes.draw(self.screen, mouse_pos)
        self.btn_confirm_no.draw(self.screen, mouse_pos)

    def _render_gameover_overlay(self, mouse_pos: tuple[int, int]) -> None:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 9, 18, 185))
        self.screen.blit(overlay, (0, 0))

        if self.winner == HUMAN_PLAYER:
            headline = "HUMAN WINS!"
            accent = DISC_THEMES[self.human_color_key]["mid"]
        elif self.winner == AI_PLAYER:
            headline = "AI WINS!"
            accent = DISC_THEMES[self.ai_color_key]["mid"]
        else:
            headline = "DRAW GAME!"
            accent = COLOR_TEXT_ACCENT

        draw_smooth_card(
            self.screen, self.gameover_card_rect, COLOR_PANEL, accent, radius=20
        )
        pygame.draw.rect(
            self.screen, accent, self.gameover_card_rect, width=2, border_radius=20
        )

        cx = self.gameover_card_rect.centerx
        top = self.gameover_card_rect.y

        h_score, a_score = self.engine.get_scores()
        title_s = self.font_title.render(headline, True, accent)
        self.screen.blit(title_s, title_s.get_rect(center=(cx, top + 72)))

        score_s = self.font_heading.render(
            f"Human: {h_score}   •   AI: {a_score}", True, COLOR_TEXT_PRIMARY
        )
        self.screen.blit(score_s, score_s.get_rect(center=(cx, top + 142)))

        heur_s = self.font_body.render(
            f"Final Board Heuristic: {self.heuristic_info['total']:+.2f}",
            True,
            COLOR_TEXT_SECONDARY,
        )
        self.screen.blit(heur_s, heur_s.get_rect(center=(cx, top + 186)))

        self.btn_play_again.draw(self.screen, mouse_pos)
        self.btn_gameover_menu.draw(self.screen, mouse_pos)

    def _draw_wrapped_text(
        self,
        text: str,
        rect: pygame.Rect,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        words = text.split(" ")
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= rect.width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        line_h = font.get_linesize()
        for idx, line in enumerate(lines[:3]):
            surf = font.render(line, True, color)
            self.screen.blit(surf, (rect.x, rect.y + idx * line_h))

    # -------------------------------------------------------------------------
    # Main Execution Loop
    # -------------------------------------------------------------------------

    def run(self) -> None:
        try:
            while self.running:
                dt = min(0.05, self.clock.tick(FPS) / 1000.0)
                self.handle_events()
                self.update(dt)
                self.render()
        finally:
            pygame.quit()


def main() -> None:
    try:
        app = Connect4GUI()
        app.run()
    except Exception as exc:
        print(f"Fatal GUI error: {exc}", file=sys.stderr)
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
