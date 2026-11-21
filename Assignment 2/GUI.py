"""Modern Pygame Graphical User Interface for Connect 4 (Human vs. AI).

This module implements the presentation, input handling, animations, and
asynchronous AI execution for Connect 4 while delegating all game rules and
search logic to external modules (`board.py` and `Algos.py`) through dedicated
adapter classes (`GameEngineAdapter` and `AIAgentAdapter`).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
import math
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

WINDOW_WIDTH = 1140
WINDOW_HEIGHT = 780
WINDOW_TITLE = "Connect 4 — Human vs. AI"
FPS = 60

DEFAULT_ROWS = 6
DEFAULT_COLS = 7
MIN_ROWS = 6
MIN_COLS = 7

HUMAN_PLAYER = 1
AI_PLAYER = 2
EMPTY_CELL = 0

ALGORITHM_OPTIONS = [
    ("minimax", "Minimax"),
    ("alpha_beta", "Alpha-Beta"),
    ("expectiminimax", "Expectiminimax"),
]

BOARD_SIZE_OPTIONS = [
    (6, 7, "6 x 7 (Standard)"),
    (7, 8, "7 x 8 (Medium)"),
    (8, 9, "8 x 9 (Large)"),
]

MIN_DEPTH_K = 1
MAX_DEPTH_K = 6
DEFAULT_DEPTH_K = 3

# Color Palette (Modern Dark Slate + Classic Connect 4 Blue/Red/Yellow)
COLOR_BG = (15, 23, 42)              # #0F172A Deep Slate
COLOR_PANEL = (30, 41, 59)           # #1E293B Card Surface
COLOR_PANEL_ALT = (23, 33, 49)       # Subtle inset card
COLOR_BORDER = (51, 65, 85)          # #334155 Slate Border
COLOR_TEXT_PRIMARY = (248, 250, 252) # #F8FAFC Off-white
COLOR_TEXT_SECONDARY = (148, 163, 184)  # #94A3B8 Muted Slate
COLOR_TEXT_ACCENT = (56, 189, 248)   # #38BDF8 Sky Blue

COLOR_BOARD_BODY = (29, 78, 216)     # #1D4ED8 Classic Connect 4 Blue
COLOR_BOARD_EDGE = (30, 58, 138)     # #1E3A8A Darker Blue Bevel
COLOR_SLOT_EMPTY = (11, 17, 32)      # #0B1120 Deep Hole Cutout
COLOR_SLOT_RING = (37, 99, 235)      # #2563EB Subtle Slot Rim
COLOR_HOVER_BEAM = (255, 255, 255, 24)

COLOR_HUMAN = (239, 68, 68)          # #EF4444 Coral Red
COLOR_HUMAN_LIGHT = (252, 165, 165)  # #FCA5A5 Specular Highlight
COLOR_HUMAN_DARK = (185, 28, 28)     # #B91C1C Inner Ring

COLOR_AI = (250, 204, 21)            # #FACC15 Golden Yellow
COLOR_AI_LIGHT = (254, 240, 138)     # #FEF08A Specular Highlight
COLOR_AI_DARK = (202, 138, 4)        # #CA8A04 Inner Ring

COLOR_BTN_PRIMARY = (37, 99, 235)
COLOR_BTN_PRIMARY_HOVER = (59, 130, 246)
COLOR_BTN_SECONDARY = (51, 65, 85)
COLOR_BTN_SECONDARY_HOVER = (71, 85, 105)
COLOR_BTN_DANGER = (220, 38, 38)
COLOR_BTN_DANGER_HOVER = (239, 68, 68)
COLOR_BTN_SELECTED = (14, 165, 233)
COLOR_BTN_DISABLED = (30, 41, 59)

# Animation Physics
GRAVITY_ACCEL = 3200.0  # pixels / s^2
BOUNCE_FACTOR = 0.22


# =============================================================================
# Game States & Data Structures
# =============================================================================

class GameState(Enum):
    MENU = auto()
    PLAYING = auto()
    AI_THINKING = auto()
    CONFIRM_RESTART = auto()
    GAME_OVER = auto()


@dataclass
class AIDecision:
    """Structured result returned by `AIAgentAdapter`."""
    column: Optional[int]
    heuristic_value: float = 0.0
    nodes_expanded: int = 0
    time_taken: float = 0.0
    error: Optional[str] = None


@dataclass
class DiscAnimation:
    """Tracks a single disc falling smoothly into its target slot."""
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
            if not self.bounced and self.vy > 260.0:
                self.vy = -self.vy * BOUNCE_FACTOR
                self.bounced = True
            else:
                self.vy = 0.0
                self.finished = True


# =============================================================================
# Engine & AI Integration Adapters
# =============================================================================

class GameEngineAdapter:
    """Clean interface isolating the GUI from the external `board.py` engine.

    If the underlying board class method names change, update only this class.
    """

    def __init__(self, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> None:
        self.rows = max(MIN_ROWS, rows)
        self.cols = max(MIN_COLS, cols)
        self._board = ExternalBoard(l=self.rows, w=self.cols)

    def reset_board(self, rows: Optional[int] = None, cols: Optional[int] = None) -> None:
        if rows is not None:
            self.rows = max(MIN_ROWS, rows)
        if cols is not None:
            self.cols = max(MIN_COLS, cols)
        self._board = ExternalBoard(l=self.rows, w=self.cols)

    def get_grid(self) -> list[list[int]]:
        return self._board.grid

    def get_valid_moves(self) -> list[int]:
        """Return a sorted list of non-full column indices."""
        if hasattr(self._board, "get_valid_columns"):
            return sorted(self._board.get_valid_columns())
        if hasattr(self._board, "moves"):
            return [c for c, r in enumerate(self._board.moves()) if r != -1]
        return [c for c in range(self.cols) if self._board.grid[0][c] == EMPTY_CELL]

    def get_landing_row(self, column: int) -> int:
        """Return the row index where a disc dropped in `column` will land (-1 if full)."""
        if not (0 <= column < self.cols):
            return -1
        if hasattr(self._board, "moves"):
            return self._board.moves()[column]
        for r in range(self.rows - 1, -1, -1):
            if self._board.grid[r][column] == EMPTY_CELL:
                return r
        return -1

    def make_move(self, column: int, player: int) -> bool:
        """Validate and apply a move to the underlying board engine."""
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
        """Return (human_connect_fours, ai_connect_fours)."""
        if hasattr(self._board, "count_connect_fours"):
            return (
                int(self._board.count_connect_fours(HUMAN_PLAYER)),
                int(self._board.count_connect_fours(AI_PLAYER)),
            )
        return (0, 0)

    def check_winner(self) -> Optional[int]:
        """Determine the match outcome once the board is full.

        Returns:
            HUMAN_PLAYER (1) if Human has more connected-fours,
            AI_PLAYER (2) if AI has more connected-fours,
            0 if tied on a full board,
            None if the game is still ongoing.
        """
        if not self.is_board_full():
            return None
        human_score, ai_score = self.get_scores()
        if human_score > ai_score:
            return HUMAN_PLAYER
        if ai_score > human_score:
            return AI_PLAYER
        return 0

    def evaluate_board(self) -> float:
        """Evaluate the current board using the external heuristic function."""
        if hasattr(external_algos, "evaluate"):
            try:
                return float(external_algos.evaluate(self._board))
            except Exception:
                pass
        if hasattr(external_algos, "default_evaluate"):
            return float(external_algos.default_evaluate(self._board))
        human_score, ai_score = self.get_scores()
        return float(ai_score - human_score)

    def resolve_ai_landing_column(self, chosen_col: int, algorithm: str) -> int:
        """Apply stochastic landing slip when playing Expectiminimax."""
        if algorithm == "expectiminimax" and hasattr(external_algos, "sample_landing_column"):
            return int(external_algos.sample_landing_column(self._board, chosen_col))
        return chosen_col

    def clone_raw_board(self) -> Any:
        """Return a deep copy of the underlying board for thread-safe AI search."""
        return deepcopy(self._board)


class AIAgentAdapter:
    """Clean interface isolating the GUI from the external AI search module.

    Supports `Algos.get_best_move(board, algorithm, max_depth, ...)` as well as
    simpler `get_best_move(board, difficulty)` signatures.
    """

    def __init__(self, evaluate_fn: Optional[Any] = None) -> None:
        self.evaluate_fn = evaluate_fn

    def get_best_move(self, board_snapshot: Any, algorithm: str, depth_k: int) -> AIDecision:
        """Query the external AI module and normalize its response into `AIDecision`."""
        try:
            if not hasattr(external_algos, "get_best_move"):
                raise AttributeError("`Algos.py` does not define `get_best_move`.")

            kwargs: dict[str, Any] = {
                "print_tree": True,
                "stochastic_players": (AI_PLAYER,),
            }
            if self.evaluate_fn is not None:
                kwargs["evaluate_fn"] = self.evaluate_fn

            try:
                raw_result = external_algos.get_best_move(
                    board_snapshot, algorithm, depth_k, **kwargs
                )
            except TypeError:
                # Fallback if a simplified `get_best_move(board, difficulty)` is plugged in
                raw_result = external_algos.get_best_move(board_snapshot, depth_k)

            if isinstance(raw_result, dict):
                return AIDecision(
                    column=raw_result.get("best_col"),
                    heuristic_value=float(raw_result.get("heuristic_value", 0.0)),
                    nodes_expanded=int(raw_result.get("nodes_expanded", 0)),
                    time_taken=float(raw_result.get("time_taken", 0.0)),
                )
            return AIDecision(column=int(raw_result))

        except Exception as exc:
            return AIDecision(column=None, error=str(exc))


# =============================================================================
# Reusable UI Components
# =============================================================================

class Button:
    """Clean, rounded interactive button widget for Pygame."""

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
        if not self.enabled:
            fill = COLOR_BTN_DISABLED
            label_color = COLOR_TEXT_SECONDARY
        elif self.selected:
            fill = self.selected_color
            label_color = self.text_color
        elif self.is_hovered(mouse_pos):
            fill = self.hover_color
            label_color = self.text_color
        else:
            fill = self.base_color
            label_color = self.text_color

        pygame.draw.rect(surface, fill, self.rect, border_radius=self.border_radius)
        border_col = COLOR_TEXT_ACCENT if self.selected else COLOR_BORDER
        pygame.draw.rect(
            surface, border_col, self.rect, width=2, border_radius=self.border_radius
        )

        text_surf = self.font.render(self.text, True, label_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)


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

        # Typography
        self.font_title = pygame.font.SysFont("segoeui", 54, bold=True)
        self.font_subtitle = pygame.font.SysFont("segoeui", 20)
        self.font_heading = pygame.font.SysFont("segoeui", 24, bold=True)
        self.font_body = pygame.font.SysFont("segoeui", 18)
        self.font_body_bold = pygame.font.SysFont("segoeui", 18, bold=True)
        self.font_small = pygame.font.SysFont("segoeui", 15)
        self.font_btn = pygame.font.SysFont("segoeui", 18, bold=True)

        # Match Configuration (adjustable from Main Menu)
        self.selected_algo_idx = 1  # Default: Alpha-Beta
        self.selected_size_idx = 0  # Default: 6 x 7
        self.depth_k = DEFAULT_DEPTH_K

        # Engine & AI Adapters
        rows, cols, _ = BOARD_SIZE_OPTIONS[self.selected_size_idx]
        self.engine = GameEngineAdapter(rows=rows, cols=cols)
        eval_fn = getattr(external_algos, "evaluate", None)
        self.ai_agent = AIAgentAdapter(evaluate_fn=eval_fn)

        # Runtime State
        self.state = GameState.MENU
        self.previous_state = GameState.PLAYING
        self.current_turn = HUMAN_PLAYER
        self.hovered_col: Optional[int] = None
        self.active_animation: Optional[DiscAnimation] = None
        self.moves_played = 0
        self.status_message = "Click any column to drop your Red disc."

        # Telemetry & Heuristic Display
        self.board_heuristic = 0.0
        self.last_ai_decision = AIDecision(column=None)
        self.winner: Optional[int] = None

        # Background AI Thread State
        self._ai_lock = threading.Lock()
        self._ai_thread: Optional[threading.Thread] = None
        self._pending_ai_decision: Optional[AIDecision] = None

        # Board Geometry Cache
        self.board_rect = pygame.Rect(36, 110, 680, 620)
        self.cell_size = 90
        self.disc_radius = 38
        self._recompute_board_geometry()

        # Build UI Controls
        self._init_menu_widgets()
        self._init_hud_widgets()
        self._init_modal_widgets()

    # -------------------------------------------------------------------------
    # Layout & Widget Setup
    # -------------------------------------------------------------------------

    def _recompute_board_geometry(self) -> None:
        """Compute responsive board slot size and centering for any grid dimension."""
        max_board_w = 700
        max_board_h = 610
        origin_x = 36
        origin_y = 126

        # Leave 1 extra row height above the board for the hover preview disc
        cell_w = max_board_w // self.engine.cols
        cell_h = max_board_h // (self.engine.rows + 1)
        self.cell_size = max(44, min(cell_w, cell_h))
        self.disc_radius = int(self.cell_size * 0.41)

        total_w = self.cell_size * self.engine.cols
        total_h = self.cell_size * self.engine.rows
        board_x = origin_x + (max_board_w - total_w) // 2
        board_y = origin_y + self.cell_size + (max_board_h - self.cell_size - total_h) // 2
        self.board_rect = pygame.Rect(board_x, board_y, total_w, total_h)

    def _init_menu_widgets(self) -> None:
        cx = WINDOW_WIDTH // 2
        self.menu_algo_buttons: list[Button] = []
        btn_w, btn_h, gap = 180, 44, 14
        total_w = len(ALGORITHM_OPTIONS) * btn_w + (len(ALGORITHM_OPTIONS) - 1) * gap
        start_x = cx - total_w // 2
        for idx, (_, label) in enumerate(ALGORITHM_OPTIONS):
            rect = pygame.Rect(start_x + idx * (btn_w + gap), 310, btn_w, btn_h)
            self.menu_algo_buttons.append(
                Button(rect, label, self.font_small, base_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_SECONDARY_HOVER)
            )

        self.menu_size_buttons: list[Button] = []
        for idx, (_, _, label) in enumerate(BOARD_SIZE_OPTIONS):
            rect = pygame.Rect(start_x + idx * (btn_w + gap), 405, btn_w, btn_h)
            self.menu_size_buttons.append(
                Button(rect, label, self.font_small, base_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_SECONDARY_HOVER)
            )

        self.btn_depth_minus = Button(
            pygame.Rect(cx - 110, 498, 46, 42), "-", self.font_heading,
            base_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_SECONDARY_HOVER
        )
        self.btn_depth_plus = Button(
            pygame.Rect(cx + 64, 498, 46, 42), "+", self.font_heading,
            base_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_SECONDARY_HOVER
        )

        self.btn_start_game = Button(
            pygame.Rect(cx - 150, 580, 300, 54), "Start Game", self.font_btn,
            base_color=COLOR_BTN_PRIMARY, hover_color=COLOR_BTN_PRIMARY_HOVER
        )
        self.btn_exit_game = Button(
            pygame.Rect(cx - 150, 648, 300, 48), "Exit", self.font_btn,
            base_color=COLOR_BTN_SECONDARY, hover_color=COLOR_BTN_DANGER_HOVER
        )

    def _init_hud_widgets(self) -> None:
        panel_x = 768
        panel_w = WINDOW_WIDTH - panel_x - 32
        self.hud_panel_rect = pygame.Rect(panel_x, 32, panel_w, WINDOW_HEIGHT - 64)

        btn_y = WINDOW_HEIGHT - 148
        self.btn_restart = Button(
            pygame.Rect(panel_x + 20, btn_y, panel_w - 40, 44),
            "Restart Game",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_PRIMARY_HOVER,
        )
        self.btn_main_menu = Button(
            pygame.Rect(panel_x + 20, btn_y + 56, panel_w - 40, 44),
            "Main Menu",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_DANGER_HOVER,
        )

    def _init_modal_widgets(self) -> None:
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        self.modal_rect = pygame.Rect(cx - 230, cy - 125, 460, 250)
        self.btn_confirm_yes = Button(
            pygame.Rect(cx - 185, cy + 45, 170, 46),
            "Yes, Reset",
            self.font_btn,
            base_color=COLOR_BTN_DANGER,
            hover_color=COLOR_BTN_DANGER_HOVER,
        )
        self.btn_confirm_no = Button(
            pygame.Rect(cx + 15, cy + 45, 170, 46),
            "Cancel",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )

        # End-game action buttons
        self.gameover_card_rect = pygame.Rect(cx - 250, cy - 165, 500, 330)
        self.btn_play_again = Button(
            pygame.Rect(cx - 205, cy + 88, 195, 48),
            "Play Again",
            self.font_btn,
            base_color=COLOR_BTN_PRIMARY,
            hover_color=COLOR_BTN_PRIMARY_HOVER,
        )
        self.btn_gameover_menu = Button(
            pygame.Rect(cx + 10, cy + 88, 195, 48),
            "Main Menu",
            self.font_btn,
            base_color=COLOR_BTN_SECONDARY,
            hover_color=COLOR_BTN_SECONDARY_HOVER,
        )

    # -------------------------------------------------------------------------
    # Game Lifecycle & Turn Management
    # -------------------------------------------------------------------------

    @property
    def current_algorithm_key(self) -> str:
        return ALGORITHM_OPTIONS[self.selected_algo_idx][0]

    @property
    def current_algorithm_label(self) -> str:
        return ALGORITHM_OPTIONS[self.selected_algo_idx][1]

    def _start_new_game(self) -> None:
        rows, cols, _ = BOARD_SIZE_OPTIONS[self.selected_size_idx]
        self.engine.reset_board(rows=rows, cols=cols)
        self._recompute_board_geometry()

        with self._ai_lock:
            self._pending_ai_decision = None

        self.state = GameState.PLAYING
        self.current_turn = HUMAN_PLAYER
        self.active_animation = None
        self.moves_played = 0
        self.winner = None
        self.board_heuristic = self.engine.evaluate_board()
        self.last_ai_decision = AIDecision(column=None)
        self.status_message = "Your turn! Click a column to drop a Red disc."

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
            self.status_message = f"Dropping Red disc into Column {col + 1}..."

    def _commit_animated_move(self, anim: DiscAnimation) -> None:
        """Apply the move to the game engine once its falling animation lands."""
        applied = self.engine.make_move(anim.col, anim.player)
        if not applied:
            self.status_message = f"Move rejected in Column {anim.col + 1}."
            return

        self.moves_played += 1
        self.board_heuristic = self.engine.evaluate_board()

        # Check end-of-game (board full -> compare connected-fours)
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

        # Switch turns
        if anim.player == HUMAN_PLAYER:
            self.current_turn = AI_PLAYER
            self._start_ai_turn()
        else:
            self.current_turn = HUMAN_PLAYER
            self.state = GameState.PLAYING
            self.status_message = "Your turn! Choose a column."

    def _start_ai_turn(self) -> None:
        """Spawn a daemon worker thread so AI search never freezes the UI."""
        self.state = GameState.AI_THINKING
        self.status_message = f"AI is calculating ({self.current_algorithm_label}, K={self.depth_k})..."

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
            # Graceful fallback if external AI throws an exception or returns an invalid column
            chosen_col = valid_moves[0]
            err_note = f" ({decision.error})" if decision.error else ""
            self.status_message = f"AI fallback to Column {chosen_col + 1}{err_note}."

        actual_col = self.engine.resolve_ai_landing_column(
            chosen_col, self.current_algorithm_key
        )
        if actual_col not in valid_moves:
            actual_col = chosen_col

        if actual_col != chosen_col:
            self.status_message = (
                f"Expectiminimax: AI aimed at Col {chosen_col + 1}, "
                f"slipped into Col {actual_col + 1}!"
            )
        else:
            self.status_message = f"AI drops Yellow disc in Column {actual_col + 1}."

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

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if self.state == GameState.CONFIRM_RESTART:
                    self.state = self.previous_state
                elif self.state in (GameState.PLAYING, GameState.GAME_OVER):
                    self._request_navigation(GameState.MENU)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_click(event.pos)

    def _request_navigation(self, target_action: str | GameState) -> None:
        """Show confirmation dialog if a match is actively in progress."""
        if self.state == GameState.AI_THINKING or self.active_animation is not None:
            self.status_message = "Wait for the current turn to finish before resetting."
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
            for idx, btn in enumerate(self.menu_algo_buttons):
                if btn.handle_click(pos):
                    self.selected_algo_idx = idx
                    return
            for idx, btn in enumerate(self.menu_size_buttons):
                if btn.handle_click(pos):
                    self.selected_size_idx = idx
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
            # Block board clicks and mid-search resets while AI thread is active
            self.status_message = "AI is thinking — please wait..."

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
            if self.btn_play_again.handle_click(pos) or self.btn_restart.handle_click(pos):
                self._start_new_game()
            elif self.btn_gameover_menu.handle_click(pos) or self.btn_main_menu.handle_click(pos):
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
        self.screen.fill(COLOR_BG)
        mouse_pos = pygame.mouse.get_pos()

        if self.state == GameState.MENU:
            self._render_menu(mouse_pos)
        else:
            self._render_game_screen(mouse_pos)
            if self.state == GameState.CONFIRM_RESTART:
                self._render_confirm_modal(mouse_pos)
            elif self.state == GameState.GAME_OVER:
                self._render_gameover_overlay(mouse_pos)

        pygame.display.flip()

    def _draw_disc(
        self,
        surface: pygame.Surface,
        cx: float,
        cy: float,
        radius: int,
        player: int,
        alpha: int = 255,
    ) -> None:
        if player == HUMAN_PLAYER:
            main_col, dark_col, light_col = COLOR_HUMAN, COLOR_HUMAN_DARK, COLOR_HUMAN_LIGHT
        else:
            main_col, dark_col, light_col = COLOR_AI, COLOR_AI_DARK, COLOR_AI_LIGHT

        if alpha < 255:
            disc_surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            center = (radius + 2, radius + 2)
            pygame.draw.circle(disc_surf, (*main_col, alpha), center, radius)
            pygame.draw.circle(disc_surf, (*dark_col, alpha), center, radius, width=3)
            surface.blit(disc_surf, (int(cx - radius - 2), int(cy - radius - 2)))
            return

        center_int = (int(cx), int(cy))
        pygame.draw.circle(surface, main_col, center_int, radius)
        pygame.draw.circle(surface, dark_col, center_int, radius, width=3)
        pygame.draw.circle(surface, dark_col, center_int, int(radius * 0.72), width=2)
        highlight_pos = (int(cx - radius * 0.3), int(cy - radius * 0.3))
        pygame.draw.circle(surface, light_col, highlight_pos, max(3, int(radius * 0.16)))

    def _render_menu(self, mouse_pos: tuple[int, int]) -> None:
        cx = WINDOW_WIDTH // 2

        # Outer card
        card = pygame.Rect(cx - 330, 65, 660, 655)
        pygame.draw.rect(self.screen, COLOR_PANEL, card, border_radius=18)
        pygame.draw.rect(self.screen, COLOR_BORDER, card, width=2, border_radius=18)

        # Decorative discs flanking title
        self._draw_disc(self.screen, cx - 215, 138, 22, HUMAN_PLAYER)
        self._draw_disc(self.screen, cx + 215, 138, 22, AI_PLAYER)

        title_surf = self.font_title.render("CONNECT 4", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title_surf, title_surf.get_rect(center=(cx, 138)))

        sub_surf = self.font_subtitle.render(
            "Human (Red) vs. AI (Yellow)  •  Full-Board Scoring",
            True,
            COLOR_TEXT_SECONDARY,
        )
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(cx, 192)))

        # Algorithm Selector
        algo_lbl = self.font_body_bold.render("Select AI Algorithm", True, COLOR_TEXT_ACCENT)
        self.screen.blit(algo_lbl, algo_lbl.get_rect(center=(cx, 282)))
        for idx, btn in enumerate(self.menu_algo_buttons):
            btn.selected = idx == self.selected_algo_idx
            btn.draw(self.screen, mouse_pos)

        # Board Dimension Selector
        size_lbl = self.font_body_bold.render("Board Dimensions (Rows x Columns)", True, COLOR_TEXT_ACCENT)
        self.screen.blit(size_lbl, size_lbl.get_rect(center=(cx, 380)))
        for idx, btn in enumerate(self.menu_size_buttons):
            btn.selected = idx == self.selected_size_idx
            btn.draw(self.screen, mouse_pos)

        # Depth K Selector
        depth_lbl = self.font_body_bold.render("Search Tree Depth (K)", True, COLOR_TEXT_ACCENT)
        self.screen.blit(depth_lbl, depth_lbl.get_rect(center=(cx, 474)))
        self.btn_depth_minus.enabled = self.depth_k > MIN_DEPTH_K
        self.btn_depth_plus.enabled = self.depth_k < MAX_DEPTH_K
        self.btn_depth_minus.draw(self.screen, mouse_pos)
        self.btn_depth_plus.draw(self.screen, mouse_pos)

        k_box = pygame.Rect(cx - 50, 498, 100, 42)
        pygame.draw.rect(self.screen, COLOR_PANEL_ALT, k_box, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, k_box, width=2, border_radius=8)
        k_val_surf = self.font_heading.render(str(self.depth_k), True, COLOR_TEXT_PRIMARY)
        self.screen.blit(k_val_surf, k_val_surf.get_rect(center=k_box.center))

        # Primary Actions
        self.btn_start_game.draw(self.screen, mouse_pos)
        self.btn_exit_game.draw(self.screen, mouse_pos)

    def _render_game_screen(self, mouse_pos: tuple[int, int]) -> None:
        # Top Header Bar above board
        header_title = self.font_heading.render("CONNECT 4", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(header_title, (self.board_rect.x, 34))

        sub_info = self.font_small.render(
            f"Grid: {self.engine.rows}x{self.engine.cols}  |  Play until full — highest Connected-4s wins",
            True,
            COLOR_TEXT_SECONDARY,
        )
        self.screen.blit(sub_info, (self.board_rect.x, 66))

        # Hover Preview Disc & Column Highlight
        valid_moves = self.engine.get_valid_moves()
        if (
            self.state == GameState.PLAYING
            and self.current_turn == HUMAN_PLAYER
            and self.active_animation is None
            and self.hovered_col is not None
        ):
            col_x = self.board_rect.x + self.hovered_col * self.cell_size
            beam_surf = pygame.Surface((self.cell_size, self.board_rect.height), pygame.SRCALPHA)
            beam_surf.fill(COLOR_HOVER_BEAM)
            self.screen.blit(beam_surf, (col_x, self.board_rect.y))

            preview_cx = col_x + self.cell_size / 2.0
            preview_cy = self.board_rect.y - self.cell_size * 0.52
            alpha = 235 if self.hovered_col in valid_moves else 70
            self._draw_disc(
                self.screen, preview_cx, preview_cy, self.disc_radius, HUMAN_PLAYER, alpha=alpha
            )

        # Draw Falling Animation Disc behind the board frame
        if self.active_animation is not None:
            self._draw_disc(
                self.screen,
                self.active_animation.x,
                self.active_animation.y,
                self.disc_radius,
                self.active_animation.player,
            )

        # Draw Board Frame & Committed Discs
        outer_frame = self.board_rect.inflate(18, 18)
        pygame.draw.rect(self.screen, COLOR_BOARD_EDGE, outer_frame, border_radius=18)
        pygame.draw.rect(self.screen, COLOR_BOARD_BODY, self.board_rect, border_radius=14)

        grid = self.engine.get_grid()
        for r in range(self.engine.rows):
            for c in range(self.engine.cols):
                cx, cy = self._slot_center(r, c)
                cell_val = grid[r][c]
                if cell_val == EMPTY_CELL:
                    pygame.draw.circle(self.screen, COLOR_SLOT_EMPTY, (int(cx), int(cy)), self.disc_radius)
                    pygame.draw.circle(
                        self.screen, COLOR_SLOT_RING, (int(cx), int(cy)), self.disc_radius, width=2
                    )
                else:
                    pygame.draw.circle(self.screen, COLOR_SLOT_EMPTY, (int(cx), int(cy)), self.disc_radius)
                    self._draw_disc(self.screen, cx, cy, self.disc_radius, cell_val)

        # Column Index Labels below board
        for c in range(self.engine.cols):
            cx, _ = self._slot_center(self.engine.rows - 1, c)
            lbl = self.font_small.render(str(c + 1), True, COLOR_TEXT_SECONDARY)
            self.screen.blit(lbl, lbl.get_rect(center=(int(cx), self.board_rect.bottom + 24)))

        # Side HUD Panel
        self._render_hud_panel(mouse_pos)

    def _render_hud_panel(self, mouse_pos: tuple[int, int]) -> None:
        pygame.draw.rect(self.screen, COLOR_PANEL, self.hud_panel_rect, border_radius=16)
        pygame.draw.rect(self.screen, COLOR_BORDER, self.hud_panel_rect, width=2, border_radius=16)

        px = self.hud_panel_rect.x + 20
        pw = self.hud_panel_rect.width - 40
        y = self.hud_panel_rect.y + 22

        # 1. Turn Indicator Card
        turn_box = pygame.Rect(px, y, pw, 68)
        pygame.draw.rect(self.screen, COLOR_PANEL_ALT, turn_box, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_BORDER, turn_box, width=1, border_radius=12)

        if self.state == GameState.GAME_OVER:
            turn_text = "MATCH COMPLETE"
            turn_col = COLOR_TEXT_ACCENT
            indicator_player = self.winner if self.winner in (HUMAN_PLAYER, AI_PLAYER) else HUMAN_PLAYER
        elif self.state == GameState.AI_THINKING:
            dots = "." * ((pygame.time.get_ticks() // 350) % 4)
            turn_text = f"AI THINKING{dots}"
            turn_col = COLOR_AI
            indicator_player = AI_PLAYER
        elif self.current_turn == HUMAN_PLAYER:
            turn_text = "YOUR TURN (HUMAN)"
            turn_col = COLOR_HUMAN_LIGHT
            indicator_player = HUMAN_PLAYER
        else:
            turn_text = "AI TURN"
            turn_col = COLOR_AI
            indicator_player = AI_PLAYER

        self._draw_disc(self.screen, px + 28, y + 34, 15, indicator_player)
        turn_surf = self.font_body_bold.render(turn_text, True, turn_col)
        self.screen.blit(turn_surf, (px + 54, y + 22))
        y += 84

        # 2. Connected-4s Scoreboard
        human_fours, ai_fours = self.engine.get_scores()
        score_box = pygame.Rect(px, y, pw, 96)
        pygame.draw.rect(self.screen, COLOR_PANEL_ALT, score_box, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_BORDER, score_box, width=1, border_radius=12)

        score_hdr = self.font_small.render("CONNECTED-4s SCORE", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(score_hdr, (px + 14, y + 10))

        self._draw_disc(self.screen, px + 26, y + 48, 10, HUMAN_PLAYER)
        h_lbl = self.font_body_bold.render(f"Human (Red):  {human_fours}", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(h_lbl, (px + 44, y + 36))

        self._draw_disc(self.screen, px + 26, y + 76, 10, AI_PLAYER)
        a_lbl = self.font_body_bold.render(f"AI (Yellow):     {ai_fours}", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(a_lbl, (px + 44, y + 64))
        y += 112

        # 3. AI Settings & Heuristic Evaluation Telemetry
        info_box = pygame.Rect(px, y, pw, 198)
        pygame.draw.rect(self.screen, COLOR_PANEL_ALT, info_box, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_BORDER, info_box, width=1, border_radius=12)

        info_hdr = self.font_small.render("AI & HEURISTIC TELEMETRY", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(info_hdr, (px + 14, y + 10))

        rows_data = [
            ("Algorithm:", f"{self.current_algorithm_label} (K={self.depth_k})"),
            ("Board Heuristic:", f"{self.board_heuristic:+.2f}"),
            ("AI Search Value:", f"{self.last_ai_decision.heuristic_value:+.2f}"),
            ("Nodes Expanded:", f"{self.last_ai_decision.nodes_expanded:,}"),
            ("AI Think Time:", f"{self.last_ai_decision.time_taken:.4f} s"),
        ]
        row_y = y + 38
        for k_txt, v_txt in rows_data:
            ks = self.font_small.render(k_txt, True, COLOR_TEXT_SECONDARY)
            val_color = COLOR_TEXT_ACCENT if "Heuristic" in k_txt else COLOR_TEXT_PRIMARY
            vs = self.font_body_bold.render(v_txt, True, val_color)
            self.screen.blit(ks, (px + 14, row_y + 2))
            self.screen.blit(vs, (px + pw - vs.get_width() - 14, row_y))
            row_y += 30
        y += 214

        # 4. Status Message Box (wrapped text)
        status_box = pygame.Rect(px, y, pw, 94)
        pygame.draw.rect(self.screen, COLOR_PANEL_ALT, status_box, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_BORDER, status_box, width=1, border_radius=12)
        status_hdr = self.font_small.render("GAME STATUS", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(status_hdr, (px + 14, y + 10))
        self._draw_wrapped_text(
            self.status_message,
            pygame.Rect(px + 14, y + 34, pw - 28, 52),
            self.font_small,
            COLOR_TEXT_PRIMARY,
        )

        # 5. Control Buttons
        controls_enabled = self.state in (GameState.PLAYING, GameState.GAME_OVER)
        self.btn_restart.enabled = controls_enabled
        self.btn_main_menu.enabled = controls_enabled
        self.btn_restart.draw(self.screen, mouse_pos)
        self.btn_main_menu.draw(self.screen, mouse_pos)

    def _render_confirm_modal(self, mouse_pos: tuple[int, int]) -> None:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 10, 20, 185))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, COLOR_PANEL, self.modal_rect, border_radius=16)
        pygame.draw.rect(self.screen, COLOR_TEXT_ACCENT, self.modal_rect, width=2, border_radius=16)

        cx = self.modal_rect.centerx
        title = self.font_heading.render("Game In Progress", True, COLOR_TEXT_PRIMARY)
        self.screen.blit(title, title.get_rect(center=(cx, self.modal_rect.y + 48)))

        msg = self.font_body.render(
            "Abandon the current match and reset?", True, COLOR_TEXT_SECONDARY
        )
        self.screen.blit(msg, msg.get_rect(center=(cx, self.modal_rect.y + 98)))

        self.btn_confirm_yes.draw(self.screen, mouse_pos)
        self.btn_confirm_no.draw(self.screen, mouse_pos)

    def _render_gameover_overlay(self, mouse_pos: tuple[int, int]) -> None:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 10, 20, 175))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, COLOR_PANEL, self.gameover_card_rect, border_radius=18)
        border_col = (
            COLOR_HUMAN if self.winner == HUMAN_PLAYER
            else COLOR_AI if self.winner == AI_PLAYER
            else COLOR_TEXT_ACCENT
        )
        pygame.draw.rect(self.screen, border_col, self.gameover_card_rect, width=3, border_radius=18)

        cx = self.gameover_card_rect.centerx
        top = self.gameover_card_rect.y

        if self.winner == HUMAN_PLAYER:
            headline = "HUMAN WINS!"
        elif self.winner == AI_PLAYER:
            headline = "AI WINS!"
        else:
            headline = "DRAW GAME!"

        h_score, a_score = self.engine.get_scores()
        title_surf = self.font_title.render(headline, True, border_col)
        self.screen.blit(title_surf, title_surf.get_rect(center=(cx, top + 68)))

        score_surf = self.font_heading.render(
            f"Human: {h_score}   |   AI: {a_score}", True, COLOR_TEXT_PRIMARY
        )
        self.screen.blit(score_surf, score_surf.get_rect(center=(cx, top + 134)))

        heur_surf = self.font_body.render(
            f"Final Board Heuristic: {self.board_heuristic:+.2f}",
            True,
            COLOR_TEXT_SECONDARY,
        )
        self.screen.blit(heur_surf, heur_surf.get_rect(center=(cx, top + 176)))

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
