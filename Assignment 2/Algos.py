"""Depth-limited Connect 4 search. Human = 1, AI = 2, empty = 0."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import inf, isfinite
from numbers import Integral
import random
from time import perf_counter
import board

from TreeTrace import TreeNode, format_tree


HUMAN = 1
AI = 2


def _valid_columns(board):
    if hasattr(board, "get_valid_columns"):
        return sorted(board.get_valid_columns())
    # moves() in the current board class returns rows, not columns.
    if hasattr(board, "moves"):
        return [col for col, row in enumerate(board.moves()) if row != -1]
    return [col for col, cell in enumerate(board.grid[0]) if cell == 0]


def _child_board(board, col, player):
    """Copy first so a sibling branch never sees another branch's move."""
    child = deepcopy(board)
    if hasattr(child, "drop_piece"):
        result = child.drop_piece(col, player)
        if result is False:
            raise ValueError(f"Board rejected legal column {col}")
        # Support both a returned board and a method that changes its own board.
        return result if hasattr(result, "grid") else child
    if not child.play_move(player, col):
        raise ValueError(f"Board rejected legal column {col}")
    return child


def _windows(grid):
    rows, cols = len(grid), len(grid[0])
    for row in range(rows):
        for col in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_row, end_col = row + 3 * dr, col + 3 * dc
                if 0 <= end_row < rows and 0 <= end_col < cols:
                    yield [grid[row + step * dr][col + step * dc]
                           for step in range(4)]


def evaluate(board):
    eval_fours=board.count_connect_fours(2) - board.count_connect_fours(1)
    eval_threats= board.count_threats(2) - 1.1*board.count_threats(1)
    eval_forks= board.count_forks(2) - board.count_forks(1)
    return (1000*eval_fours + 200*eval_threats + 50*eval_forks)


def landing_probabilities(board, chosen_col):
    """Return (actual column, probability) pairs for an intended move.

    A full or out-of-bounds neighbor is unavailable. The remaining neighbor gets
    0.4; if both are unavailable, the chosen column gets all the probability.
    """
    valid = _valid_columns(board)
    if (isinstance(chosen_col, bool) or not isinstance(chosen_col, Integral)
            or chosen_col not in valid):
        raise ValueError("Choose a non-full column inside the board")
    neighbors = [col for col in (chosen_col - 1, chosen_col + 1) if col in valid]
    if not neighbors:
        return [(chosen_col, 1.0)]
    if len(neighbors) == 1:
        return [(chosen_col, 0.6), (neighbors[0], 0.4)]
    return [(chosen_col, 0.6), (neighbors[0], 0.2), (neighbors[1], 0.2)]


def sample_landing_column(board, chosen_col, rng=None):
    """Use this in the game loop to apply a stochastic move after the search."""
    outcomes = landing_probabilities(board, chosen_col)
    generator = rng if rng is not None else random
    return generator.choices(
        [col for col, _ in outcomes],
        weights=[probability for _, probability in outcomes],
        k=1,
    )[0]


@dataclass
class _Stats:
    nodes_visited: int = 0
    nodes_expanded: int = 0
    leaves_evaluated: int = 0
    pruned_branches: int = 0


class _Search:
    def __init__(self, evaluate_fn, max_depth, collect_trace, stochastic_players):
        self.evaluate = evaluate_fn
        self.max_depth = max_depth
        self.collect_trace = collect_trace
        self.stochastic_players = stochastic_players
        self.stats = _Stats()

    def _node(self, kind, depth, edge=""):
        self.stats.nodes_visited += 1
        return TreeNode(kind, depth, edge) if self.collect_trace else None

    def _leaf_value(self, board, depth, valid, node):
        # Four in a row earns a point; it does not end this version of the game.
        if valid and depth < self.max_depth:
            return None
        value = float(self.evaluate(board))
        if not isfinite(value):
            raise ValueError("The evaluator must return a finite number")
        self.stats.leaves_evaluated += 1
        if node is not None:
            node.value = value
            node.note = "heuristic; K reached" if valid else "terminal; board full"
        return value

    def minimax(self, board, depth, player, alpha=-inf, beta=inf,
                pruning=False, edge=""):
        node = self._node("MAX" if player == AI else "MIN", depth, edge)
        valid = _valid_columns(board)
        leaf_value = self._leaf_value(board, depth, valid, node)
        if leaf_value is not None:
            return leaf_value, None, node

        self.stats.nodes_expanded += 1
        best_value = -inf if player == AI else inf
        best_col = None
        for index, col in enumerate(valid):
            child = _child_board(board, col, player)
            value, _, child_node = self.minimax(
                child, depth + 1, 3 - player, alpha, beta, pruning,
                edge=f"col={col}",
            )
            if node is not None:
                node.children.append(child_node)
            if (best_col is None
                    or (player == AI and value > best_value)
                    or (player == HUMAN and value < best_value)):
                best_value, best_col = value, col

            if pruning:
                if player == AI:
                    alpha = max(alpha, best_value)
                else:
                    beta = min(beta, best_value)
                if alpha >= beta:
                    skipped = valid[index + 1:]
                    self.stats.pruned_branches += len(skipped)
                    if node is not None:
                        bound = "lower" if player == AI else "upper"
                        node.note = (f"{bound} bound; alpha={alpha:.6g}, "
                                     f"beta={beta:.6g}")
                        node.pruned_columns = skipped
                    break

        if node is not None:
            node.value = best_value
        return best_value, best_col, node

    def expectiminimax(self, board, depth, player, edge=""):
        node = self._node("MAX" if player == AI else "MIN", depth, edge)
        valid = _valid_columns(board)
        leaf_value = self._leaf_value(board, depth, valid, node)
        if leaf_value is not None:
            return leaf_value, None, node

        self.stats.nodes_expanded += 1
        best_value = -inf if player == AI else inf
        best_col = None
        for col in valid:
            if player in self.stochastic_players:
                value, child_node = self.chance(board, depth, player, col)
            else:
                child = _child_board(board, col, player)
                value, _, child_node = self.expectiminimax(
                    child, depth + 1, 3 - player, edge=f"col={col}",
                )
            if node is not None:
                node.children.append(child_node)
            if (best_col is None
                    or (player == AI and value > best_value)
                    or (player == HUMAN and value < best_value)):
                best_value, best_col = value, col

        if node is not None:
            node.value = best_value
        return best_value, best_col, node

    def chance(self, board, depth, player, chosen_col):
        # Chance resolves the same move. Only placing the disc consumes depth.
        node = self._node("CHANCE", depth, edge=f"choose col={chosen_col}")
        self.stats.nodes_expanded += 1
        expected_value = 0.0
        for actual_col, probability in landing_probabilities(board, chosen_col):
            child = _child_board(board, actual_col, player)
            value, _, child_node = self.expectiminimax(
                child, depth + 1, 3 - player,
                edge=f"land col={actual_col}, p={probability:g}",
            )
            expected_value += probability * value
            if node is not None:
                node.children.append(child_node)
        if node is not None:
            node.value = expected_value
        return expected_value, node


def get_best_move(board, algorithm, max_depth, *, evaluate_fn=evaluate,
                  print_tree=True, collect_trace=True,
                  stochastic_players=(HUMAN, AI)):
    """Choose an AI move without modifying board.

    K counts discs placed, with the root at depth zero. Both players' moves are
    stochastic in expectiminimax by default; pass stochastic_players=(AI,) if
    the tutorial specifies that only AI moves can slip.

    nodes_expanded counts nodes whose successors were generated, including
    CHANCE nodes. nodes_visited also counts leaves. time_taken is search time
    in seconds, excluding tree formatting and console output.

    Pass Student 2's evaluate(board) as evaluate_fn. It must use AI's perspective
    and return the final score difference (in compatible units) on a full board.
    """
    if (isinstance(max_depth, bool) or not isinstance(max_depth, Integral)
            or max_depth < 1):
        raise ValueError("max_depth must be an integer of at least 1")
    if not isinstance(algorithm, str):
        raise ValueError("algorithm must be a string")
    name = algorithm.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"alphabeta": "alpha_beta", "expected_minimax": "expectiminimax",
               "expecti_minimax": "expectiminimax"}
    name = aliases.get(name, name)
    if name not in ("minimax", "alpha_beta", "expectiminimax"):
        raise ValueError("Choose minimax, alpha_beta, or expectiminimax")
    if print_tree and not collect_trace:
        raise ValueError("Printing a tree requires collect_trace=True")
    stochastic_players = tuple(stochastic_players)
    if any(isinstance(p, bool) or p not in (HUMAN, AI)
           for p in stochastic_players):
        raise ValueError("stochastic_players may contain only HUMAN and AI")
    evaluator = default_evaluate if evaluate_fn is None else evaluate_fn
    if not callable(evaluator):
        raise ValueError("evaluate_fn must be callable")

    search = _Search(evaluator, max_depth, collect_trace, stochastic_players)
    start = perf_counter()
    if name == "expectiminimax":
        value, col, root = search.expectiminimax(board, 0, AI)
    else:
        value, col, root = search.minimax(board, 0, AI, pruning=name == "alpha_beta")
    elapsed = perf_counter() - start
    tree_trace = format_tree(root) if root is not None else ""
    if print_tree:
        print(tree_trace)
    return {
        "best_col": col,
        "heuristic_value": value,
        "nodes_expanded": search.stats.nodes_expanded,
        "time_taken": elapsed,
        "tree_trace": tree_trace,
        "nodes_visited": search.stats.nodes_visited,
        "leaves_evaluated": search.stats.leaves_evaluated,
        "pruned_branches": search.stats.pruned_branches,
    }


def minimax(board, max_depth, **options):
    return get_best_move(board, "minimax", max_depth, **options)


def alpha_beta(board, max_depth, **options):
    return get_best_move(board, "alpha_beta", max_depth, **options)


def expectiminimax(board, max_depth, **options):
    return get_best_move(board, "expectiminimax", max_depth, **options)
