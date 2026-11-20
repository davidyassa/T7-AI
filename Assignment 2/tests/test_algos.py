"""Run with: python -m unittest discover -s tests -v"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import math
import random
import unittest

from Algos import (
    alpha_beta, default_evaluate, expectiminimax, get_best_move,
    landing_probabilities, minimax, sample_landing_column,
)
from board import board


class SmallBoard:
    """Small positions make exhaustive reference searches cheap."""

    def __init__(self, grid):
        self.grid = deepcopy(grid)

    def get_valid_columns(self):
        return [col for col, value in enumerate(self.grid[0]) if value == 0]

    def drop_piece(self, col, player):
        for row in range(len(self.grid) - 1, -1, -1):
            if self.grid[row][col] == 0:
                self.grid[row][col] = player
                return None
        return False


class ReturningBoard(SmallBoard):
    def drop_piece(self, col, player):
        child = SmallBoard(self.grid)
        child.drop_piece(col, player)
        return child


def weighted_evaluate(position):
    return sum((1 if value == 2 else -1 if value == 1 else 0)
               * (row + 1) * (col + 1)
               for row, cells in enumerate(position.grid)
               for col, value in enumerate(cells))


def reference_search(grid, depth, expected=False, player=2, stochastic=(1, 2)):
    """Independent exhaustive calculation; no production search helpers."""
    valid = [col for col, value in enumerate(grid[0]) if value == 0]
    if depth == 0 or not valid:
        return weighted_evaluate(SmallBoard(grid)), None
    values = []
    for chosen in valid:
        outcomes = [(chosen, 1.0)]
        if expected and player in stochastic:
            left, right = chosen - 1 in valid, chosen + 1 in valid
            outcomes = [(chosen, 0.6 if left or right else 1.0)]
            if left:
                outcomes.append((chosen - 1, 0.2 if right else 0.4))
            if right:
                outcomes.append((chosen + 1, 0.2 if left else 0.4))
        total = 0.0
        for actual, probability in outcomes:
            child = deepcopy(grid)
            for row in reversed(child):
                if row[actual] == 0:
                    row[actual] = player
                    break
            total += probability * reference_search(
                child, depth - 1, expected, 3 - player, stochastic,
            )[0]
        values.append((total, chosen))
    best = max(value for value, _ in values) if player == 2 else min(
        value for value, _ in values)
    return best, next(col for value, col in values if value == best)


class SearchTests(unittest.TestCase):
    def search(self, position, algorithm="minimax", depth=2, **options):
        return get_best_move(position, algorithm, depth, print_tree=False, **options)

    def test_searches_match_exhaustive_reference(self):
        rng = random.Random(17)
        for _ in range(12):
            position = SmallBoard([[0] * 3 for _ in range(3)])
            for turn in range(rng.randrange(6)):
                position.drop_piece(rng.choice(
                    position.get_valid_columns()), turn % 2 + 1)
            for depth in (1, 2, 3):
                for algorithm in ("minimax", "alpha_beta", "expectiminimax"):
                    with self.subTest(grid=position.grid, depth=depth, algorithm=algorithm):
                        expected = reference_search(
                            position.grid, depth, algorithm == "expectiminimax")
                        result = self.search(position, algorithm, depth,
                                             evaluate_fn=weighted_evaluate)
                        self.assertAlmostEqual(
                            result["heuristic_value"], expected[0])
                        self.assertEqual(result["best_col"], expected[1])

    def test_ai_only_chance_matches_reference(self):
        position = SmallBoard([[0, 0, 0], [1, 0, 2]])
        expected = reference_search(position.grid, 3, True, stochastic=(2,))
        result = self.search(position, "expectiminimax", 3,
                             evaluate_fn=weighted_evaluate, stochastic_players=(2,))
        self.assertAlmostEqual(result["heuristic_value"], expected[0])
        self.assertEqual(result["best_col"], expected[1])

    def test_no_stochastic_players_matches_minimax(self):
        position = SmallBoard([[0, 0, 0], [1, 0, 2]])
        plain = self.search(position, evaluate_fn=weighted_evaluate)
        expected = self.search(position, "expectiminimax",
                               evaluate_fn=weighted_evaluate, stochastic_players=())
        self.assertEqual(plain["heuristic_value"], expected["heuristic_value"])
        self.assertEqual(plain["best_col"], expected["best_col"])

    def test_input_is_unchanged_for_every_algorithm(self):
        position = board()
        position.play_move(1, 0)
        before = deepcopy(position.__dict__)
        for algorithm in ("minimax", "alpha_beta", "expectiminimax"):
            self.search(position, algorithm)
            self.assertEqual(position.__dict__, before)

    def test_both_board_interfaces(self):
        for cls in (SmallBoard, ReturningBoard):
            position = cls([[0, 0, 0], [1, 0, 2]])
            before = deepcopy(position.grid)
            result = self.search(position, evaluate_fn=weighted_evaluate)
            self.assertEqual(result["best_col"],
                             reference_search(before, 2)[1])
            self.assertEqual(position.grid, before)

    def test_full_board_returns_no_move(self):
        position = SmallBoard([[1, 2], [2, 1]])
        for algorithm in ("minimax", "alpha_beta", "expectiminimax"):
            result = self.search(position, algorithm,
                                 evaluate_fn=weighted_evaluate)
            self.assertIsNone(result["best_col"])
            self.assertEqual(result["heuristic_value"],
                             weighted_evaluate(position))
            self.assertEqual(result["nodes_expanded"], 0)
            self.assertEqual(result["leaves_evaluated"], 1)
            self.assertIn("terminal; board full", result["tree_trace"])

    def test_one_legal_move(self):
        position = SmallBoard([[1, 0, 2], [2, 1, 1]])
        for algorithm in ("minimax", "alpha_beta", "expectiminimax"):
            result = self.search(position, algorithm, 5,
                                 evaluate_fn=weighted_evaluate)
            self.assertEqual(result["best_col"], 1)

    def test_four_does_not_end_search(self):
        position = board()
        for col in range(4):
            position.play_move(2, col)
        result = self.search(position, depth=1)
        self.assertIsNotNone(result["best_col"])
        self.assertEqual(result["nodes_visited"], 8)

    def test_exact_minimax_node_counts(self):
        result = self.search(board(), depth=2, evaluate_fn=lambda b: 0)
        self.assertEqual(result["nodes_expanded"], 8)
        self.assertEqual(result["nodes_visited"], 57)
        self.assertEqual(result["leaves_evaluated"], 49)

    def test_chance_nodes_do_not_consume_depth(self):
        result = self.search(board(), "expectiminimax",
                             1, evaluate_fn=lambda b: 0)
        self.assertEqual(result["nodes_expanded"], 8)
        self.assertEqual(result["nodes_visited"], 27)
        self.assertEqual(result["leaves_evaluated"], 19)

    def test_pruning_reduces_work_and_prints_cutoffs(self):
        plain = self.search(board(), depth=3, evaluate_fn=lambda b: 0)
        pruned = self.search(board(), "alpha_beta", 3, evaluate_fn=lambda b: 0)
        self.assertEqual(pruned["best_col"], plain["best_col"])
        self.assertEqual(pruned["heuristic_value"], plain["heuristic_value"])
        self.assertLess(pruned["nodes_visited"], plain["nodes_visited"])
        self.assertGreater(pruned["pruned_branches"], 0)
        self.assertIn("PRUNED", pruned["tree_trace"])
        self.assertIn("bound", pruned["tree_trace"])

    def test_ties_choose_leftmost_legal_column(self):
        position = SmallBoard([[1, 0, 0], [2, 1, 2]])
        for algorithm in ("minimax", "alpha_beta", "expectiminimax"):
            result = self.search(position, algorithm, evaluate_fn=lambda b: 0)
            self.assertEqual(result["best_col"], 1)

    def test_trace_has_values_probabilities_and_player_types(self):
        trace = self.search(board(), "expectiminimax", 2)["tree_trace"]
        for text in ("MAX", "MIN", "CHANCE", "value=", "p=0.6", "K reached"):
            self.assertIn(text, trace)

    def test_default_call_prints_the_returned_tree(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = minimax(board(), 1)
        self.assertEqual(output.getvalue().strip(), result["tree_trace"])

    def test_tracing_can_be_disabled_for_benchmarks(self):
        traced = self.search(board())
        quiet = self.search(board(), collect_trace=False)
        for field in ("best_col", "heuristic_value", "nodes_visited", "nodes_expanded"):
            self.assertEqual(traced[field], quiet[field])
        self.assertEqual(quiet["tree_trace"], "")

    def test_algorithm_wrappers(self):
        for function in (minimax, alpha_beta, expectiminimax):
            self.assertIn(function(board(), 1, print_tree=False)
                          ["best_col"], range(7))

    def test_bad_arguments(self):
        for depth in (0, -1, 1.5, True, "2"):
            with self.assertRaises(ValueError):
                self.search(board(), depth=depth)
        for algorithm in ("unknown", None):
            with self.assertRaises(ValueError):
                self.search(board(), algorithm=algorithm)
        with self.assertRaises(ValueError):
            get_best_move(board(), "minimax", 1, collect_trace=False)
        with self.assertRaises(ValueError):
            self.search(board(), evaluate_fn=lambda b: math.nan)
        with self.assertRaises(ValueError):
            self.search(board(), stochastic_players=(3,))


class ProbabilityTests(unittest.TestCase):
    def test_middle_and_edge(self):
        self.assertEqual(landing_probabilities(
            board(), 3), [(3, .6), (2, .2), (4, .2)])
        self.assertEqual(landing_probabilities(board(), 0), [(0, .6), (1, .4)])
        self.assertEqual(landing_probabilities(board(), 6), [(6, .6), (5, .4)])

    def test_full_neighbors(self):
        position = SmallBoard([[1, 0, 0], [2, 1, 2]])
        self.assertEqual(landing_probabilities(
            position, 1), [(1, .6), (2, .4)])
        position.grid[0][2] = 2
        self.assertEqual(landing_probabilities(position, 1), [(1, 1.0)])

    def test_invalid_intended_column(self):
        for col in (-1, 7, 1.5, True):
            with self.assertRaises(ValueError):
                landing_probabilities(board(), col)
        with self.assertRaises(ValueError):
            landing_probabilities(SmallBoard([[1, 0], [2, 1]]), 0)

    def test_sampler_uses_the_same_distribution(self):
        class RecordingRandom:
            def choices(self, population, weights, k):
                self.arguments = population, weights, k
                return [population[-1]]
        rng = RecordingRandom()
        self.assertEqual(sample_landing_column(board(), 3, rng), 4)
        self.assertEqual(rng.arguments, ([3, 2, 4], [.6, .2, .2], 1))


class EvaluationTests(unittest.TestCase):
    def test_empty_board_is_neutral(self):
        self.assertEqual(default_evaluate(board()), 0)

    def test_full_board_scores_every_window(self):
        # All-AI 6x7: 24 horizontal + 21 vertical + 24 diagonal = 69.
        position = board()
        position.grid = [[2] * 7 for _ in range(6)]
        self.assertEqual(default_evaluate(position), 69)
        position.grid = [[1] * 7 for _ in range(6)]
        self.assertEqual(default_evaluate(position), -69)

    def test_each_direction_and_overlapping_fours(self):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            position = board()
            row, col = (0, 3) if dc == -1 else (0, 0)
            for step in range(4):
                position.grid[row + dr * step][col + dc * step] = 2
            self.assertGreaterEqual(default_evaluate(position), 1)
        position = board()
        position.grid[5][:5] = [2] * 5
        self.assertGreaterEqual(default_evaluate(position), 2)


if __name__ == "__main__":
    unittest.main()
