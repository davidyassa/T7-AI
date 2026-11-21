"""Run the original board demo, try the searches, or collect benchmark data."""

import csv
import random

from Algos import get_best_move, sample_landing_column, evaluate
from board import board


ALGORITHMS = ("minimax", "alpha_beta", "expectiminimax")


def board_demo():
    """The manual test that was previously at the bottom of board.py."""
    board1 = board()
    board1.printBoard()
    board1.play_move(1, 0)
    print()
    board1.printBoard()
    print()
    board1.play_move(1, 5)
    board1.printBoard()
    print()
    board1.play_move(2, 0)
    board1.printBoard()
    print(f"Board is full: {board1.isFull()}")
    return board1


def sample_board():
    position = board()
    for player, col in ((1, 0), (1, 5), (2, 0)):
        position.play_move(player, col)
    return position


def search_demo(position, algorithms, depth, show_tree, stochastic_players):
    for algorithm in algorithms:
        print(f"\n{algorithm}, K={depth} (columns start at 0)")
        result = get_best_move(
            position, algorithm, depth, print_tree=show_tree,
            stochastic_players=stochastic_players,
        )
        print(f"Chosen column: {result['best_col']}")
        print(f"Value: {result['heuristic_value']:.6g}")
        print(f"Nodes expanded: {result['nodes_expanded']}, "
              f"visited: {result['nodes_visited']}, "
              f"leaves: {result['leaves_evaluated']}")
        print(f"Search time: {result['time_taken']:.6f} seconds")
        if algorithm == "expectiminimax" and result["best_col"] is not None:
            actual_col = sample_landing_column(
                position, result["best_col"], rng=random.Random(42),
            )
            print(f"Example sampled landing column: {actual_col}")
    print("\nEach search used the same board; the demo did not apply the AI move.")


def benchmark(position, algorithms, depths, stochastic_players, csv_path=None):
    """Use the same board and evaluator for every algorithm and depth."""
    records = []
    print("\nAlgorithm           K   Expanded   Visited     Seconds    Column       Value")
    for depth in depths:
        for algorithm in algorithms:
            result = get_best_move(
                position, algorithm, depth, print_tree=False, collect_trace=False,
                stochastic_players=stochastic_players,
            )
            record = {"algorithm": algorithm, "K": depth}
            record.update({key: value for key, value in result.items()
                           if key != "tree_trace"})
            records.append(record)
            print(f"{algorithm:<19} {depth:>2} "
                  f"{result['nodes_expanded']:>10} {result['nodes_visited']:>9} "
                  f"{result['time_taken']:>11.6f} {str(result['best_col']):>9} "
                  f"{result['heuristic_value']:>11.6g}")
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nSaved measurements to {csv_path}")
    return records


def main(algorithm="all", depth=2, show_tree=True, board_only=False,
         run_benchmark=False, benchmark_depths=(1, 2, 3), csv_path=None,
         stochastic_players=(1, 2)):
    if algorithm not in ("all",) + ALGORITHMS:
        raise ValueError("Choose all, minimax, alpha_beta, or expectiminimax")
    if depth < 1 or not benchmark_depths or any(k < 1 for k in benchmark_depths):
        raise ValueError("Depths must be at least 1")
    algorithms = ALGORITHMS if algorithm == "all" else (algorithm,)
    if run_benchmark:
        benchmark(sample_board(), algorithms, benchmark_depths,
                  stochastic_players, csv_path)
    else:
        position = board_demo()
        if not board_only:
            search_demo(position, algorithms, depth,
                        show_tree, stochastic_players)


def play_game():
    rows = int(input("Enter rows (>= 6): ") or 6)
    cols = int(input("Enter columns (>= 7): ") or 7)
    depth = int(input("Enter max depth K (>= 1): ") or 3)
    print("Algorithms: 1 = minimax | 2 = alpha_beta | 3 = expectiminimax")
    choice = input("Choose algorithm (1/2/3): ").strip()
    algo_map = {"1": "minimax", "2": "alpha_beta", "3": "expectiminimax"}
    algorithm = algo_map.get(choice, "alpha_beta")

    game = board(rows, cols)
    print(f"\nStarting Connect 4 ({rows}x{cols}) | Algorithm: {algorithm} | K={depth}")
    game.printBoard()
    print(f"Initial Board Heuristic: {evaluate(game):.2f}")

    while not game.isFull():
        # --- Human Turn (Player 1) ---
        valid_cols = [c for c, r in enumerate(game.moves()) if r != -1]
        while True:
            try:
                col = int(input(f"\nHuman turn - choose column {valid_cols}: "))
                if col in valid_cols:
                    break
            except ValueError:
                pass
            print("Invalid column. Try again.")

        if algorithm == "expectiminimax":
            actual_col = sample_landing_column(game, col)
            print(f"Human chose col {col} -> disc landed in col {actual_col}")
            game.play_move(1, actual_col)
        else:
            game.play_move(1, col)

        game.printBoard()
        print(f"Board Heuristic after Human move: {evaluate(game):.2f}")

        if game.isFull():
            break

        # --- AI Turn (Player 2) ---
        print("\nAI thinking...")
        result = get_best_move(
            game,
            algorithm,
            depth,
            evaluate_fn=evaluate,
            print_tree=True,
        )
        chosen = result["best_col"]

        if algorithm == "expectiminimax":
            actual_col = sample_landing_column(game, chosen)
            print(f"AI chose col {chosen} -> disc landed in col {actual_col}")
            game.play_move(2, actual_col)
        else:
            print(f"AI chose col {chosen}")
            game.play_move(2, chosen)

        game.printBoard()
        print(f"Board Heuristic after AI move: {evaluate(game):.2f} | "
              f"Search Root Value: {result['heuristic_value']:.2f}")
        print(f"Nodes Expanded: {result['nodes_expanded']} | "
              f"Time: {result['time_taken']:.4f}s")

    # --- Game Over: Count Connected-Fours ---
    human_score = game.count_connect_fours(1)
    ai_score = game.count_connect_fours(2)
    print("\n=== BOARD FULL - FINAL SCORE ===")
    print(f"Human Connected-4s: {human_score} | AI Connected-4s: {ai_score}")
    if ai_score > human_score:
        print("Winner: AI")
    elif human_score > ai_score:
        print("Winner: Human")
    else:
        print("Result: Draw")
        
        
if __name__ == "__main__":
    # Change these arguments to choose what you want to run.
    main(
        algorithm="all",              # minimax, alpha_beta, expectiminimax, or all
        depth=2,
        show_tree=True,
        board_only=False,
        run_benchmark=False,
        benchmark_depths=(1, 2, 3),
        csv_path=None,                # e.g. "examples/benchmark.csv"
        stochastic_players=(1, 2),      # (2,) for chance on AI moves only
        
    )
    play_game()
