"""
8-Puzzle Search Algorithms Benchmark Suite.
Benchmarks BFS, DFS, IDFS, AStar_Manhattan, and AStar_Euclidean.
Generates performance comparison tables and plots benchmark_chart.png.

Run with: uv run python benchmark.py
Or with custom board: uv run python benchmark.py --board "1,2,5,3,4,0,6,7,8"
"""

import sys
import time
import argparse
from typing import List

# Import algorithms from Search_Algos
try:
    try:
        from Search_Algos import BFS, DFS, IDFS, AStar_Manhattan, AStar_Euclidean
    except ImportError:
        from search_algos import BFS, DFS, IDFS, AStar_Manhattan, AStar_Euclidean
except ImportError as e:
    print(f"[ERROR] Could not import algorithms from Search_Algos: {e}")
    print("Please make sure Search_Algos.py is in the current directory.")
    sys.exit(1)


def parse_board(board_str: str) -> List[int]:
    parts = board_str.replace(",", " ").strip().split()
    return [int(x) for x in parts]


def run_benchmark(board: List[int], save_plot: bool = True):
    print("=" * 80)
    print("8-PUZZLE ALGORITHM BENCHMARK SUITE (Alexandria University - Assignment 1)")
    print(f"Initial Board: {board}")
    print("=" * 80)
    print(f"{'Algorithm':<22} | {'Cost (Moves)':<12} | {'Nodes Explored':<16} | {'Depth':<8} | {'Time (ms)':<10}")
    print("-" * 80)

    algos = [
        ("BFS", BFS),
        ("DFS", DFS),
        ("IDFS", IDFS),
        ("A* (Manhattan)", AStar_Manhattan),
        ("A* (Euclidean)", AStar_Euclidean),
    ]

    labels = []
    costs = []
    nodes_explored = []
    times_ms = []

    for name, func in algos:
        try:
            t0 = time.perf_counter()
            path, explored_l = func(list(board))
            elapsed = time.perf_counter() - t0

            # Calculate path cost
            cost = len(path)
            if path and list(path[0]) == list(board):
                cost = max(0, len(path) - 1)

            t_ms = elapsed * 1000.0
            print(f"{name:<22} | {cost:<12} | {explored_l:<16,d} | {cost:<8} | {t_ms:<10.2f}")

            labels.append(name)
            costs.append(cost)
            nodes_explored.append(explored_l)
            times_ms.append(t_ms)
        except Exception as e:
            print(f"{name:<22} | {'FAILED':<12} | {'-':<16} | {'-':<8} | {str(e):<10}")

    print("=" * 80)

    print("\nHeuristic Analysis (Assignment 1, Section 2.2):")
    print("• Manhattan Distance dominates Euclidean Distance: h_Manhattan >= h_Euclidean.")
    print("• Both are admissible, but A* Manhattan expands fewer nodes and runs faster.\n")

    # Save Matplotlib Chart
    if save_plot and labels:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
            fig.patch.set_facecolor("#0F172A")

            # 1. Nodes Explored Plot
            bars1 = ax1.bar(range(len(labels)), nodes_explored, color="#38BDF8", edgecolor="#0284C7")
            ax1.set_title("Search Space (Nodes Explored)", color="#F8FAFC", fontsize=11, fontweight="bold")
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=25, ha="right", color="#94A3B8")
            ax1.set_facecolor("#1E293B")
            ax1.tick_params(colors="#94A3B8")
            ax1.grid(True, linestyle="--", alpha=0.2, color="#64748B")

            for bar in bars1:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f"{int(yval):,}",
                         ha='center', va='bottom', color='#38BDF8', fontsize=8)

            # 2. Running Time Plot
            bars2 = ax2.bar(range(len(labels)), times_ms, color="#A855F7", edgecolor="#7E22CE")
            ax2.set_title("Execution Time (ms)", color="#F8FAFC", fontsize=11, fontweight="bold")
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=25, ha="right", color="#94A3B8")
            ax2.set_facecolor("#1E293B")
            ax2.tick_params(colors="#94A3B8")
            ax2.grid(True, linestyle="--", alpha=0.2, color="#64748B")

            for bar in bars2:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.1f}ms",
                         ha='center', va='bottom', color='#C084FC', fontsize=8)

            plt.tight_layout()
            out_filename = "benchmark_chart.png"
            plt.savefig(out_filename, dpi=200, facecolor=fig.get_facecolor())
            print(f"Chart saved to: {out_filename}")
        except Exception as pe:
            print(f"(Matplotlib chart skipped: {pe})")


def main():
    parser = argparse.ArgumentParser(description="8-Puzzle Benchmark Runner")
    parser.add_argument(
        "--board",
        type=str,
        default="1,2,5,3,0,4,6,7,8",
        help="Initial board as 9 comma-separated digits (e.g. 1,2,5,3,0,4,6,7,8)",
    )
    args = parser.parse_args()
    board = parse_board(args.board)
    run_benchmark(board)


if __name__ == "__main__":
    main()