from Search_Algos import *
from pyboard import *

if __name__ == "__main__":
    board = [1, 2, 5,
             3, 0, 4,
             6, 7, 8]

    for algo in [BFS, DFS, IDFS, AStar_Manhattan, AStar_Euclidean]:
        path, explored_l = algo(board)
        print(f"{algo.__name__} took {len(path)} steps")
        if algo.__name__ != "DFS":
            for i, step in enumerate(path):
                print(f"Step #{i+1}:")
                print(p_board(step))
                print()
        print("Explored: ", explored_l, "Nodes.")
        print("-" * 50)
