from search_algos import *


def moves(board, n=3):
    idx = board.index(0)
    row, col = divmod(idx, n)
    moves = []
    if row > 0:
        moves.append(idx - n)
    if row < n - 1:
        moves.append(idx + n)
    if col > 0:
        moves.append(idx - 1)
    if col < n - 1:
        moves.append(idx + 1)
    return moves


def p_board(board):
    b = board
    if isinstance(b, tuple):
        b = list(b)
    res = ""
    for i, x in enumerate(b):
        if x == 0:
            res += "_ "
        else:
            res += (f"{x} ")
        if 0 < i < len(b)-1 and (i+1) % 3 == 0:
            res += "\n"
    return res.strip()


def child(board, idx):
    res = board[:]
    res[res.index(0)], res[idx] = res[idx],  res[res.index(0)]
    return res


def children(board):
    res = []
    for i in moves(board):
        res.append(child(board, i))
    return res


board = [1, 2, 5,
         3, 0, 4,
         6, 7, 8]

# print(f"legal moves = ", moves(board))
# print(f"board =\n{p_board(board)}\n")
# for i, c in enumerate(children(board)):
#   print(f"child[{i}] =\n{p_board(c)}\n")

if __name__ == "__main__":
    board = [1, 2, 5,
             3, 0, 4,
             6, 7, 8]

    for algo in [BFS, DFS, IDFS, AStar_Manhattan,AStar_Euclidean]:
        path,explored_l = algo(board)
        print(f"{algo.__name__} took {len(path)} steps")
        # print(p_board(path[-1]))
        if algo.__name__ != "DFS":
            for i, step in enumerate(path):
                print(f"Step #{i+1}:")
                print(p_board(step))
                print()
        print("Explored: ", explored_l, "Nodes.")
        print("-" * 50)

