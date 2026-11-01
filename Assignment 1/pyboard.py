# Helper functions for board manipulation

def moves(board, n=3):
    b = board
    if isinstance(b, tuple):
        b = list(b)
    idx = b.index(0)
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
