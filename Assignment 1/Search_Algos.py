import math
from cmath import sqrt

import main
from collections import deque


def isGoal(board):
    if (board == [0, 1, 2, 3, 4, 5, 6, 7, 8]):
        return True
    else:
        return False


def BFS(board):
    frontier = deque([(board, [board])])
    explored = set()
    while frontier:
        state, path = frontier.popleft()
        explored.add(tuple(state))
        if (isGoal(state)):
            return path
        for child in main.children(state):
            states = [item[0] for item in frontier]
            if (child in states or tuple(child) in explored):
                continue
            else:
                frontier.append((child, path+[child]))
    return None


def DFS(board):
    depth = {tuple(board): 0}
    frontier = deque([board])  # DFS is too memory consuming
    explored = set()
    # for faster comparisons to avoid long stack loops
    frontier_set = {tuple(board)}
    pmap = {tuple(board): None}  # map node to parent
    while frontier:
        state = frontier.pop()
        frontier_set.remove(tuple(state))
        explored.add(tuple(state))
        if (isGoal(state)):
            path = []
            current = state
            while current:
                path.append(current)
                current = pmap[tuple(current)]
            path.reverse()
            return path
        for child in main.children(state):
            if (tuple(child) in frontier_set or tuple(child) in explored):
                continue
            else:
                frontier.append(child)
                frontier_set.add(tuple(child))
                depth[tuple(child)] = depth[tuple(state)] + 1
                pmap[tuple(child)] = state
    return None


def LDFS(board, L):
    depth = {tuple(board): 0}
    frontier = deque([board])  # DFS is too memory consuming
    explored = set()
    # for faster comparisons to avoid long stack loops
    frontier_set = {tuple(board)}
    pmap = {tuple(board): None}  # map node to parent
    while (frontier):
        state = frontier.pop()
        frontier_set.remove(tuple(state))
        explored.add(tuple(state))
        if (isGoal(state)):
            path = []
            current = state
            while current:
                path.append(current)
                current = pmap[tuple(current)]
            path.reverse()
            return path
        for child in main.children(state):
            cd = depth[tuple(state)] + 1
            child_t = tuple(child)
            if cd > L:
                continue
            if child_t in frontier_set:
                if cd < depth[child_t]:
                    depth[child_t] = cd
                    pmap[child_t] = state
                continue
            elif child_t in explored:
                if cd >= depth[child_t]:  # not an improvement, skip
                    continue
                explored.remove(child_t)  # shallower route: re-open it
            frontier.append(child)
            frontier_set.add(tuple(child))
            depth[tuple(child)] = depth[tuple(state)] + 1
            pmap[tuple(child)] = state
    return None


def IDFS(board):
    for L in range(32):  # if L ==32 -> board is unsolvable
        path = LDFS(board, L)
        if path is not None:
            return path
    return None

def manhattanDistance(board):

    sum=0
    for i in (board):
        n=board[i]
        x=n%3
        y=n//3
        xi=i%3
        yi=i//3
        h=abs(x-xi)+abs(y-yi)
        sum+=h
    return sum

def euclideanDistance(board):
    sum = 0
    for i in (board):
        n = board[i]
        x = n % 3
        y = n // 3
        xi = i % 3
        yi = i // 3
        hx = (x - xi)*(x- xi)
        hy = (y - yi)*(y-yi)
        h=hx+hy
        h=math.sqrt(h)
        sum += h
    return sum

