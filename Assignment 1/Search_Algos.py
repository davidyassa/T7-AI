import main
from collections import deque

def isGoal(board):
    if(board == [0,1,2,3,4,5,6,7,8]):
        return True
    else:
        return False

def BFS(board):
    frontier = deque(board)
    explored = set()
    while frontier:
        state = frontier.popleft()
        explored.add(state)
        if(isGoal(state)):
            return state
        for child in main.children(board):
            if(child in frontier or child in explored):
                continue
            else:
                frontier.append(child)

    return [0,0,0,0,0,0,0,0,0]

