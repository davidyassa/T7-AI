import main
from collections import deque

def isGoal(board):
    if(board == [0,1,2,3,4,5,6,7,8]):
        return True
    else:
        return False

def BFS(board):
    frontier = deque([(board,[board])])
    explored = set()
    while frontier:
        state,path = frontier.popleft()
        explored.add(tuple(state))
        if(isGoal(state)):
            return path
        for child in main.children(state):
            states=[item[0] for item in frontier]
            if(states in frontier or tuple(child) in explored):
                continue
            else:
                frontier.append((child,path+[child]))

    return None

