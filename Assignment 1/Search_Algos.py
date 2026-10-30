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
            if(child in states or tuple(child) in explored):
                continue
            else:
                frontier.append((child,path+[child]))

    return None

def DFS(board):
    frontier = deque([board]) #DFS is too memory consuming
    explored = set()
    frontier_set = {tuple(board)} # for faster comparisons to avoid long stack loops
    pmap={tuple(board):None}
    while frontier:
        state = frontier.pop()
        frontier_set.remove(tuple(state))
        explored.add(tuple(state))
        if(isGoal(state)):
            path=[]
            current=state
            while current:
                path.append(current)
                current=pmap[tuple(current)]
            path.reverse()
            return path
        for child in main.children(state):

            if(tuple(child) in frontier_set or tuple(child) in explored):
                continue
            else:
                frontier.append(child)
                frontier_set.add(tuple(child))
                pmap[tuple(child)]=state

    return None

