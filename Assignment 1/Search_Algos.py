import math
from cmath import sqrt
import heapq
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
            return path,len(explored)
        for child in main.children(state):
            states = [item[0] for item in frontier]
            if (child in states or tuple(child) in explored):
                continue
            else:
                frontier.append((child, path+[child]))
    return None,len(explored)


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
            return path,len(explored)
        for child in main.children(state):
            if (tuple(child) in frontier_set or tuple(child) in explored):
                continue
            else:
                frontier.append(child)
                frontier_set.add(tuple(child))
                depth[tuple(child)] = depth[tuple(state)] + 1
                pmap[tuple(child)] = state
    return None,len(explored)


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
            return path,len(explored)
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
    return None,len(explored)


def IDFS(board):
    for L in range(32):  # if L ==32 -> board is unsolvable
        path,explored_l = LDFS(board, L)
        if path is not None:
            return path,explored_l
    return None,explored_l

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

def AStar_Manhattan(board):
    counter=0
    frontier = []
    explored = set()
    heapq.heappush(frontier, (manhattanDistance(board), counter, board, [board]))
    #input manhattan distance, then counter for tie breaks, then the actual board, then the path to the board
    gcost={tuple(board): 0}
    #actual cost to board (step count)

    while frontier:
        f,c,state,path=heapq.heappop(frontier)
        if isGoal(state):
            return path,len(explored)
        explored.add(tuple(state))
        for child in main.children(state):
            if tuple(child) in explored:
                continue

            costChild=len(path)
            if tuple(child) not in gcost or costChild < gcost[tuple(child)]:
                gcost[tuple(child)] = costChild #g(n)
                cost= costChild + manhattanDistance(child) #f(n) = g(n)+ h(n)
                counter = counter + 1 # counter since heaps break ties by comparing the next element
                heapq.heappush(frontier, (cost, counter, child, path+[child]))

    return None,len(explored)


def AStar_Euclidean(board):
    counter = 0
    frontier = []
    explored = set()
    heapq.heappush(frontier, (euclideanDistance(board), counter, board, [board]))
    # input manhattan distance, then counter for tie breaks, then the actual board, then the path to the board
    gcost = {tuple(board): 0}
    # actual cost to board (step count)

    while frontier:
        f, c, state, path = heapq.heappop(frontier)
        if isGoal(state):
            return path,len(explored)
        explored.add(tuple(state))
        for child in main.children(state):
            if tuple(child) in explored:
                continue

            costChild = len(path)
            if tuple(child) not in gcost or costChild < gcost[tuple(child)]:
                gcost[tuple(child)] = costChild  # g(n)
                cost = costChild + euclideanDistance(child)  # f(n) = g(n)+ h(n)
                counter = counter + 1  # counter since heaps break ties by comparing the next element
                heapq.heappush(frontier, (cost, counter, child, path + [child]))

    return None,len(explored)

if __name__ == "__main__":
    board = [5, 1, 4,
             8, 0, 2,
             3, 6, 7]
    path,expl = (BFS(board))
    if path is not None:
        for i,step in enumerate(path):
            print("step no",i)
            print(main.p_board(step))
            print()
    else:
        print("no path")
    print("Explored: ", expl, "nodes")

