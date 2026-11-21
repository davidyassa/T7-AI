class board:

    def __init__(self, l=6, w=7):
        if l < 6 or w < 7:
            raise ValueError("Rows cant be less than 6 and columns cant be less than 7")
        self.rows = l
        self.cols = w
        self.grid = [[0] * self.cols for _ in range(self.rows)]

    def printBoard(self):
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.grid[i][j])
            print(row)

    def play_move(self, player, column):  # player=1 human - player=2 AI
        if self.grid[0][column] != 0:
            return False
        if self.grid[self.rows - 1][column] == 0:
            self.grid[self.rows - 1][column] = player
            return True
        else:
            for i in range(self.rows):
                if self.grid[i][column] != 0:
                    self.grid[i - 1][column] = player
                    return True

    def isFull(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 0:
                    return False
        return True

    def moves(self):
        res = [-1] * self.cols
        for r in range(self.rows - 1, 0 - 1, -1):
            if -1 not in res:
                break
            for c in range(self.cols):
                if res[c] != -1 or self.grid[0][c] != 0:
                    continue
                if self.grid[r][c] == 0:
                    res[c] = r
        return res

    def count_connect_fours(self, player):
        count = 0

        for r in range(self.rows - 3):
            for c in range(self.cols):
                if (
                    self.grid[r][c] == player
                    and self.grid[r + 1][c] == player
                    and self.grid[r + 2][c] == player
                    and self.grid[r + 3][c] == player
                ):
                    count += 1

        for r in range(self.rows):
            for c in range(self.cols - 3):
                if (
                    self.grid[r][c] == player
                    and self.grid[r][c + 1] == player
                    and self.grid[r][c + 2] == player
                    and self.grid[r][c + 3] == player
                ):
                    count += 1

        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if (
                    self.grid[r][c] == player
                    and self.grid[r + 1][c + 1] == player
                    and self.grid[r + 2][c + 2] == player
                    and self.grid[r + 3][c + 3] == player
                ):
                    count += 1

        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if (
                    self.grid[r][c] == player
                    and self.grid[r - 1][c + 1] == player
                    and self.grid[r - 2][c + 2] == player
                    and self.grid[r - 3][c + 3] == player
                ):
                    count += 1

        return count

    def count_threats(b, player): #3 in a row return 1 if playable , 0.3 if next turn
        score = 0.0

        
        for r in range(b.rows - 3):
            for c in range(b.cols):
                cells = [b.grid[r][c], b.grid[r + 1][c], b.grid[r + 2][c], b.grid[r + 3][c]]
                if cells.count(player) == 3 and cells.count(0) == 1:
                    er = r + cells.index(0)
                    if er == b.rows - 1 or b.grid[er + 1][c] != 0:
                        score += 1.0
                    else:
                        score += 0.3

        
        for r in range(b.rows):
            for c in range(b.cols - 3):
                cells = [b.grid[r][c], b.grid[r][c + 1], b.grid[r][c + 2], b.grid[r][c + 3]]
                if cells.count(player) == 3 and cells.count(0) == 1:
                    ec = c + cells.index(0)
                    if r == b.rows - 1 or b.grid[r + 1][ec] != 0:
                        score += 1.0
                    else:
                        score += 0.3

        
        for r in range(b.rows - 3):
            for c in range(b.cols - 3):
                cells = [b.grid[r][c], b.grid[r + 1][c + 1], b.grid[r + 2][c + 2], b.grid[r + 3][c + 3]]
                if cells.count(player) == 3 and cells.count(0) == 1:
                    idx = cells.index(0)
                    er, ec = r + idx, c + idx
                    if er == b.rows - 1 or b.grid[er + 1][ec] != 0:
                        score += 1.0
                    else:
                        score += 0.3

        
        for r in range(3, b.rows):
            for c in range(b.cols - 3):
                cells = [b.grid[r][c], b.grid[r - 1][c + 1], b.grid[r - 2][c + 2], b.grid[r - 3][c + 3]]
                if cells.count(player) == 3 and cells.count(0) == 1:
                    idx = cells.index(0)
                    er, ec = r - idx, c + idx
                    if er == b.rows - 1 or b.grid[er + 1][ec] != 0:
                        score += 1.0
                    else:
                        score += 0.3

        return score 
    
    def count_forks(b, player): # forks are empty cells that complete 2 or more 3 in a rows at once
        hits = [[0] * b.cols for _ in range(b.rows)]

        
        for r in range(b.rows - 3):
            for c in range(b.cols):
                cells = [b.grid[r][c], b.grid[r + 1][c], b.grid[r + 2][c], b.grid[r + 3][c]]
                if cells.count(player) == 3 and cells.count(0) == 1:
                    hits[r + cells.index(0)][c] += 1

  
        for r in range(b.rows):
            for c in range(b.cols - 3):
                cells = [b.grid[r][c], b.grid[r][c + 1], b.grid[r][c + 2], b.grid[r][c + 3]]
                if cells.count(player) == 3 and cells.count(0) == 1:
                    hits[r][c + cells.index(0)] += 1

        
        for r in range(b.rows - 3):
            for c in range(b.cols - 3):
                cells = [b.grid[r][c], b.grid[r + 1][c + 1], b.grid[r + 2][c + 2], b.grid[r + 3][c + 3]]
                if cells.count(player) == 3 and cells.count(0) == 1:
                    idx = cells.index(0)
                    hits[r + idx][c + idx] += 1

        
        for r in range(3, b.rows):
            for c in range(b.cols - 3):
                cells = [b.grid[r][c], b.grid[r - 1][c + 1], b.grid[r - 2][c + 2], b.grid[r - 3][c + 3]]
                if cells.count(player) == 3 and cells.count(0) == 1:
                    idx = cells.index(0)
                    hits[r - idx][c + idx] += 1

        forks = 0
        for r in range(b.rows):
            for c in range(b.cols):
                if hits[r][c] >= 2:
                    forks += 1
        return forks