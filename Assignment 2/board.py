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

        for r in range(self.rows):
            for c in range(self.cols):
                if (
                    self.grid[r][c] == player
                    and self.grid[r + 1][c] == player
                    and self.grid[r + 2][c] == player
                    and self.grid[r + 3][c] == player
                ):
                    count += 1
                    
        for r in range(self.rows):
            for c in range(self.cols):
                if (
                    self.grid[r][c] == player
                    and self.grid[r][c + 1] == player
                    and self.grid[r][c + 2] == player
                    and self.grid[r][c + 3] == player
                ):
                    count += 1
                    
        for r in range(self.rows):
            for c in range(self.cols):
                if (
                    self.grid[r][c] == player
                    and self.grid[r + 1][c + 1] == player
                    and self.grid[r + 2][c + 2] == player
                    and self.grid[r + 3][c + 3] == player
                ):
                    count += 1
                    
        for r in range(self.rows):
            for c in range(self.cols):
                if (
                    self.grid[r][c] == player
                    and self.grid[r - 1][c + 1] == player
                    and self.grid[r - 2][c - 2] == player
                    and self.grid[r - 3][c + 3] == player
                ):
                    count += 1
                    
        return count
