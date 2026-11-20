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


play = board()
play.printBoard()
play.play_move(1, 0)
print("")
play.printBoard()
print("")
play.play_move(1, 5)
play.printBoard()
print("")
play.play_move(2, 0)
play.printBoard()
