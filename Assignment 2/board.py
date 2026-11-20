class board:
    
    def __init__(self,l=6,w=7):
        if l<6 or w<7:
            raise ValueError("Rows cant be less than 6 and columns cant be less than 7")
        self.rows=l
        self.cols=w
        self.grid=[[0]*self.cols for _ in range(self.rows)]
        
    def printBoard(self):
        for i in range(self.rows):
            row=[]
            for j in range(self.cols):
                row.append(self.grid[i][j])
            print(row)


play=board()
play.printBoard()