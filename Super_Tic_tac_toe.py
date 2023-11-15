import os

def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)



class tic_tac_toe:
    def __init__(self):
        self.board = [
        ['0','1','2'],
        ['3','4','5'],
        ['6','7','8'],
    ]
    def showboard (self,lineno = None ):
        if lineno == None :
            for line in self.board:
                print("","  | ".join(line))
                print("---|---|---")
        else:
            return(" | ".join(self.board[lineno])) 
    def play (self,num,C_P):
        self.board[num//3][num%3]=C_P
    # def checkWin(self , xState, zState):
    #     wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    #     for win in wins:
    #         if(sum(xState[win[0]], xState[win[1]], xState[win[2]]) == 3):
    #             print("X Won the match")
    #             return 1
    #         if(sum(zState[win[0]], zState[win[1]], zState[win[2]]) == 3):
    #             print("O Won the match")
    #             return 0
    #     return -1                   
        
class super_tic_tac_toe:
    def __init__(self) :
        self.sboard =  [
            [tic_tac_toe(),tic_tac_toe(),tic_tac_toe()],    
            [tic_tac_toe(),tic_tac_toe(),tic_tac_toe()],
            [tic_tac_toe(),tic_tac_toe(),tic_tac_toe()],
        ]
    def showboard (self,lineno = None ):    
        for i in range(3):
            for k in range(3):  
                print(' ',end="")
                for j in range(3):
                    print(self.sboard[i][j].showboard(k),end=" || ")
                print()
                if k == 2 :
                    print("===|===|===||"*3)
                else:
                    print("---|---|---||"*3)
    def get_Board(self,boardno):
        return self.sboard[boardno//3][boardno%3]

player = "X" 
boardno = 4
Tic = super_tic_tac_toe()
while True:
    clearConsole()
    Tic.showboard()
    print()
    Inp = int(input(f"{player} Input : "))
    Tic.get_Board(boardno).play(Inp,player)
    boardno = Inp
    
      
    player = "O" if player == "X" else "X"
        
  
    


