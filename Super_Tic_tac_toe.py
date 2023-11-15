import os

def cls():
    os.system('cls' if os.name in ('nt', 'dos') else 'clear')

def logoDisplay():
    print(
        """
   ____                                             
  / __/_ _____  ___ ____                            
 _\ \/ // / _ \/ -_) __/                            
/___/\_,_/ .__/\__/_/                               
 _______/_/       ______              ______        
/_  __(_)___ ___ /_  __/__ _____ ___ /_  __/__  ___ 
 / / / / __//___/ / / / _ `/ __//___/ / / / _ \/ -_)
/_/ /_/\__/      /_/  \_,_/\__/      /_/  \___/\__/ 
"""
    )


class tic_tac_toe:
    def __init__(self):
        self.board = [
            
            ['0','1','2'],
            ['3','4','5'],
            ['6','7','8'],
        ]
            
        self.won = None
        
    def showboard (self,lineno = None ):
        if lineno == None :
            for line in self.board:
                print("","  | ".join(line))
                print("---|---|---")
        else:
            return(" │ ".join(self.board[lineno])) 
    
    def checkWin(self):
        wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for win in wins:
            if self.board[win[0]//3][win[0]%3] == self.board[win[1]//3][win[1]%3] == self.board[win[2]//3][win[2]%3]:
                return self.board[win[1]//3][win[1]%3]    
            return None
        
    def play (self,num,C_P):
        self.board[num//3][num%3]=C_P
            
        self.won = self.checkWin() 
                     
                
        
class super_tic_tac_toe:
    def __init__(self) :
        self.sboard =  [
            [tic_tac_toe(),tic_tac_toe(),tic_tac_toe()],    
            [tic_tac_toe(),tic_tac_toe(),tic_tac_toe()],
            [tic_tac_toe(),tic_tac_toe(),tic_tac_toe()],
        ]
    def showboard (self,lineno = None ):    
        print("╔"+"═"*(12*3-1)+"╗")
        for i in range(3):
            for k in range(3):  
                print('║ ',end="")
                for j in range(3):
                    print(self.sboard[i][j].showboard(k),end=" ║ ")
                print()
                if k == 2 :
                    print("╬"+"═"*(12*3-1)+"╬")
                else:
                    print("║" + "---┼---┼---║"*3)
    def get_Board(self,boardno):
        return self.sboard[boardno//3][boardno%3]
    
    def checkWin(self):
        wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for win in wins:
            if self.sboard[win[0]//3][win[0]%3].won == self.sboard[win[1]//3][win[1]%3].won == self.sboard[win[2]//3][win[2]%3].won:
                return self.sboard[win[1]//3][win[1]%3].won    
            return None

def playgame():
    player = "X" 
    boardno = 4
    Tic = super_tic_tac_toe()
    while True:
        cls()
        logoDisplay()
        Tic.showboard()
        print()
        if Tic.checkWin() != None :
            """Print winner"""
            print(f"{player} is the winner !!!")
            
            break
        
        Inp = int(input(f"{player} Input : "))
        if Tic.get_Board(Inp).won != None :
        #    print(f"Please choose a board other then {Inp}")
           boardno = int(input(f"Please choose a board other then {Inp}"))
           Inp = int(input(f"{player} Input : "))
        Tic.get_Board(boardno).play(Inp,player)
        boardno = Inp
        
        
        player = "O" if player == "X" else "X"
        
playgame()
        
    
  
    

