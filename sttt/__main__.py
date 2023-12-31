import os
"""...Fix input bug..."""
def cls():
    os.system('cls' if os.name in ('nt', 'dos') else 'clear')

disabled = '\033[02m'
highlight_color = '\033[47m'+'\033[30m'
X_color = '\033[01m'+'\033[93m'
Y_color = '\033[01m'+'\033[36m'
RESET = "\033[0m"
"""The instructions for color and color reset """

def logoDisplay():
    print('\033[35m',
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
    ,end=RESET)

'''Title ascii art which will be in purple'''
class tic_tac_toe:
    def __init__(self):
        self.board = [
            ['1','2','3'],
            ['4','5','6'],
            ['7','8','9'],
        ]
            
        self.won = None
    """Printing Building small board inside the large one """    
    def showboard (self,lineno = None ):
        if lineno == None :
            for line in self.board:
                print("","  | ".join(line))
                print("---|---|---")
        else:
            return " │ ".join(self.board[lineno])
    '''Checking for winner in the smaller board for each player'''
    def checkWin(self):
        wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for win in wins:
            if self.board[win[0]//3][win[0]%3] == self.board[win[1]//3][win[1]%3] == self.board[win[2]//3][win[2]%3]:#Comparing the 
                return self.board[win[1]//3][win[1]%3]                                          # elements of the board to check for winner
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
    def showboard (self, selected_board = None ):    
        print("╔"+"═"*(12*3-1)+"╗")

        for i in range(3):
            for k in range(3):  

                print('║ ',end="")

                for j in range(3):

                    box_disabled = self.sboard[i][j].won!=None
                    highlight = i*3+j == selected_board
        
                    line = ""
                    for c in self.sboard[i][j].showboard(k):
                        line += disabled if box_disabled else highlight_color if highlight else RESET
                        if c=="X":
                            line+=X_color+c+RESET # +last_color
                        elif c=="O":
                            line+=Y_color+c+RESET # +last_color
                        else:
                            line+=c
                    print(line, end=RESET+" ║ ")
                    # print(, end="")
    
                print()
                if k == 2 :
                    print("║"+"═"*(12*3-1)+"║")  # ╬
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
    x = 1
    player = "X" 
    boardno = 4
    Tic = super_tic_tac_toe()
    ele_check = tic_tac_toe()
    while True:
        cls()
        logoDisplay()
        Tic.showboard(boardno)
        print()
        if Tic.checkWin() != None :
            """Print winner"""
            print(f"{Tic.checkWin()} is the winner !!!")
            break
        while Tic.get_Board(boardno).won != None :
            boardno = int(input(f"Please enter a board other then {boardno+1} : "))-1
        #The input peradox problem
        while True :
            try:
                Inp = int(input(f"{(X_color if player == 'X' else Y_color)+player} Input : {RESET}"))-1
                if Inp not in range(9):
                    print(f"{Inp+1} is not on the board")
                    continue
                if Tic.get_Board(boardno).board[Inp//3][Inp%3] in ["X","O"]:
                    print("The box is already filled")
                    continue
                break
            except KeyboardInterrupt:
                exit()
            except:
                print(f"Invalid input!")
                continue 
        # the above loop will force the user to enter a valid input.INPUT PARADOX SOLVED 
              
         
        Tic.get_Board(boardno).play(Inp,player)
        
        boardno = Inp
        player = "O" if player == "X" else "X"

if __name__ == "__main__":
    playgame()

    

# Testing sequence
# [(0,0),(6,0),(2,0),(4,1),(0,"b2",4),(4,2),(6,4),(8,0),("b8",4,"b2",8),(6,8),(2,"Winner!!")]    