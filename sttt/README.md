# Super Tic Tac Toe Console Game

This Python console-based game is an implementation of the classic Tic Tac Toe with an added twist - Super Tic Tac Toe. The game is played on a larger board consisting of nine smaller Tic Tac Toe boards arranged in a 3x3 grid.

## Rules of the Game

1. **Board Layout**:
   - The game is played on a 3x3 grid, where each cell of the grid contains another 3x3 Tic Tac Toe board.
   - The larger 3x3 grid represents the global board, and each smaller 3x3 board is referred to as a local board.

2. **Gameplay**:
   - Players take turns to place their respective markers ('X' or 'O') on the local board.
   - The first player marks an empty cell on any of the local boards.
   - The placement of the marker dictates the next local board where the opponent must place their marker.
   - For example, if a player places their marker in the top-right cell of a local board, the opponent must play in the local board corresponding to the top-right cell of the global board.
   - Players continue taking turns until one wins or the entire global board is filled.

3. **Winning the Game**:
   - To win the game, a player must win three local boards in a row - horizontally, vertically, or diagonally.
   - Once a local board is won by a player, it becomes "owned" by that player, and subsequent moves made by the opponent in that board don't count.
   - The first player to win three local boards in a row or to achieve a draw across the global board wins the game.

4. **Game Over**:
   - The game ends when a player wins three local boards in a row or when the entire global board is filled with no player achieving three local board wins.

## How to Play

1. **Setup**:
   - Ensure you have Python 3.x installed on your system.
   - Download the Python script for the Super Tic Tac Toe game.

2. **Running the Game**:
   - Open your terminal or command prompt.
   - Navigate to the directory where the Python script is located.
   - Run the script using the command: `python super_tic_tac_toe.py` or `python3 super_tic_tac_toe.py` depending on your Python installation.

3. **Game Controls**:
   - Follow the prompts displayed in the console to input your moves.
   - Input the row and column of the local board (1-3) where you want to place your marker.
   - Input the row and column of the cell (1-3) within that local board where you want to place your marker.
   - The game will continue until a player wins or the entire board is filled.

4. **Winning the Game**:
   - Achieve three local board wins in a row - horizontally, vertically, or diagonally - to win the game.
   - The game will announce the winner or declare a draw once the game is over.

Enjoy playing Super Tic Tac Toe!

## Additional Notes

- This game is designed for console-based interaction and does not have a graphical user interface.
- It's a strategic game that requires planning and a good understanding of the game's mechanics to win.

For any inquiries or issues related to the game, feel free to contact the developer at [zamiulrashid1@gmail.com , abyashrirproyas@gmail.com](mailto:zamiulrashid1@gmail.com,mailto:abyashrirproyas@gmail.com).

Have fun!
