import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.game.tic_tac_toe import TicTacToe

game = TicTacToe()
game.print_board()
print("Comece a jogar! (X = 1, O = -1)")

while not game.is_game_over():
    print(f"Jogador {'X' if game.current_player == 1 else 'O'}")
    pos = int(input("Escolha a posição (0-8): "))
    if not game.make_move(pos, game.current_player):
        print("Jogada inválida! Tente novamente.")
    game.print_board()
    print()

if game.winner == 1:
    print("X venceu!")
elif game.winner == -1:
    print("O venceu!")
elif game.winner == 0:
    print("Empate!")
else:
    print("Jogo em andamento.")