import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural_network.mlp import MLP

# Inicializa a rede neural
mlp = MLP()

# Estado do tabuleiro: tudo vazio
tabuleiro = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# Testa o método predict
saida = mlp.predict(tabuleiro)
print("Saída da rede (probabilidades):", saida)
print("Shape da saída:", saida.shape)

# Testa o método get_best_move
melhor_jogada = mlp.get_best_move(tabuleiro)
print("Melhor jogada sugerida:", melhor_jogada)

# Testa com algumas posições ocupadas
tabuleiro2 = [1, -1, 1, 0, 0, 0, -1, 1, 0]
print("Tabuleiro:", tabuleiro2)
print("Melhor jogada sugerida:", mlp.get_best_move(tabuleiro2))