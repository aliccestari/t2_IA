import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.genetic_algorithm.genetic_algo import GeneticAlgorithm
from src.neural_network.mlp import MLP
from src.game.tic_tac_toe import TicTacToe

# Parâmetros
population_size = 10
mlp = MLP()
chromosome_length = mlp.get_total_weights_count()

# Inicializa o AG
ga = GeneticAlgorithm(
    population_size=population_size,
    chromosome_length=chromosome_length,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elitism_rate=0.1
)
ga.initialize_population()

# Instâncias do jogo e da rede neural
game = TicTacToe()
neural_network = MLP(input_size=9, hidden_size=18, output_size=9)

# Teste: evoluir por 3 gerações contra adversário aleatório
for _ in range(3):
    ga.evolve_generation(neural_network, minimax_player=None, game=game)
    stats = ga.get_statistics()
    print(f'Geração {stats["generation"]}: Melhor fitness = {stats["best_fitness"]:.2f}, Média = {stats["mean_fitness"]:.2f}')

# Mostra o melhor cromossomo encontrado
best = ga.get_best_chromosome()
print('Melhor cromossomo (primeiros 5 genes):', best[:5])