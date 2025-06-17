import numpy as np
import matplotlib.pyplot as plt
from ..neural_network.mlp import MLP
from ..genetic_algorithm.genetic_algo import GeneticAlgorithm
from ..minimax.minimax import MinimaxPlayer
from ..game.tic_tac_toe import TicTacToe

class Trainer:
    """
    Coordena o treinamento da rede neural usando algoritmo genético
    A rede aprende jogando contra o minimax em diferentes dificuldades
    """
    
    def __init__(self, config=None):
        # Configurações padrão
        self.config = config or {
            'population_size': 100 ,
            'max_generations': 100,
            'neural_network': {
                'hidden_size': 18,
            },
            'genetic_algorithm': {
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elitism_rate': 0.1
            },
            'training_schedule': ['easy', 'medium', 'hard'],  # Ordem das dificuldades
            'generations_per_difficulty': [10, 15, 100]
        }
        
        # Inicializar componentes
        self.neural_network = None
        self.genetic_algorithm = None
        self.minimax_player = None
        self.game = None
        
        # Estatísticas de treinamento
        self.training_history = []
        self.current_difficulty = 0
        
    def initialize_components(self):
        """Inicializa todos os componentes necessários"""
        hidden_size = self.config['neural_network']['hidden_size']
        self.neural_network = MLP(hidden_size=hidden_size)
        chromosome_length = self.neural_network.get_total_weights_count()
        ga_cfg = self.config['genetic_algorithm']
        self.genetic_algorithm = GeneticAlgorithm(
            population_size=self.config['population_size'],
            chromosome_length=chromosome_length,
            mutation_rate=ga_cfg['mutation_rate'],
            crossover_rate=ga_cfg['crossover_rate'],
            elitism_rate=ga_cfg['elitism_rate']
        )
        self.genetic_algorithm.initialize_population()
        self.game = TicTacToe()
        self.minimax_player = MinimaxPlayer(difficulty='easy', symbol='O')
    
    def train(self):
        self.initialize_components()
        schedule = self.config['training_schedule']
        generations_per_difficulty = self.config.get('generations_per_difficulty', [30, 30, 40])
        for idx, difficulty in enumerate(schedule):
            gens = generations_per_difficulty[idx] if idx < len(generations_per_difficulty) else 30
            print(f"Treinando com dificuldade: {difficulty} por {gens} gerações...")
            self.train_with_difficulty(difficulty, gens)
        print("Treinamento concluído!")
        self.plot_training_progress()
    
    def train_with_difficulty(self, difficulty, generations):
        self.minimax_player.set_difficulty(difficulty)
        for gen in range(generations):
            self.genetic_algorithm.evolve_generation(
                self.neural_network,
                self.minimax_player,
                self.game
            )
            stats = self.genetic_algorithm.get_statistics()
            self.training_history.append(stats)
            self.save_training_progress(self.genetic_algorithm.generation, stats)
            print(f"Geração {stats['generation']}: Melhor fitness = {stats['best_fitness']:.2f}, Média = {stats['mean_fitness']:.2f}")
    
    def play_training_match(self, chromosome):
        self.game.reset_game()
        self.neural_network.set_weights_from_chromosome(chromosome)
        current_player = 1
        while not self.game.is_game_over():
            if current_player == 1:
                move = self.neural_network.get_best_move(self.game.get_board_state_for_nn())
                self.game.make_move(move, 1)
            else:
                move = self.minimax_player.get_move(self.game.get_board_copy())
                self.game.make_move(move, -1)
            current_player *= -1
        return self.game.winner

    def evaluate_neural_network(self, chromosome, num_games=10):
        results = []
        for _ in range(num_games):
            result = self.play_training_match(chromosome)
            results.append(result)
        return self.calculate_fitness_score(results)

    def calculate_fitness_score(self, game_results):
        WIN_REWARD = 15
        DRAW_REWARD = 3
        LOSS_PENALTY = -2
        fitness = -2
        for result in game_results:
            if result == 1:
                fitness += WIN_REWARD
            elif result == 0:
                fitness += DRAW_REWARD
            elif result == -1:
                fitness += LOSS_PENALTY
        return fitness / len(game_results) if game_results else 0

    def save_training_progress(self, generation, stats):
        self.training_history.append(stats)  # Adiciona as estatísticas à lista
        print(f"[Progresso] Geração {generation}: {stats}")

    def plot_training_progress(self):
        if not self.training_history:
            print("Nenhum dado de treinamento para plotar.")
            return
        generations = [h['generation'] for h in self.training_history]
        best = [h['best_fitness'] for h in self.training_history]
        mean = [h['mean_fitness'] for h in self.training_history]
        plt.figure(figsize=(10, 5))
        plt.plot(generations, best, label='Melhor Fitness')
        plt.plot(generations, mean, label='Fitness Médio')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.title('Progresso do Treinamento')
        plt.legend()
        plt.show()

    def get_best_neural_network(self):
        if self.genetic_algorithm is None or not hasattr(self.genetic_algorithm, 'population') or not self.genetic_algorithm.population:
            raise RuntimeError("A população genética não foi inicializada. Rode o treinamento antes de testar a performance final.")
        best_chromosome = self.genetic_algorithm.get_best_chromosome()
        self.neural_network.set_weights_from_chromosome(best_chromosome)
        return self.neural_network
    
    def test_final_performance(self, num_test_games=100):
        # Garante que os componentes estejam inicializados
        if self.game is None or self.minimax_player is None or self.neural_network is None or self.genetic_algorithm is None:
            self.initialize_components()
        best_nn = self.get_best_neural_network()
        results = []
        for _ in range(num_test_games):
            self.game.reset_game()
            current_player = 1
            while not self.game.is_game_over():
                if current_player == 1:
                    move = best_nn.get_best_move(self.game.get_board_state_for_nn())
                    self.game.make_move(move, 1)
                else:
                    move = self.minimax_player.get_move(self.game.get_board_copy())
                    self.game.make_move(move, -1)
                current_player *= -1
            results.append(self.game.winner)
        print(f"Resultados dos {num_test_games} jogos de teste: {results}")
        return results 