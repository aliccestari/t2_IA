import numpy as np
import random

class GeneticAlgorithm:
    """
    Algoritmo Genético para evoluir os pesos da rede neural
    Cromossomos = pesos da rede neural
    """
    
    def __init__(self, population_size=200, chromosome_length=None, 
                 mutation_rate=0.1, crossover_rate=0.8, elitism_rate=0.1):
        """
        Inicializa o Algoritmo Genético
        Args:
            population_size: tamanho da população
            chromosome_length: tamanho do cromossomo (total de pesos da rede)
            mutation_rate: taxa de mutação
            crossover_rate: taxa de cruzamento
            elitism_rate: taxa de elitismo
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_fitness_history = []
        self.mean_fitness_history = []
        
    def initialize_population(self):
        """
        Cria a população inicial com pesos aleatórios
        """
        self.population = [np.random.uniform(-1, 1, self.chromosome_length) for _ in range(self.population_size)]
        self.fitness_scores = [0.0 for _ in range(self.population_size)]
        print(f"População inicializada com {self.population_size} cromossomos.")

    def evaluate_fitness(self, neural_network, minimax_player, game, games_per_chromosome=5):
        """
        Avalia a aptidão de toda a população
        Args:
            neural_network: instância da rede neural (MLP)
            minimax_player: jogador minimax (ou adversário aleatório para teste)
            game: instância do jogo da velha
            games_per_chromosome: número de partidas por cromossomo
        """
        self.fitness_scores = []
        for chromosome in self.population:
            fitness = self.fitness_function(chromosome, neural_network, minimax_player, game, games_per_chromosome)
            self.fitness_scores.append(fitness)

    def fitness_function(self, chromosome, neural_network, minimax_player, game, games_per_chromosome=5):
        """
        Função de aptidão para um cromossomo individual
        Mede desempenho da rede contra o minimax (ou adversário aleatório)
        Args:
            chromosome: cromossomo a ser avaliado
            neural_network: rede neural
            minimax_player: minimax opponent (ou aleatório)
            game: jogo da velha
            games_per_chromosome: número de partidas
        Returns:
            fitness: valor de aptidão
        """
        # Parâmetros de pontuação (pode importar do config.py se preferir)
        WIN_REWARD = 15
        DRAW_REWARD = 3
        LOSS_PENALTY = -2
        INVALID_MOVE_PENALTY = -1
        GAME_LENGTH_BONUS = 0.2

        total_fitness = 0
        for _ in range(games_per_chromosome):
            game.reset_game()
            neural_network.set_weights_from_chromosome(chromosome)
            current_player = 1  # Rede neural sempre começa
            moves = 0
            while not game.is_game_over():
                if current_player == 1:
                    # Rede neural joga
                    move = neural_network.get_best_move(game.get_board_state_for_nn())
                    valid = game.make_move(move, 1)
                    if not valid:
                        total_fitness += INVALID_MOVE_PENALTY
                        break  # Fim do jogo por jogada inválida
                else:
                    # Minimax ou adversário aleatório joga
                    available = game.get_available_moves()
                    if not available:
                        break
                    # Se minimax_player for None, faz jogada aleatória
                    if minimax_player is not None:
                        move = minimax_player.get_move(game.get_board_copy())
                    else:
                        move = random.choice(available)
                    game.make_move(move, -1)
                moves += 1
                current_player *= -1
            # Avalia resultado
            if game.winner == 1:
                total_fitness += WIN_REWARD
            elif game.winner == -1:
                total_fitness += LOSS_PENALTY
            elif game.winner == 0:
                total_fitness += DRAW_REWARD
            total_fitness += moves * GAME_LENGTH_BONUS
        return total_fitness / games_per_chromosome
    
    def selection_tournament(self, tournament_size=3):
        """
        Seleção por torneio
        """
        selected = []
        for _ in range(self.population_size):
            participants = random.sample(list(enumerate(self.fitness_scores)), tournament_size)
            winner_idx = max(participants, key=lambda x: x[1])[0]
            selected.append(self.population[winner_idx].copy())
        return selected

    def selection_elitism(self):
        """
        Seleção por elitismo - mantém os melhores
        """
        elite_count = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:][::-1]
        elite = [self.population[i].copy() for i in elite_indices]
        return elite
    
    def crossover_real_valued(self, parent1, parent2, alpha=0.5):
        """
        Cruzamento BLX-α para valores reais
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        for i in range(len(parent1)):
            c_min = min(parent1[i], parent2[i])
            c_max = max(parent1[i], parent2[i])
            I = c_max - c_min
            low = c_min - alpha * I
            high = c_max + alpha * I
            offspring1[i] = np.random.uniform(low, high)
            offspring2[i] = np.random.uniform(low, high)
        return offspring1, offspring2
    
    def mutation_gaussian(self, chromosome, mutation_std=0.1):
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if np.random.rand() < 0.2:  # Aumento da taxa de mutação
                mutated[i] += np.random.normal(0, mutation_std)
        return mutated

    
    def evolve_generation(self, neural_network, minimax_player, game):
        """
        Executa uma geração completa do AG
        """
        # Avalia fitness de toda a população
        self.evaluate_fitness(neural_network, minimax_player, game)
        new_population = []
        # Elitismo
        elite = self.selection_elitism()
        new_population.extend(elite)
        # Seleção por torneio para preencher o resto
        selected = self.selection_tournament()
        # Cruzamento e mutação
        while len(new_population) < self.population_size:
            if np.random.rand() < self.crossover_rate:
                p1, p2 = random.sample(selected, 2)
                o1, o2 = self.crossover_real_valued(p1, p2)
            else:
                o1, o2 = random.sample(selected, 2)
            o1 = self.mutation_gaussian(o1)
            o2 = self.mutation_gaussian(o2)
            new_population.append(o1)
            if len(new_population) < self.population_size:
                new_population.append(o2)
        self.population = new_population[:self.population_size]
        self.generation += 1
        # Atualiza histórico do melhor fitness
        best_fitness = max(self.fitness_scores)
        mean_fitness = float(np.mean(self.fitness_scores))
        self.best_fitness_history.append(best_fitness)
        self.mean_fitness_history.append(mean_fitness)

    def get_best_chromosome(self):
        """
        Retorna o melhor cromossomo da população atual
        """
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx].copy()

    def get_statistics(self):
        """
        Retorna estatísticas da população atual
        """
        stats = {
            'generation': self.generation,
            'best_fitness': float(np.max(self.fitness_scores)),
            'mean_fitness': float(np.mean(self.fitness_scores)),
            'std_fitness': float(np.std(self.fitness_scores)),
        }
        return stats
    
    def should_stop(self, max_generations=100, convergence_threshold=0.001):
        """
        Critério de parada do algoritmo
        """
        if self.generation >= max_generations:
            return True
        if len(self.best_fitness_history) >= 10:
            recent = self.best_fitness_history[-10:]
            if max(recent) - min(recent) < convergence_threshold:
                return True
        return False

    def relu(self, x):
        return np.maximum(0, x)

    def plot_training_progress(self):
        import matplotlib.pyplot as plt
        generations = list(range(1, len(self.best_fitness_history) + 1))
        plt.figure(figsize=(10, 5))
        plt.plot(generations, self.best_fitness_history, label="Melhor Fitness")
        plt.plot(generations, self.mean_fitness_history, label="Fitness Médio")
        plt.title("Progresso do Treinamento")
        plt.xlabel("Geração")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show() 