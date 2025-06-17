import time
import numpy as np
import os
from ..game.tic_tac_toe import TicTacToe
from ..minimax.minimax import MinimaxPlayer
from ..neural_network.mlp import MLP
from ..training.trainer import Trainer

class ConsoleInterface:
    """
    Interface de console para o jogo da velha com 3 modos:
    1. Usuário vs Minimax
    2. Treinamento: Rede Neural vs Minimax
    3. Usuário vs Rede Neural treinada
    """
    
    def __init__(self):
        """Inicializa interface de console"""
        self.trainer = None
        self.neural_network = None
        self.minimax_player = None
        self.game = None
        self.trained_chromosome = None

    def show_menu(self):
        """Mostra menu principal"""
        while True:
            print("\n=== Jogo da Velha IA - Menu Principal ===")
            print("[1] Jogar contra o Minimax")
            print("[2] Treinar a rede (Rede vs Minimax)")
            print("[3] Jogar contra a rede treinada")
            print("[0] Sair")
            choice = input("Escolha uma opção: ")
            if choice == '1':
                self.run_user_vs_minimax()
            elif choice == '2':
                self.run_training()
            elif choice == '3':
                self.run_user_vs_neural_network()
            elif choice == '0':
                print("Saindo...")
                break
            else:
                print("Opção inválida. Tente novamente.")

    def run_user_vs_minimax(self):
        """Executa modo usuário vs minimax"""
        print("\n=== Modo Usuário vs Minimax ===")
        # Escolha de dificuldade
        difficulties = {'1': 'easy', '2': 'medium', '3': 'hard'}
        print("Escolha a dificuldade do Minimax:")
        print("[1] Fácil (25% Minimax)")
        print("[2] Médio (50% Minimax)")
        print("[3] Difícil (100% Minimax)")
        diff_choice = input("Opção: ")
        difficulty = difficulties.get(diff_choice, 'easy')
        print(f"Dificuldade selecionada: {difficulty}")
        # Inicializa jogo e minimax
        self.game = TicTacToe()
        self.minimax_player = MinimaxPlayer(difficulty=difficulty, symbol='O')
        current_player = 1  # Usuário sempre começa como X (1)
        while not self.game.is_game_over():
            self.print_board(self.game.board)
            if current_player == 1:
                # Jogada do usuário
                move = self.get_user_move()
                valid = self.game.make_move(move, 1)
                if not valid:
                    print("Jogada inválida! Tente novamente.")
                    continue
            else:
                # Jogada do minimax
                move = self.minimax_player.get_move(self.game.get_board_copy())
                print(f"Minimax joga na posição {move}")
                self.game.make_move(move, -1)
            current_player *= -1
        # Fim do jogo
        self.print_board(self.game.board)
        if self.game.winner == 1:
            print("Parabéns! Você venceu!")
        elif self.game.winner == -1:
            print("Minimax venceu!")
        else:
            print("Empate!")

    def print_board(self, board):
        """Imprime tabuleiro no console"""
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("\nTabuleiro:")
        for i in range(3):
            row = [symbols[board[3*i + j]] for j in range(3)]
            print(' | '.join(row))
            if i < 2:
                print('--+---+--')

    def get_user_move(self):
        """Obtém jogada do usuário via input"""
        while True:
            try:
                move = int(input("Sua jogada (0-8): "))
                if 0 <= move <= 8:
                    return move
                else:
                    print("Digite um número entre 0 e 8.")
            except ValueError:
                print("Entrada inválida. Digite um número entre 0 e 8.")

    def run_training(self):
        """Executa treinamento via console"""
        print("\n=== Modo Treinamento: Rede Neural vs Minimax ===")
        # Configuração ajustada para melhor aprendizado
        config = {
            "population_size": 150,
            "max_generations": 100,
            "neural_network": {"hidden_size": 18},
            "genetic_algorithm": {
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "elitism_rate": 0.1,
            },
            "training_schedule": ["easy", "medium", "hard"],
            "generations_per_difficulty": [5, 10, 50],
        }
        self.trainer = Trainer(config)
        self.trainer.initialize_components()
        schedule = config['training_schedule']
        generations_per_difficulty = config['generations_per_difficulty']
        for idx, difficulty in enumerate(schedule):
            gens = generations_per_difficulty[idx] if idx < len(generations_per_difficulty) else 1
            print(f"Treinando com dificuldade: {difficulty} por {gens} gerações...")
            self.trainer.minimax_player.set_difficulty(difficulty)
            for gen in range(gens):
                # Aumentar o número de jogos por cromossomo para 10
                self.trainer.genetic_algorithm.evaluate_fitness(
                    self.trainer.neural_network,
                    self.trainer.minimax_player,
                    self.trainer.game,
                    games_per_chromosome=10
                )
                self.trainer.genetic_algorithm.evolve_generation(
                    self.trainer.neural_network,
                    self.trainer.minimax_player,
                    self.trainer.game
                )
                stats = self.trainer.genetic_algorithm.get_statistics()
                print(f"Geração {stats['generation']}: Melhor fitness = {stats['best_fitness']:.2f}, Média = {stats['mean_fitness']:.2f}")
        print("Treinamento concluído!")
        # Salvar melhor cromossomo
        best_chromosome = self.trainer.genetic_algorithm.get_best_chromosome()
        self.trainer.neural_network.set_weights_from_chromosome(best_chromosome)
        ts = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/best_chromosome_{ts}.npy"
        np.save(path, best_chromosome)
        print(f"Pesos do melhor cromossomo salvos em {path}")
        self.trained_chromosome = best_chromosome
        print("Você pode agora jogar contra a rede treinada pelo menu principal!")

    def run_user_vs_neural_network(self):
        """Executa modo usuário vs rede neural"""
        print("\n=== Modo Usuário vs Rede Neural Treinada ===")
        # Sugerir o arquivo mais recente de pesos salvo
        checkpoints_dir = "checkpoints"
        if not os.path.exists(checkpoints_dir):
            print("Nenhum treinamento encontrado. Treine a rede primeiro!")
            return
        files = [f for f in os.listdir(checkpoints_dir) if f.startswith("best_chromosome") and f.endswith(".npy")]
        if not files:
            print("Nenhum arquivo de pesos encontrado. Treine a rede primeiro!")
            return
        files.sort(reverse=True)
        default_path = os.path.join(checkpoints_dir, files[0])
        print(f"Arquivo de pesos sugerido: {default_path}")
        path = input(f"Digite o caminho do arquivo de pesos para carregar [ENTER para usar sugerido]: ").strip()
        if not path:
            path = default_path
        try:
            chromosome = np.load(path)
        except Exception as e:
            print(f"Erro ao carregar pesos: {e}")
            return
        # Detectar hidden_size a partir do tamanho do cromossomo
        input_size = 9
        output_size = 9
        total_len = len(chromosome)
        hidden_size = (total_len - output_size) // (input_size + output_size + 1)
        if hidden_size <= 0 or (input_size*hidden_size + hidden_size + hidden_size*output_size + output_size) != total_len:
            print("Erro: O arquivo de pesos não é compatível com a arquitetura esperada.")
            return
        nn = MLP(hidden_size=hidden_size)
        nn.set_weights_from_chromosome(chromosome)
        self.neural_network = nn
        self.game = TicTacToe()
        print(f"Rede neural carregada! (hidden_size={hidden_size}) Você é o X (1), a rede é o O (-1). Comece jogando!")
        current_player = 1  # Usuário começa
        while not self.game.is_game_over():
            self.print_board(self.game.board)
            if current_player == 1:
                move = self.get_user_move()
                valid = self.game.make_move(move, 1)
                if not valid:
                    print("Jogada inválida! Tente novamente.")
                    continue
            else:
                move = self.neural_network.get_best_move(self.game.get_board_state_for_nn())
                print(f"Rede Neural joga na posição {move}")
                self.game.make_move(move, -1)
            current_player *= -1
        self.print_board(self.game.board)
        if self.game.winner == 1:
            print("Parabéns! Você venceu a rede neural!")
        elif self.game.winner == -1:
            print("A rede neural venceu!")
        else:
            print("Empate!")

    def run(self):
        """Inicia interface de console"""
        self.show_menu() 