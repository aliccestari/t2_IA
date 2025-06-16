import random

class MinimaxPlayer:
    """
    Algoritmo Minimax adaptado com 3 modos de dificuldade
    - Fácil: usa minimax em 25% das jogadas
    - Médio: usa minimax em 50% das jogadas  
    - Difícil: usa minimax em 100% das jogadas
    """
    
    def __init__(self, difficulty='hard', symbol='O'):
        """
        Inicializa o jogador Minimax
        Args:
            difficulty: 'easy', 'medium', 'hard'
            symbol: símbolo do jogador ('X' ou 'O')
        """
        self.difficulty = difficulty
        self.symbol = symbol
        self.opponent_symbol = 'X' if symbol == 'O' else 'O'
        
        # Probabilidades de usar minimax por dificuldade
        self.minimax_probability = {
            'easy': 0.25,
            'medium': 0.50,
            'hard': 1.0
        }
    
    def get_move(self, board_state):
        """
        Retorna a próxima jogada baseada na dificuldade.

        Dependendo da dificuldade configurada, o jogador utiliza o algoritmo
        minimax em uma determinada porcentagem das vezes. Caso o minimax não
        seja utilizado, uma jogada aleatória válida é retornada.

        Args:
            board_state: estado atual do tabuleiro como lista/array de tamanho 9

        Returns:
            move: posição escolhida (0-8) ou ``None`` se não houver jogadas
                disponíveis
        """

        available_moves = self.get_available_moves(board_state)
        if not available_moves:
            return None

        use_minimax = random.random() < self.minimax_probability.get(
            self.difficulty, 1.0
        )

        if not use_minimax:
            return self.get_random_move(board_state)

        player_val = 1 if self.symbol == 'X' else -1
        best_score = -float("inf")
        best_move = None

        for move in available_moves:
            board_copy = list(board_state)
            board_copy[move] = player_val
            score = self.minimax(board_copy, 0, False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
    
    def minimax(self, board, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
        """
        Algoritmo Minimax com poda alfa-beta
        Args:
            board: estado do tabuleiro
            depth: profundidade atual
            is_maximizing: se é turno do maximizador
            alpha: valor alfa para poda
            beta: valor beta para poda
        Returns:
            score: valor da posição
        """
        player_val = 1 if self.symbol == 'X' else -1
        result = self.evaluate_board(board)

        if result is not None:
            # Pontuação já está no ponto de vista do jogador
            return result

        if is_maximizing:
            best_score = -float('inf')
            for move in self.get_available_moves(board):
                board[move] = player_val
                score = self.minimax(board, depth + 1, False, alpha, beta)
                board[move] = 0
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            opponent_val = -player_val
            best_score = float('inf')
            for move in self.get_available_moves(board):
                board[move] = opponent_val
                score = self.minimax(board, depth + 1, True, alpha, beta)
                board[move] = 0
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score
    
    def evaluate_board(self, board):
        """
        Avalia o estado atual do tabuleiro
        Args:
            board: estado do tabuleiro
        Returns:
            score: pontuação da posição
        """
        win_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # linhas
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colunas
            (0, 4, 8), (2, 4, 6)              # diagonais
        ]

        player_val = 1 if self.symbol == 'X' else -1

        for a, b, c in win_combinations:
            if board[a] == board[b] == board[c] != 0:
                return 1 if board[a] == player_val else -1

        if 0 not in board:
            return 0

        return None
    
    def get_random_move(self, board_state):
        """
        Retorna uma jogada aleatória válida
        Args:
            board_state: estado do tabuleiro
        Returns:
            move: posição aleatória válida
        """
        moves = self.get_available_moves(board_state)
        return random.choice(moves) if moves else None
    
    def get_available_moves(self, board_state):
        """
        Retorna lista de movimentos válidos
        Args:
            board_state: estado do tabuleiro
        Returns:
            moves: lista de posições disponíveis
        """
        return [i for i, v in enumerate(board_state) if v == 0]
    
    def set_difficulty(self, difficulty):
        """
        Altera a dificuldade do minimax
        Args:
            difficulty: 'easy', 'medium', 'hard'
        """
        if difficulty not in self.minimax_probability:
            raise ValueError("Invalid difficulty level")
        self.difficulty = difficulty