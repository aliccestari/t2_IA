import numpy as np

class MLP:
    """
    Rede Neural MLP de 2 camadas para aprender jogo da velha
    Entrada: estado do tabuleiro (9 posições)
    Saída: probabilidades para cada posição (9 saídas)
    """
    
    def __init__(self, input_size=9, hidden_size=18, output_size=9, activation_function='sigmoid'):
        """
        Inicializa a rede neural
        Args:
            input_size: tamanho da entrada (9 para tabuleiro 3x3)
            hidden_size: neurônios na camada oculta
            output_size: tamanho da saída (9 posições possíveis)
            activation_function: função de ativação (default: sigmoid)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function
        # Inicialização dos pesos
        self.W1 = np.random.uniform(-1, 1, (self.hidden_size, self.input_size))
        self.b1 = np.zeros((self.hidden_size, 1))
        self.W2 = np.random.uniform(-1, 1, (self.output_size, self.hidden_size))
        self.b2 = np.zeros((self.output_size, 1))
        
    def sigmoid(self, x):
        """Função de ativação sigmoid"""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        """
        Propagação direta da entrada até a saída
        Args:
            x: vetor de entrada (shape: [9,] ou [9, 1])
        Returns:
            output: vetor de saída (shape: [9, 1])
        """
        x = np.array(x).reshape(-1, 1)
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        output = self.sigmoid(z2)
        return output
    
    def predict(self, board_state):
        """
        Recebe o estado do tabuleiro e retorna as probabilidades de jogada
        Args:
            board_state: vetor do tabuleiro (9,)
        Returns:
            output: vetor de probabilidades (9,)
        """
        output = self.forward(board_state)
        return output.flatten()
    
    def get_best_move(self, board_state):
        """
        Retorna a melhor jogada baseada na saída da rede
        Args:
            board_state: estado atual do tabuleiro
        Returns:
            move: posição escolhida (0-8)
        """
        output = self.predict(board_state)
        # Zera as probabilidades das posições já ocupadas
        available = np.array(board_state) == 0
        masked_output = np.where(available, output, -np.inf)
        return int(np.argmax(masked_output))
    
    def get_total_weights_count(self):
        """
        Retorna o número total de pesos na rede
        Returns:
            count: número total de pesos e bias
        """
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size
        )

    def set_weights_from_chromosome(self, chromosome):
        """
        Define os pesos da rede a partir de um vetor cromossomo
        Args:
            chromosome: vetor 1D com todos os pesos e bias
        """
        idx = 0
        w1_size = self.W1.size
        b1_size = self.b1.size
        w2_size = self.W2.size
        b2_size = self.b2.size
        self.W1 = np.array(chromosome[idx:idx+w1_size]).reshape(self.W1.shape)
        idx += w1_size
        self.b1 = np.array(chromosome[idx:idx+b1_size]).reshape(self.b1.shape)
        idx += b1_size
        self.W2 = np.array(chromosome[idx:idx+w2_size]).reshape(self.W2.shape)
        idx += w2_size
        self.b2 = np.array(chromosome[idx:idx+b2_size]).reshape(self.b2.shape) 