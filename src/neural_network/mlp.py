import numpy as np

class MLP:
    """
    Rede Neural MLP de 2 camadas para aprender jogo da velha
    Entrada: estado do tabuleiro (9 posições)
    Saída: probabilidades para cada posição (9 saídas)
    """
    
    def __init__(self, input_size=9, hidden_size=18, output_size=9, activation_function='sigmoid'):
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
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        x = np.array(x).reshape(-1, 1)
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        output = self.sigmoid(z2)
        return output
    
    def predict(self, board_state):
        output = self.forward(board_state)
        return output.flatten()
    
    def get_best_move(self, board_state):
        output = self.predict(board_state)
        # Zera as probabilidades das posições já ocupadas
        available = np.array(board_state) == 0
        masked_output = np.where(available, output, -np.inf)
        return int(np.argmax(masked_output))
    
    def get_total_weights_count(self):
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size
        )

    def set_weights_from_chromosome(self, chromosome):
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