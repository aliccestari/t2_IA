from src.training.trainer import Trainer
import numpy as np
import json, time, os

if __name__ == "__main__":
    config = {
        "population_size": 100,
        "max_generations": 5,
        "neural_network": {"hidden_size": 9},
        "genetic_algorithm": {
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elitism_rate": 0.1,
        },
        # ordem de dificuldades e nº de gerações em cada uma
        "training_schedule": ["easy", "medium", "hard"],
        "generations_per_difficulty": [10, 15, 100],
    }

    trainer = Trainer(config)
    trainer.train()

    # salva o melhor cromossomo obtido
    best_chromosome = trainer.genetic_algorithm.get_best_chromosome()
    trainer.neural_network.set_weights_from_chromosome(best_chromosome)

    ts = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("checkpoints", exist_ok=True)
    np.save(f"checkpoints/best_chromosome_{ts}.npy", best_chromosome)
    with open(f"checkpoints/config_{ts}.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Pesos do melhor cromossomo salvos em checkpoints/best_chromosome_*.npy")

    # avalia a rede treinada
    trainer.test_final_performance(num_test_games=10)
