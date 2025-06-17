#!/usr/bin/env python3
"""
Trabalho T2 - Inteligência Artificial
PUCRS - Profa. Silvia Moraes

Aprendizagem por Reforço: RN + AG + Minimax
Jogo da Velha com Rede Neural treinada por Algoritmo Genético

Autores: Alice Colares, Mykelly Barros, Giovana Raupp, Samara Tavares
Data: Junho 2024
"""

import sys
import argparse
from src.frontend.gui import ConsoleInterface

def main():
    """Função principal da aplicação"""
    parser = argparse.ArgumentParser(description='Jogo da Velha com IA')
    parser.add_argument('--mode', choices=['console'], default='console',
                        help='Modo de interface (console)')
    
    args = parser.parse_args()
    
    try:
        # Inicia interface de console
        app = ConsoleInterface()
        app.run()
            
    except KeyboardInterrupt:
        print("\nAplicação encerrada pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"Erro na aplicação: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 