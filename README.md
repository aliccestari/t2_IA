# T2 - Aprendizagem por Reforço: Jogo da Velha com RN + AG + Minimax

**Disciplina:** Inteligência Artificial - PUCRS  
**Professora:** Silvia Moraes

## Descrição

Este projeto implementa uma IA capaz de aprender a jogar o jogo da velha usando uma Rede Neural (MLP) treinada por Algoritmo Genético, tendo o Minimax como "professor". O sistema permite jogar contra o Minimax, treinar a rede e jogar contra a rede treinada, tudo via interface de console.

## Autores

- Alice Colares
- Mykelly Barros
- Giovana Raupp
- Samara Tavares

## Requisitos

- Python 3.8+
- numpy
- matplotlib

Instale as dependências com:
```bash
pip install -r requirements.txt
```

## Como Executar

Para iniciar o sistema, execute:
```bash
python3 main.py
```

## Modos de Jogo

Ao rodar o programa, escolha no menu:

1. **Jogar contra o Minimax**  
   - Escolha a dificuldade: fácil (25% Minimax), médio (50%), difícil (100%)
2. **Treinar a Rede Neural**  
   - A rede aprende jogando contra o Minimax, evoluindo via Algoritmo Genético
   - O progresso do treinamento é exibido no console
3. **Jogar contra a Rede Neural Treinada**  
   - Após o treinamento, jogue contra a IA e veja sua performance

## Estrutura Técnica (resumido)

- **Rede Neural:** MLP de 2 camadas (entrada: 9, oculta: 18, saída: 9)
- **Algoritmo Genético:** Evolve os pesos da rede (BLX-α, mutação gaussiana, elitismo)
- **Minimax:** 3 níveis de dificuldade, serve como adversário/professor

## Como Treinar e Testar a Rede

1. Escolha a opção "Treinar a rede" no menu.
2. O sistema irá evoluir a rede por várias gerações.
3. Após o treino, escolha "Jogar contra a rede treinada" para testar a IA.

Os pesos da melhor rede são salvos automaticamente na pasta `checkpoints/`.

## Observações

- O projeto não possui interface gráfica, apenas console.
- Para dúvidas, consulte o relatório entregue junto ao projeto.

---
