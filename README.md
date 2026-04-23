# Football AI — Ambiente de Futebol 2D com Algoritmo Genetico

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.x-00B140?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)

Um ambiente de futebol 2D top-down desenvolvido em Pygame com suporte a controle humano e treinamento de agentes por algoritmo genetico com rede neural simples.

---

## Visao Geral

O projeto é dividido em dois arquivos principais. O `football_env.py` implementa o campo de futebol, a fisica da bola, o jogador e o sistema de recompensas. O `main.py` implementa a rede neural, o algoritmo genetico e a visualizacao do treinamento em tempo real.

O ambiente foi projetado para ser simples de entender e facil de estender, especialmente para quem esta comecando na area de inteligencia artificial assim como eu.

---

## Como funciona

O treinamento segue o ciclo classico de neuroevolucao:

1. Uma populacao de agentes e criada com pesos aleatorios
2. Cada agente joga um episodio e recebe uma nota (fitness)
3. Os melhores agentes sao selecionados
4. Uma nova geracao e criada a partir dos melhores, com pequenas mutações nos pesos
5. O ciclo se repete até que os agentes aprenderem a fazer gol

A rede neural de cada agente possui tres camadas: entrada com 8 neuronios, uma camada oculta com 12 neuronios e saida com 6 neuronios . Os pesos sãoo otimizados pelo algoritmo genético, sem backpropagation.

---

## Requisitos

- Python 3.8 ou superior
- Pygame
- NumPy

Instalacão das dependencias:

```bash
pip install pygame numpy
```

---

## Como executar

Para jogar manualmente:

```bash
python football_env.py
```

Para iniciar o treinamento dos agentes:

```bash
python main.py
```

Caso queira configurar coisas especificas dos agentes como: número de agenets na população, velocidade de treinamento, cores etc. Basta mudar no arquivo settings.py

---

## Controles (modo humano)

| Tecla | Acao |
|---|---|
| W / Seta cima | Mover para cima |
| S / Seta baixo | Mover para baixo |
| A / Seta esquerda | Mover para esquerda |
| D / Seta direita | Mover para direita |
| Espaco | Chutar a bola |
| R | Reiniciar |
| ESC | Sair |

---

## Parametros de treinamento

Os parametros ficam no topo do `train.py` e podem ser ajustados livremente:

obs: esses são os parametros padrões do experimento.

| Parametro | Valor padrao | Descricao |
|---|---|---|
| `POPULACAO` | 30 | Numero de agentes por geracao |
| `PASSOS_EPISODIO` | 600 | Frames que cada agente joga por episodio |
| `TAXA_MUTACAO` | 0.15 | Percentual de pesos que sofrem mutacao |
| `FORCA_MUTACAO` | 0.4 | Intensidade da mudanca em cada mutacao |
| `ELITE` | 5 | Agentes que passam direto para a proxima geracao |
| `NEURONIOS_OCULTO` | 12 | Neuronios na camada oculta da rede |
| `VELOCIDADE_SIM` | 3 | Multiplicador de velocidade da simulacao |

---

## Interface de treinamento

Durante o treinamento a janela e dividida em dois paineis. O lado esquerdo exibe o campo com o melhor agente jogando em tempo real. O lado direito exibe a rede neural do melhor agente com as ativacoes de cada neuronio atualizadas a cada frame, o grafico de fitness ao longo das geracoes (linha azul para o melhor, linha vermelha para a media) e as informacoes da geracao atual.

---

## Estrutura do projeto

```
.
|__ football_env.py   
|__ settings.py
|__ algoritimoGenetico.py
|__ viwer.py
|__ rede_neural.py
├── main.py          
└── README.md
```

---

## Interface para agentes de IA

O ambiente segue uma interface padrão para facilitar a integração com qualquer algoritmo:

```python
env = FootballEnv(render_mode=False)
obs = env.reset()

obs, reward, done, info = env.step(action)
```

O vetor de observacao retornado pelo `reset()` e `step()` contem 8 valores normalizados entre 0 e 1: posicao do jogador (x, y), posicao da bola (x, y), velocidade da bola (x, y) e posicao do gol (x, y).

As ações possiveis sao inteiros de 0 a 5, representando nada, cima, baixo, esquerda, direita e chutar respectivamente.

---

## Proximos passos sugeridos

- Adicionar um segundo agente para treinamento competitivo
- Adicionar um agente de goleiro
- Salvar e carregar os pesos dos melhores agentes