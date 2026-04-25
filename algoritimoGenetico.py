from rede_neural import RedeNeural
from settings import ELITE, PASSOS_EPISODIO, POPULACAO, TAXA_MUTACAO, FORCA_MUTACAO
import numpy as np
import pygame
from football_env import FootballEnv
from utils import *

class AlgoritmoGenetico:
    """
    Gerencia a população de agentes e a evolução entre gerações.
    """

    def __init__(self):
        # Cria a população inicial com redes aleatórias
        self.pesos = carregar_modelo("modelos/CR7_F140.npy", RedeNeural)
        self.populacao = [RedeNeural(self.pesos) for _ in range(POPULACAO)] #carrega o melhor modelo
        self.fitness    = [0.0] * POPULACAO
        self.geracao    = 0
        self.historico_fitness = []   # fitness máximo por geração (para o gráfico)
        self.historico_media   = []   # fitness médio por geração

    def avaliar_todos(self, callback_progresso=None):
        """
        Faz cada agente jogar e calcula o fitness de todos.
        callback_progresso(i) é chamado a cada agente avaliado.
        """
        env = FootballEnv(render_mode=False)   # sem janela = mais rápido

        for i, rede in enumerate(self.populacao):
            self.fitness[i] = self._avaliar_agente(rede, env)
            if callback_progresso:
                callback_progresso(i)

    def _avaliar_agente(self, rede, env):
        """
        Faz um agente jogar e retorna seu fitness.

        Fitness = quanto o agente foi bom:
          + gols marcados (peso alto)
          + se chegou perto da bola
          + se chegou perto do gol com a bola
          - distância da bola ao gol no fim
        """
        obs    = env.reset()
        fitness = 0.0

        for _ in range(PASSOS_EPISODIO):
            acao = rede.pensar(obs)
            obs, recompensa, done, info = env.step(acao)
            fitness += recompensa

            #despreza se a ação for nada 
            if acao == 0:
                fitness -= 0.5
                
            if done:
                break

        # Bônus extra por gols (para incentivar mais)
        fitness += info["score"] * 20.0
        return fitness

    def nova_geracao(self):
        """
        Cria a próxima geração:
          1. Ordena pelo fitness
          2. Os ELITE melhores passam direto
          3. O resto é gerado mutando os melhores
        """
        # Ordena: melhor fitness primeiro
        ordem = sorted(range(POPULACAO), key=lambda i: self.fitness[i], reverse=True)

        melhor_fitness = self.fitness[ordem[0]]
        media_fitness  = sum(self.fitness) / POPULACAO
        self.historico_fitness.append(melhor_fitness)
        self.historico_media.append(media_fitness)

         #Salvando a melhor rede para treinar mais tarde
        melhor_rede = self.populacao[ordem[0]]
        caminho = f"modelos/geracao_{self.geracao:04d}_fit_{melhor_fitness:.1f}.npy"
        salvar_modelo(melhor_rede, caminho)


        nova_pop = []

        # Elite: os melhores passam sem mudar
        for i in range(min(ELITE, len(ordem))):
            nova_pop.append(RedeNeural(self.populacao[ordem[i]].pesos))

        # Resto: filhos dos melhores com mutação
        while len(nova_pop) < POPULACAO:
            # Escolhe um dos top 10 para ser o pai
            top = min(10, len(ordem))
            pai_idx = ordem[np.random.randint(0, top)]
            filho   = self.populacao[pai_idx].mutar(TAXA_MUTACAO, FORCA_MUTACAO)
            nova_pop.append(filho)

        self.populacao = nova_pop
        self.fitness   = [0.0] * POPULACAO
        self.geracao  += 1

    def melhor_agente(self):
        """Retorna a rede do agente com maior fitness."""
        idx = int(np.argmax(self.fitness))
        return self.populacao[idx]
