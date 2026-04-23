from settings import N_INPUTS, NEURONIOS_OCULTO, N_ACTIONS
import numpy as np

class RedeNeural:
    """
    Rede neural com 3 camadas:
      Entrada (8) → Oculta (12) → Saída (6 ações)

    Os "pesos" são só arrays de números que dizem o quanto
    cada neurônio influencia o próximo.
    """

    def __init__(self, pesos=None):
        # Tamanho de cada camada de pesos
        tam_w1 = N_INPUTS * NEURONIOS_OCULTO          # 8×12 = 96
        tam_b1 = NEURONIOS_OCULTO                     # 12
        tam_w2 = NEURONIOS_OCULTO * N_ACTIONS         # 12×6 = 72
        tam_b2 = N_ACTIONS                            # 6
        self.tam_total = tam_w1 + tam_b1 + tam_w2 + tam_b2  # 186 pesos no total

        if pesos is None:
            # Pesos aleatórios entre -1 e 1
            self.pesos = np.random.uniform(-1, 1, self.tam_total)
        else:
            self.pesos = pesos.copy()

        self._separar_pesos()

    def _separar_pesos(self):
        """Divide o array plano de pesos nas matrizes da rede."""
        p = self.pesos
        i = 0

        t = N_INPUTS * NEURONIOS_OCULTO
        self.W1 = p[i:i+t].reshape(N_INPUTS, NEURONIOS_OCULTO);  i += t
        self.b1 = p[i:i+NEURONIOS_OCULTO];                        i += NEURONIOS_OCULTO

        t = NEURONIOS_OCULTO * N_ACTIONS
        self.W2 = p[i:i+t].reshape(NEURONIOS_OCULTO, N_ACTIONS);  i += t
        self.b2 = p[i:i+N_ACTIONS]

    def pensar(self, obs):
        """
        Recebe as observações do ambiente e retorna qual ação tomar.
        É aqui que os neurônios 'pensam'.
        """
        x = np.array(obs)

        # Camada 1: multiplica pelos pesos e aplica tanh (ativa entre -1 e 1)
        x = np.tanh(x @ self.W1 + self.b1)

        # Camada 2: gera os valores para cada ação
        x = x @ self.W2 + self.b2

        # Escolhe a ação com maior valor
        return int(np.argmax(x))

    def mutar(self, taxa, forca):
        """Cria um filho com pequenas variações aleatórias nos pesos."""
        filho_pesos = self.pesos.copy()
        # Para cada peso, com probabilidade 'taxa', adiciona um ruído aleatório
        mascara = np.random.random(len(filho_pesos)) < taxa
        filho_pesos[mascara] += np.random.randn(mascara.sum()) * forca
        return RedeNeural(filho_pesos)

    def get_ativacoes(self, obs):
        """Retorna as ativações de cada camada (para visualizar a rede)."""
        x = np.array(obs)
        entrada = x.copy()
        oculta  = np.tanh(x @ self.W1 + self.b1)
        saida   = oculta @ self.W2 + self.b2
        return entrada, oculta, saida

