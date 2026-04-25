import numpy as np

def salvar_modelo(rede, caminho):
    np.save(caminho, rede.pesos)

def carregar_modelo(caminho, RedeNeural):
    pesos = np.load(caminho)
    return pesos