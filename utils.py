import numpy as np

def salvar_modelo(rede, caminho):
    np.save(caminho, rede.pesos)

def carregar_modelo(caminho, RedeNeural):
    pesos = np.load(caminho)
<<<<<<< HEAD
    return pesos
=======
    return RedeNeural(pesos)
>>>>>>> 4415604d5cc563c3d787e4f797fd5eacc70bd555
