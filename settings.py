#Configurações da rede neural

POPULACAO        = 50      # quantos agentes por geração
PASSOS_EPISODIO  = 600     # quantos frames cada agente joga
TAXA_MUTACAO     = 0.20    # chance de um peso ser mutado (0.0 a 1.0)
FORCA_MUTACAO    = 0.4     # o quanto um peso muda quando é mutado
ELITE            = 5       # quantos melhores sobrevivem direto (sem mutar)
NEURONIOS_OCULTO = 12      # neurônios na camada do meio da rede
N_INPUTS         = 8       # entradas da rede (mesmo que obs do ambiente)
VELOCIDADE_SIM   = 3     # 1=lento, 3=normal, 10=rápido
FPS              = 30

#configurações do ambiente

WIDTH, HEIGHT = 900, 600
FPS_game = 60

# Campo
FIELD_MARGIN = 50
FIELD_COLOR   = (34, 139, 34)
LINE_COLOR    = (255, 255, 255)
GRASS_DARK    = (30, 120, 30)

# Gol
GOAL_WIDTH  = 12
GOAL_HEIGHT = 120
GOAL_COLOR  = (220, 220, 220)
GOAL_NET    = (200, 200, 200)

# Jogador
PLAYER_RADIUS  = 18
PLAYER_COLOR   = (30, 80, 200)
PLAYER_OUTLINE = (10, 40, 140)
PLAYER_SPEED   = 4.0
KICK_RADIUS    = PLAYER_RADIUS + 20   # distância para chutar
KICK_FORCE     = 12.0

# Bola
BALL_RADIUS  = 11
BALL_COLOR   = (245, 245, 245)
BALL_OUTLINE = (30, 30, 30)
BALL_FRICTION = 0.97   # desaceleração por frame

# Ações discretas (úteis para RL)
ACTION_NONE  = 0
ACTION_UP    = 1
ACTION_DOWN  = 2
ACTION_LEFT  = 3
ACTION_RIGHT = 4
ACTION_KICK  = 5
N_ACTIONS    = 6
