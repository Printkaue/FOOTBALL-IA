"""
algoritmo_genetico_vs.py — Co-evolucao: Atacante vs Defensor
=============================================================

Agente 1 (Atacante): tenta fazer gols.
Agente 2 (Defensor): tenta manter a bola longe do atacante e do gol.

A pontuacao de cada um depende do outro:
  - Atacante ganha pontos por gols e por chegar perto da bola.
  - Defensor ganha pontos por manter a bola longe do gol e do atacante.
  - Atacante perde pontos quando o defensor afasta a bola.
  - Defensor perde pontos quando o atacante faz gol.

Os dois evoluem ao mesmo tempo — populacoes separadas,
avaliadas jogando uma contra a outra.
"""

from rede_neural import RedeNeural
from settings import ELITE, PASSOS_EPISODIO, POPULACAO, TAXA_MUTACAO, FORCA_MUTACAO
import numpy as np
from football_env import FootballEnv
from utils import salvar_modelo, carregar_modelo


class AlgoritmoGeneticoVS:
    """
    Treina dois agentes rivais simultaneamente.

    populacao_atk  — atacantes (tentam fazer gol)
    populacao_def  — defensores (tentam afastar a bola)

    A cada geracao, cada atacante enfrenta um defensor aleatorio
    do top da populacao defensora, e vice-versa.
    """

    def __init__(self, caminho_atk=None, caminho_def=None):
        # Carrega modelos existentes ou cria do zero
        if caminho_atk:
            pesos_atk = carregar_modelo(caminho_atk, RedeNeural)
            self.populacao_atk = [RedeNeural(pesos_atk) for _ in range(POPULACAO)]
            print(f"Atacante carregado: {caminho_atk}")
        else:
            self.populacao_atk = [RedeNeural() for _ in range(POPULACAO)]

        if caminho_def:
            pesos_def = carregar_modelo(caminho_def, RedeNeural)
            self.populacao_def = [RedeNeural(pesos_def) for _ in range(POPULACAO)]
            print(f"Defensor carregado: {caminho_def}")
        else:
            self.populacao_def = [RedeNeural() for _ in range(POPULACAO)]

        self.fitness_atk = [0.0] * POPULACAO
        self.fitness_def = [0.0] * POPULACAO

        self.geracao = 0
        self.historico_atk = []   # melhor fitness atacante por geracao
        self.historico_def = []   # melhor fitness defensor por geracao
        self.historico_media_atk = []
        self.historico_media_def = []

    # ── Avaliacao ────────────────────────────────────────────────────────────

    def avaliar_todos(self, callback_progresso=None):
        """
        Cada atacante enfrenta um defensor sorteado do top 10.
        Cada defensor enfrenta um atacante sorteado do top 10.
        """
        env = FootballEnv(render_mode=False)

        # Ordena as populacoes pelo fitness atual para selecionar oponentes
        ordem_atk = sorted(range(POPULACAO), key=lambda i: self.fitness_atk[i], reverse=True)
        ordem_def = sorted(range(POPULACAO), key=lambda i: self.fitness_def[i], reverse=True)
        top = min(10, POPULACAO)

        for i in range(POPULACAO):
            # Atacante i vs defensor aleatorio do top
            oponente_def = self.populacao_def[ordem_def[np.random.randint(0, top)]]
            fa, fd = self._disputar(self.populacao_atk[i], oponente_def, env)
            self.fitness_atk[i] = fa

            if callback_progresso:
                callback_progresso(i)

        for i in range(POPULACAO):
            # Defensor i vs atacante aleatorio do top
            oponente_atk = self.populacao_atk[ordem_atk[np.random.randint(0, top)]]
            fa, fd = self._disputar(oponente_atk, self.populacao_def[i], env)
            self.fitness_def[i] = fd

            if callback_progresso:
                callback_progresso(POPULACAO + i)

    def _disputar(self, atacante, defensor, env):
        """
        Roda um episodio com atacante vs defensor.
        Retorna (fitness_atacante, fitness_defensor).

        Posicoes:
          - Atacante: lado esquerdo (mesmo que o treinamento solo)
          - Defensor: lado direito, proximo ao gol que deve proteger

        Observacoes:
          - Atacante recebe obs normal do football_env
          - Defensor recebe obs espelhada — para ele o gol a defender
            sempre aparece como se estivesse a sua frente
        """
        cx = env.field_rect.centerx
        cy = env.field_rect.centery

        # Reseta o ambiente normalmente
        obs_atk = env.reset()

        # Posiciona o defensor proximo ao gol (lado direito)
        pos_def = [float(env.field_rect.right - 80), float(cy)]
        dir_def = math.pi   # aponta para a esquerda

        fitness_atk = 0.0
        fitness_def = 0.0

        for _ in range(PASSOS_EPISODIO):
            # Observacao do atacante — igual ao treinamento solo
            acao_atk = atacante.pensar(obs_atk)

            # Observacao do defensor — espelhada para ele sempre ver o gol a defender
            obs_def = _get_obs_defensor(env, pos_def)
            acao_def = defensor.pensar(obs_def)

            # Aplica acao do atacante no ambiente
            obs_atk, recompensa_env, done, info = env.step(acao_atk)

            # Move o defensor manualmente
            dx, dy = _acao_para_delta(acao_def)
            pos_def[0] = max(env.field_rect.left  + PLAYER_RADIUS_DEF,
                         min(env.field_rect.right - PLAYER_RADIUS_DEF, pos_def[0] + dx))
            pos_def[1] = max(env.field_rect.top   + PLAYER_RADIUS_DEF,
                         min(env.field_rect.bottom - PLAYER_RADIUS_DEF, pos_def[1] + dy))
            if dx != 0 or dy != 0:
                dir_def = math.atan2(dy, dx)

            # Defensor chuta a bola para longe
            if acao_def == ACTION_KICK:
                _defensor_chuta(env, pos_def)

            # Colisao do defensor com a bola
            _colisao_defensor(env, pos_def)

            # ── Recompensas ──────────────────────────────────────────────────

            dist_bola_gol  = _dist(env.ball_pos,
                                   [env.goal_rect.centerx, env.goal_rect.centery])
            dist_atk_bola  = _dist(env.player_pos, env.ball_pos)
            dist_def_bola  = _dist(pos_def, env.ball_pos)
            diag = math.hypot(env.field_rect.width, env.field_rect.height)

            # Atacante: recompensa do ambiente + proximidade a bola
            fitness_atk += recompensa_env
            if acao_atk == 0:
                fitness_atk -= 0.5   # penaliza ficar parado
            fitness_atk -= 0.01 * (dist_atk_bola / diag)

            # Defensor: quanto mais longe a bola do gol, melhor
            fitness_def += 0.02 * (dist_bola_gol / diag)
            # Defensor: bonus por ficar perto da bola (para interceptar)
            fitness_def -= 0.01 * (dist_def_bola / diag)
            # Defensor: penalidade pesada se o atacante fizer gol
            if info["score"] > 0 and env.episode_steps > 1:
                fitness_def -= 15.0

            if done:
                break

        # Bonus finais
        fitness_atk += info["score"] * 20.0        # atacante: gols valem muito
        fitness_def -= info["score"] * 15.0        # defensor: perde por gol tomado

        # Defensor: bonus pela posicao final da bola (longe do gol = bom)
        dist_final = _dist(env.ball_pos,
                           [env.goal_rect.centerx, env.goal_rect.centery])
        fitness_def += 3.0 * (dist_final / diag)

        return fitness_atk, fitness_def

    # ── Nova geracao ─────────────────────────────────────────────────────────

    def nova_geracao(self):
        """Evolui as duas populacoes independentemente."""

        # ── Atacantes ──
        ordem_atk = sorted(range(POPULACAO),
                           key=lambda i: self.fitness_atk[i], reverse=True)
        melhor_atk = self.fitness_atk[ordem_atk[0]]
        media_atk  = sum(self.fitness_atk) / POPULACAO
        self.historico_atk.append(melhor_atk)
        self.historico_media_atk.append(media_atk)

        salvar_modelo(self.populacao_atk[ordem_atk[0]],
                      f"modelos/atk_ger{self.geracao:04d}_fit{melhor_atk:.1f}.npy")

        nova_atk = []
        for i in range(min(ELITE, len(ordem_atk))):
            nova_atk.append(RedeNeural(self.populacao_atk[ordem_atk[i]].pesos))
        while len(nova_atk) < POPULACAO:
            top = min(10, len(ordem_atk))
            pai = self.populacao_atk[ordem_atk[np.random.randint(0, top)]]
            nova_atk.append(pai.mutar(TAXA_MUTACAO, FORCA_MUTACAO))

        # ── Defensores ──
        ordem_def = sorted(range(POPULACAO),
                           key=lambda i: self.fitness_def[i], reverse=True)
        melhor_def = self.fitness_def[ordem_def[0]]
        media_def  = sum(self.fitness_def) / POPULACAO
        self.historico_def.append(melhor_def)
        self.historico_media_def.append(media_def)

        salvar_modelo(self.populacao_def[ordem_def[0]],
                      f"modelos/def_ger{self.geracao:04d}_fit{melhor_def:.1f}.npy")

        nova_def = []
        for i in range(min(ELITE, len(ordem_def))):
            nova_def.append(RedeNeural(self.populacao_def[ordem_def[i]].pesos))
        while len(nova_def) < POPULACAO:
            top = min(10, len(ordem_def))
            pai = self.populacao_def[ordem_def[np.random.randint(0, top)]]
            nova_def.append(pai.mutar(TAXA_MUTACAO, FORCA_MUTACAO))

        self.populacao_atk = nova_atk
        self.populacao_def = nova_def
        self.fitness_atk   = [0.0] * POPULACAO
        self.fitness_def   = [0.0] * POPULACAO
        self.geracao      += 1

        print(f"  Ger {self.geracao-1:03d} | "
              f"ATK melhor={melhor_atk:.1f} media={media_atk:.1f} | "
              f"DEF melhor={melhor_def:.1f} media={media_def:.1f}")

    def melhor_atacante(self):
        idx = int(np.argmax(self.fitness_atk))
        return self.populacao_atk[idx]

    def melhor_defensor(self):
        idx = int(np.argmax(self.fitness_def))
        return self.populacao_def[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCOES AUXILIARES (fisica do defensor)
# ═══════════════════════════════════════════════════════════════════════════════

import math

PLAYER_RADIUS_DEF = 18   # mesmo raio do atacante
KICK_FORCE_DEF    = 12.0
KICK_RADIUS_DEF   = 38

ACTION_NONE  = 0
ACTION_UP    = 1
ACTION_DOWN  = 2
ACTION_LEFT  = 3
ACTION_RIGHT = 4
ACTION_KICK  = 5
PLAYER_SPEED_DEF = 4.0


def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _acao_para_delta(acao):
    if acao == ACTION_UP:    return 0.0, -PLAYER_SPEED_DEF
    if acao == ACTION_DOWN:  return 0.0,  PLAYER_SPEED_DEF
    if acao == ACTION_LEFT:  return -PLAYER_SPEED_DEF, 0.0
    if acao == ACTION_RIGHT: return  PLAYER_SPEED_DEF, 0.0
    return 0.0, 0.0


def _defensor_chuta(env, pos_def):
    """Defensor chuta a bola PARA LONGE do gol — direção oposta ao gol."""
    if _dist(pos_def, env.ball_pos) < KICK_RADIUS_DEF:
        # Angulo da bola em relacao ao defensor
        angle = math.atan2(
            env.ball_pos[1] - pos_def[1],
            env.ball_pos[0] - pos_def[0]
        )
        # Chuta na mesma direcao (empurra para longe do gol)
        env.ball_vel[0] = math.cos(angle) * KICK_FORCE_DEF
        env.ball_vel[1] = math.sin(angle) * KICK_FORCE_DEF


def _colisao_defensor(env, pos_def):
    """Colisao fisica entre defensor e bola."""
    dist     = _dist(pos_def, env.ball_pos)
    min_dist = PLAYER_RADIUS_DEF + env.goal_rect.width  # reutiliza BALL_RADIUS do env

    # Usa o BALL_RADIUS do proprio ambiente
    from settings import BALL_RADIUS
    min_dist = PLAYER_RADIUS_DEF + BALL_RADIUS

    if dist < min_dist and dist > 0:
        angle = math.atan2(
            env.ball_pos[1] - pos_def[1],
            env.ball_pos[0] - pos_def[0]
        )
        overlap = min_dist - dist
        env.ball_pos[0] += math.cos(angle) * overlap
        env.ball_pos[1] += math.sin(angle) * overlap
        if math.hypot(env.ball_vel[0], env.ball_vel[1]) < 1.0:
            env.ball_vel[0] += math.cos(angle) * 1.5
            env.ball_vel[1] += math.sin(angle) * 1.5


def _get_obs_defensor(env, pos_def):
    """
    Observacao do defensor — espelhada no eixo X.
    Para o defensor, o 'gol a defender' aparece sempre como se
    estivesse a sua direita, igual ao atacante ve o gol a atacar.

    Isso permite usar a mesma arquitetura de rede para os dois.
    """
    fw = env.field_rect.width
    fh = env.field_rect.height
    ox = env.field_rect.left
    oy = env.field_rect.top

    px = 1.0 - (pos_def[0]          - ox) / fw   # espelhado
    py =       (pos_def[1]          - oy) / fh
    bx = 1.0 - (env.ball_pos[0]     - ox) / fw   # espelhado
    by =       (env.ball_pos[1]     - oy) / fh
    vx =      -env.ball_vel[0] / KICK_FORCE_DEF   # velocidade espelhada
    vy =       env.ball_vel[1] / KICK_FORCE_DEF
    # Gol a defender: do ponto de vista do defensor e o goal_rect (direita)
    # Espelhado, ele o ve como se estivesse a sua "direita" tambem
    gx = 1.0 - (env.goal_rect.centerx - ox) / fw
    gy =       (env.goal_rect.centery - oy) / fh

    return [px, py, bx, by, vx, vy, gx, gy]