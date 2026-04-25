"""
test_env.py — Teste de Agente
==============================

Roda um agente salvo no mesmo ambiente em que foi treinado.
Tudo identico ao football_env.py — mesmas posicoes, mesma fisica,
mesmas observacoes.

Uso:
    python test_env.py --agente modelos/CR7_F140.npy
    python test_env.py --agente modelos/CR7_F140.npy --modo humano
    python test_env.py --agente1 modelos/A.npy --agente2 modelos/B.npy --modo versus

Controles (modo humano):
    WASD / Setas  — mover
    ESPACO        — chutar
    R             — reiniciar
    ESC           — sair
"""

import pygame
import math
import sys
import argparse
import numpy as np
import settings
from rede_neural import RedeNeural
from utils import carregar_modelo_para_testes

# ── Constantes identicas ao football_env.py ───────────────────────────────────

WIDTH, HEIGHT = settings.WIDTH, settings.HEIGHT
FPS           = settings.FPS_game
FIELD_MARGIN  = settings.FIELD_MARGIN
FIELD_COLOR   = settings.FIELD_COLOR
LINE_COLOR    = settings.LINE_COLOR
GRASS_DARK    = settings.GRASS_DARK
GOAL_WIDTH    = settings.GOAL_WIDTH
GOAL_HEIGHT   = settings.GOAL_HEIGHT
GOAL_COLOR    = settings.GOAL_COLOR
GOAL_NET      = settings.GOAL_NET
PLAYER_RADIUS = settings.PLAYER_RADIUS
PLAYER_COLOR  = settings.PLAYER_COLOR
PLAYER_SPEED  = settings.PLAYER_SPEED
KICK_RADIUS   = settings.KICK_RADIUS
KICK_FORCE    = settings.KICK_FORCE
BALL_RADIUS   = settings.BALL_RADIUS
BALL_COLOR    = settings.BALL_COLOR
BALL_OUTLINE  = settings.BALL_OUTLINE
BALL_FRICTION = settings.BALL_FRICTION

N_INPUTS         = settings.N_INPUTS
NEURONIOS_OCULTO = settings.NEURONIOS_OCULTO
N_ACTIONS        = settings.N_ACTIONS

ACTION_NONE  = 0
ACTION_UP    = 1
ACTION_DOWN  = 2
ACTION_LEFT  = 3
ACTION_RIGHT = 4
ACTION_KICK  = 5

NOMES_ACOES = ["Nada", "Cima", "Baixo", "Esq", "Dir", "Chutar"]

LARGURA_PAINEL = 360
COR_PAINEL     = (15, 15, 25)
COR_TEXTO      = (200, 220, 255)
COR_DESTAQUE   = (80, 200, 120)
COR_ALERTA     = (255, 180, 50)
COR_NEU_POS    = (50, 200, 100)
COR_NEU_NEG    = (200, 60, 60)
COR_J2         = (200, 50, 50)
COR_J2_OUT     = (140, 10, 10)


# ═══════════════════════════════════════════════════════════════════════════════
# AMBIENTE
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnv:
    """
    Replica exata do FootballEnv de treinamento.
    reset() usa as mesmas posicoes que o agente aprendeu.
    """

    def __init__(self, agente1, agente2=None, modo='agente'):
        self.agente1 = agente1
        self.agente2 = agente2
        self.modo    = modo

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH + LARGURA_PAINEL, HEIGHT))
        pygame.display.set_caption("Football AI — Teste")
        self.clock  = pygame.time.Clock()
        self._load_fonts()

        self.surf_campo = pygame.Surface((WIDTH, HEIGHT))

        # Geometria identica ao football_env
        self.field_rect = pygame.Rect(
            FIELD_MARGIN, FIELD_MARGIN,
            WIDTH  - 2 * FIELD_MARGIN,
            HEIGHT - 2 * FIELD_MARGIN
        )
        goal_x = self.field_rect.right
        goal_y = self.field_rect.centery - GOAL_HEIGHT // 2
        self.goal_rect = pygame.Rect(goal_x, goal_y, GOAL_WIDTH, GOAL_HEIGHT)

        self.score1 = 0
        self.score2 = 0
        self.reset()

    # ── Reset — espelho do football_env.reset() ───────────────────────────────

    def reset(self):
        cx = self.field_rect.centerx
        cy = self.field_rect.centery

        # POSICAO IDENTICA ao football_env.py atual
        # Altere aqui se alterar no football_env
        self.player_pos = [cx - 120, cy]
        self.player_dir = 0.0

        # Jogador 2 — lado oposto do campo
        self.player2_pos = [float(self.field_rect.right - 120), float(cy)]
        self.player2_dir = math.pi

        self.ball_pos      = [float(cx), float(cy)]
        self.ball_vel      = [0.0, 0.0]
        self.episode_steps = 0
        self.done          = False
        self.message       = ""
        self.msg_timer     = 0

    def reset_placar(self):
        self.score1 = 0
        self.score2 = 0

    # ── Observacao — identica ao football_env._get_obs() ─────────────────────

    def _get_obs(self, player_pos):
        fw = self.field_rect.width
        fh = self.field_rect.height
        ox = self.field_rect.left
        oy = self.field_rect.top
        return [
            (player_pos[0]          - ox) / fw,
            (player_pos[1]          - oy) / fh,
            (self.ball_pos[0]       - ox) / fw,
            (self.ball_pos[1]       - oy) / fh,
            self.ball_vel[0] / KICK_FORCE,
            self.ball_vel[1] / KICK_FORCE,
            (self.goal_rect.centerx - ox) / fw,
            (self.goal_rect.centery - oy) / fh,
        ]

    # ── Fisica — identica ao football_env ────────────────────────────────────

    def _mover(self, pos, action):
        dx, dy = 0.0, 0.0
        if action == ACTION_UP:      dy = -PLAYER_SPEED
        elif action == ACTION_DOWN:  dy =  PLAYER_SPEED
        elif action == ACTION_LEFT:  dx = -PLAYER_SPEED
        elif action == ACTION_RIGHT: dx =  PLAYER_SPEED
        direcao = math.atan2(dy, dx) if (dx or dy) else None
        pos[0] = max(self.field_rect.left  + PLAYER_RADIUS,
                 min(self.field_rect.right - PLAYER_RADIUS, pos[0] + dx))
        pos[1] = max(self.field_rect.top   + PLAYER_RADIUS,
                 min(self.field_rect.bottom - PLAYER_RADIUS, pos[1] + dy))
        return direcao

    def _try_kick(self, pos):
        if self._dist(pos, self.ball_pos) < KICK_RADIUS:
            angle = math.atan2(self.ball_pos[1]-pos[1], self.ball_pos[0]-pos[0])
            self.ball_vel[0] = math.cos(angle) * KICK_FORCE
            self.ball_vel[1] = math.sin(angle) * KICK_FORCE

    def _wall_bounce(self):
        fr = self.field_rect
        br = BALL_RADIUS
        if self.ball_pos[0] - br < fr.left:
            self.ball_pos[0] = fr.left + br;  self.ball_vel[0] *= -0.7
        elif self.ball_pos[0] + br > fr.right:
            if not (self.goal_rect.top < self.ball_pos[1] < self.goal_rect.bottom):
                self.ball_pos[0] = fr.right - br;  self.ball_vel[0] *= -0.7
        if self.ball_pos[1] - br < fr.top:
            self.ball_pos[1] = fr.top + br;    self.ball_vel[1] *= -0.7
        elif self.ball_pos[1] + br > fr.bottom:
            self.ball_pos[1] = fr.bottom - br; self.ball_vel[1] *= -0.7

    def _col_bola(self, pos):
        dist = self._dist(pos, self.ball_pos)
        md   = PLAYER_RADIUS + BALL_RADIUS
        if dist < md and dist > 0:
            angle = math.atan2(self.ball_pos[1]-pos[1], self.ball_pos[0]-pos[0])
            ov = md - dist
            self.ball_pos[0] += math.cos(angle) * ov
            self.ball_pos[1] += math.sin(angle) * ov
            if math.hypot(*self.ball_vel) < 1.0:
                self.ball_vel[0] += math.cos(angle) * 1.5
                self.ball_vel[1] += math.sin(angle) * 1.5

    def _check_goal(self):
        bx, by = self.ball_pos
        gr = self.goal_rect
        return bx + BALL_RADIUS > gr.left and gr.top < by < gr.bottom

    def _reset_ball(self):
        cx = self.field_rect.centerx
        cy = self.field_rect.centery
        self.ball_pos = [float(cx), float(cy)]
        self.ball_vel = [0.0, 0.0]

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action1, action2=ACTION_NONE):
        self.episode_steps += 1

        d = self._mover(self.player_pos, action1)
        if d is not None: self.player_dir = d
        if action1 == ACTION_KICK: self._try_kick(self.player_pos)

        if self.modo in ('humano', 'versus'):
            d2 = self._mover(self.player2_pos, action2)
            if d2 is not None: self.player2_dir = d2
            if action2 == ACTION_KICK: self._try_kick(self.player2_pos)

        self.ball_vel[0] *= BALL_FRICTION
        self.ball_vel[1] *= BALL_FRICTION
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        self._wall_bounce()
        self._col_bola(self.player_pos)
        if self.modo in ('humano', 'versus'):
            self._col_bola(self.player2_pos)

        if self._check_goal():
            self.score1 += 1
            self.message   = "GOL!"
            self.msg_timer = 120
            self._reset_ball()

        if self.msg_timer > 0:
            self.msg_timer -= 1

    # ═══════════════════════════════════════════════════════════════════════════
    # RENDERIZACAO
    # ═══════════════════════════════════════════════════════════════════════════

    def render(self, obs1):
        self._draw_campo()
        self._draw_painel(obs1)
        pygame.display.flip()

    def _draw_campo(self):
        s  = self.surf_campo
        fr = self.field_rect

        s.fill((20, 90, 20))
        pygame.draw.rect(s, FIELD_COLOR, fr)
        stripe_w = 60
        for i, x in enumerate(range(fr.left, fr.right, stripe_w)):
            pygame.draw.rect(s, FIELD_COLOR if i%2==0 else GRASS_DARK,
                             (x, fr.top, stripe_w, fr.height))

        pygame.draw.rect(s, LINE_COLOR, fr, 3)
        mx = fr.centerx
        pygame.draw.line(s, LINE_COLOR, (mx, fr.top), (mx, fr.bottom), 2)
        pygame.draw.circle(s, LINE_COLOR, fr.center, 70, 2)
        pygame.draw.circle(s, LINE_COLOR, fr.center, 4)

        area_w, area_h = 120, 240
        pygame.draw.rect(s, LINE_COLOR,
            pygame.Rect(fr.right-area_w, fr.centery-area_h//2, area_w, area_h), 2)
        pygame.draw.circle(s, LINE_COLOR, (fr.right-90, fr.centery), 4)

        gr = self.goal_rect
        for y in range(gr.top, gr.bottom, 15):
            pygame.draw.line(s, GOAL_NET, (gr.left,y), (gr.right,y), 1)
        for x in range(gr.left, gr.right, 15):
            pygame.draw.line(s, GOAL_NET, (x,gr.top), (x,gr.bottom), 1)
        pygame.draw.rect(s, GOAL_COLOR, gr, 3)
        pygame.draw.line(s, (255,255,255), (gr.left,gr.top),    (gr.right,gr.top),    4)
        pygame.draw.line(s, (255,255,255), (gr.left,gr.bottom), (gr.right,gr.bottom), 4)
        pygame.draw.line(s, (255,255,255), (gr.left,gr.top),    (gr.left,gr.bottom),  4)

        bx, by = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.draw.circle(s, (0,80,0),      (bx+2,by+2), BALL_RADIUS)
        pygame.draw.circle(s, BALL_COLOR,    (bx,by),     BALL_RADIUS)
        pygame.draw.circle(s, BALL_OUTLINE,  (bx,by),     BALL_RADIUS, 2)
        pygame.draw.circle(s, (60,60,60),    (bx,by),     BALL_RADIUS//2, 1)

        self._draw_player(s, self.player_pos, self.player_dir,
                          PLAYER_COLOR, (10,40,140), "1")
        if self.modo in ('humano', 'versus'):
            self._draw_player(s, self.player2_pos, self.player2_dir,
                              COR_J2, COR_J2_OUT, "2")

        placar = self.font_md.render(f"Gols: {self.score1}", True, (255,255,255))
        s.blit(placar, (WIDTH//2 - placar.get_width()//2, 8))

        modos = {'agente':'Agente','humano':'Humano vs Agente','versus':'Agente vs Agente'}
        mt = self.font_xs.render(modos.get(self.modo,''), True, (200,255,200))
        s.blit(mt, (WIDTH//2 - mt.get_width()//2, HEIGHT-26))

        if self.msg_timer > 0:
            alpha = min(255, self.msg_timer * 4)
            surf  = self.font_lg.render(self.message, True, (255,230,0))
            surf.set_alpha(alpha)
            s.blit(surf, surf.get_rect(center=(WIDTH//2, HEIGHT//2-60)))

        self.screen.blit(self.surf_campo, (0, 0))

    def _draw_player(self, s, pos, direcao, cor, cor_out, numero):
        px, py = int(pos[0]), int(pos[1])
        pygame.draw.circle(s, (0,80,0),  (px+3,py+3), PLAYER_RADIUS)
        pygame.draw.circle(s, cor,       (px,py),     PLAYER_RADIUS)
        pygame.draw.circle(s, cor_out,   (px,py),     PLAYER_RADIUS, 3)
        ex = px + int(math.cos(direcao) * (PLAYER_RADIUS-4))
        ey = py + int(math.sin(direcao) * (PLAYER_RADIUS-4))
        pygame.draw.line(s, (255,220,0), (px,py), (ex,ey), 3)
        num = self.font_sm.render(numero, True, (255,255,255))
        s.blit(num, num.get_rect(center=(px,py)))

    def _draw_painel(self, obs):
        px = WIDTH
        s  = self.screen
        pygame.draw.rect(s, COR_PAINEL, (px, 0, LARGURA_PAINEL, HEIGHT))
        pygame.draw.line(s, (40,40,70), (px,0), (px,HEIGHT), 2)

        y = 12
        t = self.font_gg.render("Football AI — Teste", True, COR_DESTAQUE)
        s.blit(t, (px + LARGURA_PAINEL//2 - t.get_width()//2, y)); y += 30

        self._linha(s, px, y, "Modo",  self.modo,              COR_TEXTO);  y += 20
        self._linha(s, px, y, "Gols",  str(self.score1),       COR_ALERTA); y += 20
        self._linha(s, px, y, "Frame", str(self.episode_steps), COR_TEXTO); y += 28

        y = self._draw_rede(s, px, y, obs)
        y += 12
        self._linha(s, px, y, "R",   "Reiniciar", (100,100,150)); y += 18
        self._linha(s, px, y, "ESC", "Sair",       (100,100,150))

    def _linha(self, s, px, y, label, valor, cor):
        t1 = self.font_p.render(f"{label}:", True, (100,110,150))
        t2 = self.font_m.render(valor, True, cor)
        s.blit(t1, (px+12, y))
        s.blit(t2, (px + LARGURA_PAINEL - t2.get_width() - 12, y))

    def _draw_rede(self, s, px, y_ini, obs):
        t = self.font_m.render("REDE NEURAL — Agente 1", True, COR_TEXTO)
        s.blit(t, (px + LARGURA_PAINEL//2 - t.get_width()//2, y_ini))
        y_ini += 20

        if obs is None: return y_ini + 190

        entrada, oculta, saida = self.agente1.get_ativacoes(obs)
        saida_norm = saida - saida.min()
        if saida_norm.max() > 0: saida_norm /= saida_norm.max()

        cx  = px + LARGURA_PAINEL//2
        y_e = y_ini + 20
        y_o = y_ini + 80
        y_s = y_ini + 148

        def posX(n, w=300):
            if n == 1: return [cx]
            return [cx - w//2 + i*(w//(n-1)) for i in range(n)]

        pe = posX(N_INPUTS, 310)
        po = posX(NEURONIOS_OCULTO, 330)
        ps = posX(N_ACTIONS, 290)

        for j in range(NEURONIOS_OCULTO):
            for i in range(0, N_INPUTS, 2):
                w = self.agente1.W1[i,j]; a = min(200, int(abs(w)*80))
                pygame.draw.line(s, (0,a,0) if w>0 else (a,0,0), (pe[i],y_e),(po[j],y_o), 1)
        for i in range(NEURONIOS_OCULTO):
            for j in range(N_ACTIONS):
                w = self.agente1.W2[i,j]; a = min(200, int(abs(w)*80))
                pygame.draw.line(s, (0,a,0) if w>0 else (a,0,0), (po[i],y_o),(ps[j],y_s), 1)

        for x, val in zip(pe, entrada):
            v = int((val+1)/2*255)
            pygame.draw.circle(s, (v,v,min(255,v+50)), (x,y_e), 7)
            pygame.draw.circle(s, (150,150,200), (x,y_e), 7, 1)

        for x, val in zip(po, oculta):
            cor = COR_NEU_POS if val > 0 else COR_NEU_NEG
            b   = int(abs(val)*200)
            cor = tuple(min(255, c*b//200) for c in cor)
            pygame.draw.circle(s, cor, (x,y_o), 8)
            pygame.draw.circle(s, (150,150,200), (x,y_o), 8, 1)

        acao = int(np.argmax(saida))
        for i, (x, val) in enumerate(zip(ps, saida_norm)):
            v   = int(val*220)
            cor = (v,v,50) if i==acao else (50,50,int(val*180))
            pygame.draw.circle(s, cor, (x,y_s), 10)
            pygame.draw.circle(s, COR_ALERTA if i==acao else (100,100,150), (x,y_s), 10, 2)
            lb = self.font_p.render(NOMES_ACOES[i], True, COR_ALERTA if i==acao else (100,100,150))
            s.blit(lb, (x - lb.get_width()//2, y_s+12))

        for txt, yp in [("Entradas",y_e-16),("Oculta",y_o-16),("Acoes",y_s-16)]:
            s.blit(self.font_p.render(txt, True, (80,90,130)), (px+10, yp))

        y_acao = y_s + 24
        at = self.font_m.render(f"Acao: {NOMES_ACOES[acao]}", True, COR_ALERTA)
        s.blit(at, (px + LARGURA_PAINEL//2 - at.get_width()//2, y_acao))
        return y_acao + 26

    def _load_fonts(self):
        self.font_xs = pygame.font.SysFont("monospace", 13)
        self.font_sm = pygame.font.SysFont("monospace", 11, bold=True)
        self.font_md = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 44, bold=True)
        self.font_p  = pygame.font.SysFont("monospace", 12)
        self.font_m  = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_gg = pygame.font.SysFont("monospace", 16, bold=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOOPS
# ═══════════════════════════════════════════════════════════════════════════════

def eventos(env):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()
            if event.key == pygame.K_r:
                env.reset()
                env.reset_placar()


def loop_agente(env):
    while True:
        eventos(env)
        obs  = env._get_obs(env.player_pos)
        acao = env.agente1.pensar(obs)
        env.step(acao)
        env.render(obs)
        env.clock.tick(FPS)


def loop_humano(env):
    while True:
        eventos(env)
        keys  = pygame.key.get_pressed()
        acao2 = ACTION_NONE
        if keys[pygame.K_SPACE]:                        acao2 = ACTION_KICK
        elif keys[pygame.K_w] or keys[pygame.K_UP]:    acao2 = ACTION_UP
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:  acao2 = ACTION_DOWN
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:  acao2 = ACTION_LEFT
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: acao2 = ACTION_RIGHT

        obs  = env._get_obs(env.player_pos)
        acao = env.agente1.pensar(obs)
        env.step(acao, acao2)
        env.render(obs)
        env.clock.tick(FPS)


def loop_versus(env):
    while True:
        eventos(env)
        obs1  = env._get_obs(env.player_pos)
        acao1 = env.agente1.pensar(obs1)
        obs2  = env._get_obs(env.player2_pos)
        acao2 = env.agente2.pensar(obs2)
        env.step(acao1, acao2)
        env.render(obs1)
        env.clock.tick(FPS)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agente",  type=str, default=None)
    parser.add_argument("--agente1", type=str, default=None)
    parser.add_argument("--agente2", type=str, default=None)
    parser.add_argument("--modo",    type=str, default="agente",
                        choices=["agente","humano","versus"])
    args = parser.parse_args()

    if args.agente: args.agente1 = args.agente

    if not args.agente1:
        print("Uso: python test_env.py --agente modelos/seu_modelo.npy")
        sys.exit(1)

    agente1 = carregar_modelo_para_testes(args.agente1, RedeNeural)
    agente2 = carregar_modelo_para_testes(args.agente2, RedeNeural) if args.agente2 else None

    if args.modo == 'versus' and not agente2:
        print("Modo versus requer --agente2."); sys.exit(1)

    print(f"Agente 1 : {args.agente1}")
    if agente2: print(f"Agente 2 : {args.agente2}")
    print(f"Modo     : {args.modo}")
    print("R → reiniciar  |  ESC → sair\n")

    env = TestEnv(agente1=agente1, agente2=agente2, modo=args.modo)

    if args.modo == 'agente':   loop_agente(env)
    elif args.modo == 'humano': loop_humano(env)
    elif args.modo == 'versus': loop_versus(env)


if __name__ == "__main__":
    main()