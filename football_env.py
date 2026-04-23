"""
football_env.py — Campo de Futebol 2D Top-Down (Pygame)

Estrutura preparada para agentes de IA:
  - A classe FootballEnv encapsula tudo (estado, física, renderização)
  - step(action) recebe uma ação discreta e retorna (obs, reward, done, info)
  - reset() reinicia o episódio
  - render() desenha o frame atual

Controles humanos (WASD / Setas):
  W / ↑  — mover para cima
  S / ↓  — mover para baixo
  A / ←  — mover para esquerda
  D / →  — mover para direita
  ESPAÇO  — chutar a bola (quando perto)
"""

import pygame
import math
import sys
import settings

# ─── Constantes ──────────────────────────────────────────────────────────────

WIDTH, HEIGHT = settings.WIDTH, settings.HEIGHT
FPS = settings.FPS_game

# Campo
FIELD_MARGIN = settings.FIELD_MARGIN
FIELD_COLOR   = settings.FIELD_COLOR
LINE_COLOR    = settings.LINE_COLOR
GRASS_DARK    = settings.GRASS_DARK

# Gol
GOAL_WIDTH  = settings.GOAL_WIDTH
GOAL_HEIGHT = settings.GOAL_HEIGHT
GOAL_COLOR  = settings.GOAL_COLOR
GOAL_NET    = settings.GOAL_NET

# Jogador
PLAYER_RADIUS  = settings.PLAYER_RADIUS
PLAYER_COLOR   = settings.PLAYER_COLOR
PLAYER_OUTLINE = settings.PLAYER_OUTLINE
PLAYER_SPEED   = settings.PLAYER_SPEED
KICK_RADIUS    = settings.KICK_RADIUS
KICK_FORCE     = settings.KICK_FORCE

# Bola
BALL_RADIUS  = settings.BALL_RADIUS
BALL_COLOR   = settings.BALL_COLOR
BALL_OUTLINE = settings.BALL_OUTLINE
BALL_FRICTION = settings.BALL_FRICTION

# Ações discretas 
ACTION_NONE  = 0
ACTION_UP    = 1
ACTION_DOWN  = 2
ACTION_LEFT  = 3
ACTION_RIGHT = 4
ACTION_KICK  = 5
N_ACTIONS    = 6


# ─── Classe principal ─────────────────────────────────────────────────────────

class FootballEnv:
    """
    Ambiente de futebol 2D.

    Para usar com IA, chame:
        env = FootballEnv(render_mode=False)
        obs = env.reset()
        obs, reward, done, info = env.step(action)

    Para jogar como humano:
        env = FootballEnv(render_mode=True)
        env.run_human()
    """

    def __init__(self, render_mode: bool = True):
        pygame.init()
        self.render_mode = render_mode

        if render_mode:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("⚽ Football Env — Pygame")
            self.clock = pygame.time.Clock()
            self._load_fonts()

        # Limites do campo
        self.field_rect = pygame.Rect(
            FIELD_MARGIN, FIELD_MARGIN,
            WIDTH  - 2 * FIELD_MARGIN,
            HEIGHT - 2 * FIELD_MARGIN
        )

        # Posição do gol (direita do campo)
        goal_x = self.field_rect.right
        goal_y = self.field_rect.centery - GOAL_HEIGHT // 2
        self.goal_rect = pygame.Rect(goal_x, goal_y, GOAL_WIDTH, GOAL_HEIGHT)

        self.reset()

    # ── Inicialização ────────────────────────────────────────────────────────

    def reset(self):
        """Reinicia o episódio e retorna a observação inicial."""
        cx = self.field_rect.centerx
        cy = self.field_rect.centery

        # Jogador começa no centro-esquerda
        self.player_pos  = [300 - 120, 200] # padrão [cx - 120, cy]
        self.player_dir  = 0.0      # ângulo de orientação (rad), visual only

        # Bola começa no centro
        self.ball_pos = [float(cx), float(cy)]
        self.ball_vel = [0.0, 0.0]

        self.score   = 0
        self.done    = False
        self.episode_steps = 0
        self.max_steps     = 3000
        self.message       = ""
        self.msg_timer     = 0

        return self._get_obs()

    # ── Passo do ambiente ────────────────────────────────────────────────────

    def step(self, action: int):
        """
        Executa uma ação e avança um frame.

        Retorna: (obs, reward, done, info)
        """
        reward = 0.0
        self.episode_steps += 1

        # Movimento do jogador
        dx, dy = 0.0, 0.0
        if action == ACTION_UP:    dy = -PLAYER_SPEED
        elif action == ACTION_DOWN:  dy =  PLAYER_SPEED
        elif action == ACTION_LEFT:  dx = -PLAYER_SPEED
        elif action == ACTION_RIGHT: dx =  PLAYER_SPEED

        if dx != 0 or dy != 0:
            self.player_dir = math.atan2(dy, dx)

        # Move e mantém dentro do campo
        self.player_pos[0] = max(
            self.field_rect.left  + PLAYER_RADIUS,
            min(self.field_rect.right  - PLAYER_RADIUS,
                self.player_pos[0] + dx)
        )
        self.player_pos[1] = max(
            self.field_rect.top   + PLAYER_RADIUS,
            min(self.field_rect.bottom - PLAYER_RADIUS,
                self.player_pos[1] + dy)
        )

        # Chute
        if action == ACTION_KICK:
            reward += self._try_kick()

        # Física da bola
        self.ball_vel[0] *= BALL_FRICTION
        self.ball_vel[1] *= BALL_FRICTION
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # Rebote nas paredes do campo
        self._ball_wall_bounce()

        # Colisão jogador-bola (empurrar)
        self._player_ball_collision()

        # Verifica gol
        if self._check_goal():
            self.score += 1
            reward += 10.0
            self._show_message("⚽ GOL!", 120)
            self._reset_ball()

        # Timeout
        if self.episode_steps >= self.max_steps:
            self.done = True

        # Decrementa timer da mensagem
        if self.msg_timer > 0:
            self.msg_timer -= 1

        # Pequena recompensa por proximidade à bola
        dist = self._dist(self.player_pos, self.ball_pos)
        reward -= 0.001 * dist

        info = {"score": self.score, "steps": self.episode_steps}
        return self._get_obs(), reward, self.done, info

    # ── Observação ───────────────────────────────────────────────────────────

    def _get_obs(self):
        """
        Vetor de observação normalizado [0-1] para uso com IA:
          [px, py, bx, by, bvx, bvy, gx, gy]
        """
        fw = self.field_rect.width
        fh = self.field_rect.height
        ox = self.field_rect.left
        oy = self.field_rect.top

        return [
            (self.player_pos[0] - ox) / fw,
            (self.player_pos[1] - oy) / fh,
            (self.ball_pos[0]   - ox) / fw,
            (self.ball_pos[1]   - oy) / fh,
            self.ball_vel[0] / KICK_FORCE,
            self.ball_vel[1] / KICK_FORCE,
            (self.goal_rect.centerx - ox) / fw,
            (self.goal_rect.centery - oy) / fh,
        ]

    # ── Física ───────────────────────────────────────────────────────────────

    def _try_kick(self):
        dist = self._dist(self.player_pos, self.ball_pos)
        if dist < KICK_RADIUS:
            angle = math.atan2(
                self.ball_pos[1] - self.player_pos[1],
                self.ball_pos[0] - self.player_pos[0]
            )
            self.ball_vel[0] = math.cos(angle) * KICK_FORCE
            self.ball_vel[1] = math.sin(angle) * KICK_FORCE
            return 0.5   # pequena recompensa por chutar
        return 0.0

    def _ball_wall_bounce(self):
        fr = self.field_rect
        br = BALL_RADIUS

        if self.ball_pos[0] - br < fr.left:
            self.ball_pos[0] = fr.left + br
            self.ball_vel[0] *= -0.7
        elif self.ball_pos[0] + br > fr.right:
            # Só rebate se não entrou no gol
            if not (self.goal_rect.top < self.ball_pos[1] < self.goal_rect.bottom):
                self.ball_pos[0] = fr.right - br
                self.ball_vel[0] *= -0.7

        if self.ball_pos[1] - br < fr.top:
            self.ball_pos[1] = fr.top + br
            self.ball_vel[1] *= -0.7
        elif self.ball_pos[1] + br > fr.bottom:
            self.ball_pos[1] = fr.bottom - br
            self.ball_vel[1] *= -0.7

    def _player_ball_collision(self):
        dist = self._dist(self.player_pos, self.ball_pos)
        min_dist = PLAYER_RADIUS + BALL_RADIUS
        if dist < min_dist and dist > 0:
            angle = math.atan2(
                self.ball_pos[1] - self.player_pos[1],
                self.ball_pos[0] - self.player_pos[0]
            )
            overlap = min_dist - dist
            self.ball_pos[0] += math.cos(angle) * overlap
            self.ball_pos[1] += math.sin(angle) * overlap
            # Transfere um pouco da velocidade
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if speed < 1.0:
                self.ball_vel[0] += math.cos(angle) * 1.5
                self.ball_vel[1] += math.sin(angle) * 1.5

    def _check_goal(self):
        bx, by = self.ball_pos
        gr = self.goal_rect
        return (bx + BALL_RADIUS > gr.left and
                gr.top < by < gr.bottom)

    def _reset_ball(self):
        self.ball_pos = [float(self.field_rect.centerx),
                         float(self.field_rect.centery)]
        self.ball_vel = [0.0, 0.0]

    # ── Renderização ──────────────────────────────────────────────────────────

    def render(self):
        if not self.render_mode:
            return
        self._draw_field()
        self._draw_goal()
        self._draw_ball()
        self._draw_player()
        self._draw_hud()
        if self.msg_timer > 0:
            self._draw_message()
        pygame.display.flip()

    def _draw_field(self):
        s = self.screen
        fr = self.field_rect

        # Fundo verde
        s.fill((20, 90, 20))
        pygame.draw.rect(s, FIELD_COLOR, fr)

        # Listras de grama
        stripe_w = 60
        for i, x in enumerate(range(fr.left, fr.right, stripe_w)):
            c = FIELD_COLOR if i % 2 == 0 else GRASS_DARK
            pygame.draw.rect(s, c, (x, fr.top, stripe_w, fr.height))

        # Bordas
        pygame.draw.rect(s, LINE_COLOR, fr, 3)

        # Linha do meio
        mx = fr.centerx
        pygame.draw.line(s, LINE_COLOR, (mx, fr.top), (mx, fr.bottom), 2)

        # Círculo central
        pygame.draw.circle(s, LINE_COLOR, fr.center, 70, 2)
        pygame.draw.circle(s, LINE_COLOR, fr.center, 4)

        # Área do gol (grande)
        area_w, area_h = 120, 240
        area = pygame.Rect(fr.right - area_w,
                           fr.centery - area_h // 2,
                           area_w, area_h)
        pygame.draw.rect(s, LINE_COLOR, area, 2)

        # Ponto de pênalti
        pygame.draw.circle(s, LINE_COLOR,
                           (fr.right - 90, fr.centery), 4)

    def _draw_goal(self):
        s = self.screen
        gr = self.goal_rect

        # Rede (linhas)
        net_x = gr.right
        for y in range(gr.top, gr.bottom, 15):
            pygame.draw.line(s, GOAL_NET, (gr.left, y), (net_x, y), 1)
        for x in range(gr.left, net_x, 15):
            pygame.draw.line(s, GOAL_NET, (x, gr.top), (x, gr.bottom), 1)

        # Traves
        pygame.draw.rect(s, GOAL_COLOR, gr, 3)
        pygame.draw.line(s, (255,255,255), (gr.left, gr.top),    (gr.right, gr.top),    4)
        pygame.draw.line(s, (255,255,255), (gr.left, gr.bottom), (gr.right, gr.bottom), 4)
        pygame.draw.line(s, (255,255,255), (gr.left, gr.top),    (gr.left, gr.bottom),  4)

    def _draw_player(self):
        s = self.screen
        px, py = int(self.player_pos[0]), int(self.player_pos[1])

        # Sombra
        pygame.draw.circle(s, (0, 80, 0), (px + 3, py + 3), PLAYER_RADIUS)

        # Corpo
        pygame.draw.circle(s, PLAYER_COLOR, (px, py), PLAYER_RADIUS)
        pygame.draw.circle(s, PLAYER_OUTLINE, (px, py), PLAYER_RADIUS, 3)

        # Indicador de direção
        end_x = px + int(math.cos(self.player_dir) * (PLAYER_RADIUS - 4))
        end_y = py + int(math.sin(self.player_dir) * (PLAYER_RADIUS - 4))
        pygame.draw.line(s, (255, 220, 0), (px, py), (end_x, end_y), 3)

        # Número
        num = self.font_sm.render("10", True, (255, 255, 255))
        s.blit(num, num.get_rect(center=(px, py)))

    def _draw_ball(self):
        s = self.screen
        bx, by = int(self.ball_pos[0]), int(self.ball_pos[1])

        # Sombra
        pygame.draw.circle(s, (0, 80, 0), (bx + 2, by + 2), BALL_RADIUS)

        # Bola
        pygame.draw.circle(s, BALL_COLOR, (bx, by), BALL_RADIUS)
        pygame.draw.circle(s, BALL_OUTLINE, (bx, by), BALL_RADIUS, 2)

        # Detalhes da bola (pentágono estilizado)
        pygame.draw.circle(s, (60, 60, 60), (bx, by), BALL_RADIUS // 2, 1)

    def _draw_hud(self):
        s = self.screen

        # Placar
        score_txt = self.font_md.render(f"⚽  {self.score}", True, (255, 255, 255))
        s.blit(score_txt, (WIDTH // 2 - score_txt.get_width() // 2, 8))

        # Controles
        ctrl = self.font_xs.render(
            "WASD / ↑↓←→ : Mover   |   ESPAÇO : Chutar   |   ESC : Sair",
            True, (200, 255, 200)
        )
        s.blit(ctrl, (WIDTH // 2 - ctrl.get_width() // 2, HEIGHT - 28))

    def _draw_message(self):
        alpha = min(255, self.msg_timer * 4)
        surf = self.font_lg.render(self.message, True, (255, 230, 0))
        surf.set_alpha(alpha)
        self.screen.blit(surf, surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60)))

    def _show_message(self, text, duration):
        self.message   = text
        self.msg_timer = duration

    def _load_fonts(self):
        self.font_xs = pygame.font.SysFont("monospace", 13)
        self.font_sm = pygame.font.SysFont("monospace", 11, bold=True)
        self.font_md = pygame.font.SysFont("monospace", 26, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 48, bold=True)

    # ── Utilitários ──────────────────────────────────────────────────────────

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ── Loop humano ──────────────────────────────────────────────────────────

    def run_human(self):
        """Loop principal para controle humano."""
        obs = self.reset()

        while True:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    if event.key == pygame.K_r:
                        obs = self.reset()

            # Lê teclas pressionadas
            keys = pygame.key.get_pressed()
            action = ACTION_NONE

            if keys[pygame.K_SPACE]:
                action = ACTION_KICK
            elif keys[pygame.K_w] or keys[pygame.K_UP]:
                action = ACTION_UP
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                action = ACTION_DOWN
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                action = ACTION_LEFT
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action = ACTION_RIGHT

            obs, reward, done, info = self.step(action)
            self.render()
            self.clock.tick(FPS)

            if done:
                obs = self.reset()


# ─── Entrada ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = FootballEnv(render_mode=True)
    env.run_human()
