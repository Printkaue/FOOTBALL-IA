import pygame
import sys
import numpy as np
from football_env import FootballEnv
from settings import N_INPUTS, NEURONIOS_OCULTO, N_ACTIONS, VELOCIDADE_SIM, FPS, PASSOS_EPISODIO

# Tamanho da janela principal
LARGURA_CAMPO  = 900
ALTURA_CAMPO   = 600
LARGURA_PAINEL = 380
LARGURA_TOTAL  = LARGURA_CAMPO + LARGURA_PAINEL
ALTURA_TOTAL   = ALTURA_CAMPO

# Cores do painel
COR_PAINEL    = (15, 15, 25)
COR_TEXTO     = (200, 220, 255)
COR_DESTAQUE  = (80, 200, 120)
COR_ALERTA    = (255, 180, 50)
COR_GRAFICO   = (50, 120, 220)
COR_MEDIA     = (180, 80, 80)
COR_GRID      = (30, 30, 50)

# Cores da rede neural
COR_NEURONIO_POS = (50, 200, 100)
COR_NEURONIO_NEG = (200, 60, 60)
COR_NEURONIO_NEU = (80, 80, 120)
COR_PESO_POS     = (50, 180, 80)
COR_PESO_NEG     = (180, 50, 50)

NOMES_ACOES = ["Nada", "Cima", "Baixo", "Esq", "Dir", "Chutar"]

class Visualizador:

    def __init__(self):
        pygame.init()
        self.tela = pygame.display.set_mode((LARGURA_TOTAL, ALTURA_TOTAL))
        pygame.display.set_caption("⚽ QBALL")
        self.clock = pygame.time.Clock()

        self.font_p  = pygame.font.SysFont("monospace", 12)
        self.font_m  = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_g  = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_gg = pygame.font.SysFont("monospace", 22, bold=True)

        # Superfície do campo (lado esquerdo)
        self.surf_campo = pygame.Surface((LARGURA_CAMPO, ALTURA_CAMPO))

        # Ambiente para visualização
        self.env_vis = FootballEnv(render_mode=False)

    def processar_eventos(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

    # ── Painel direito ───────────────────────────────────────────────────────

    def desenhar_painel(self, ag, historico_max, historico_med, geracao,
                        melhor_fit, med_fit, agente_atual, total_agentes,
                        obs_atual):

        px = LARGURA_CAMPO   # x onde começa o painel
        s  = self.tela

        # Fundo do painel
        pygame.draw.rect(s, COR_PAINEL, (px, 0, LARGURA_PAINEL, ALTURA_TOTAL))
        pygame.draw.line(s, (40, 40, 70), (px, 0), (px, ALTURA_TOTAL), 2)

        y = 12

        # ── Título ──
        t = self.font_gg.render("Football AI", True, COR_DESTAQUE)
        s.blit(t, (px + LARGURA_PAINEL//2 - t.get_width()//2, y));  y += 32

        # ── Info da geração ──
        self._linha(s, px, y, "GERAÇÃO",     str(geracao),    COR_ALERTA);   y += 20
        self._linha(s, px, y, "Melhor fit.", f"{melhor_fit:.1f}", COR_DESTAQUE); y += 20
        self._linha(s, px, y, "Fit. médio",  f"{med_fit:.1f}", COR_TEXTO);    y += 20

        if total_agentes > 0:
            pct = int(agente_atual / total_agentes * 100)
            self._linha(s, px, y, "Avaliando", f"{agente_atual}/{total_agentes} ({pct}%)", COR_TEXTO)
        else:
            self._linha(s, px, y, "Status", "Melhor ao vivo", COR_DESTAQUE)
        y += 28

        # ── Rede Neural ──
        y = self._desenhar_rede(s, px, y, obs_atual, ag)
        y += 10

        # ── Gráfico de fitness ──
        y = self._desenhar_grafico(s, px, y, historico_max, historico_med)

        # ── Legenda ──
        y += 8
        self._linha(s, px, y, "ESC", "Sair", (120, 120, 160))

    def _linha(self, s, px, y, label, valor, cor_valor):
        t1 = self.font_p.render(f"{label}:", True, (120, 130, 160))
        t2 = self.font_m.render(valor, True, cor_valor)
        s.blit(t1, (px + 12, y))
        s.blit(t2, (px + LARGURA_PAINEL - t2.get_width() - 12, y))

    def _desenhar_rede(self, s, px, y_inicio, obs, rede):
        """
        Desenha a rede neural do melhor agente.
        Cada círculo é um neurônio. A cor mostra se está ativo (+) ou não (-).
        As linhas mostram os pesos (verde=positivo, vermelho=negativo).
        """
        t = self.font_m.render("REDE NEURAL", True, COR_TEXTO)
        s.blit(t, (px + LARGURA_PAINEL//2 - t.get_width()//2, y_inicio))
        y_inicio += 20

        if obs is None:
            return y_inicio + 100

        entrada, oculta, saida = rede.get_ativacoes(obs)
        saida_norm = saida - saida.min()
        if saida_norm.max() > 0:
            saida_norm /= saida_norm.max()

        # Posições das camadas
        cx = px + LARGURA_PAINEL // 2
        alturas = {
            "entrada": y_inicio + 20,
            "oculta":  y_inicio + 80,
            "saida":   y_inicio + 145,
        }

        # Raio dos neurônios
        r_e = 7    # entrada
        r_o = 8    # oculta
        r_s = 10   # saída

        # ── Calcula posições X ──
        def posicoes_x(n, largura=300):
            if n == 1: return [cx]
            espacamento = largura // (n - 1)
            inicio = cx - largura // 2
            return [inicio + i * espacamento for i in range(n)]

        pos_entrada = posicoes_x(N_INPUTS, 310)
        pos_oculta  = posicoes_x(NEURONIOS_OCULTO, 330)
        pos_saida   = posicoes_x(N_ACTIONS, 300)

        y_e = alturas["entrada"]
        y_o = alturas["oculta"]
        y_s = alturas["saida"]

        # ── Conexões entrada→oculta (amostra para não poluir) ──
        for j in range(NEURONIOS_OCULTO):
            for i in range(0, N_INPUTS, 2):   # pula de 2 em 2 para simplificar
                w = rede.W1[i, j]
                alfa = min(200, int(abs(w) * 80))
                cor = (0, alfa, 0) if w > 0 else (alfa, 0, 0)
                pygame.draw.line(s, cor, (pos_entrada[i], y_e), (pos_oculta[j], y_o), 1)

        # ── Conexões oculta→saída ──
        for i in range(NEURONIOS_OCULTO):
            for j in range(N_ACTIONS):
                w = rede.W2[i, j]
                alfa = min(200, int(abs(w) * 80))
                cor = (0, alfa, 0) if w > 0 else (alfa, 0, 0)
                pygame.draw.line(s, cor, (pos_oculta[i], y_o), (pos_saida[j], y_s), 1)

        # ── Neurônios da entrada ──
        nomes_entrada = ["Jog.X","Jog.Y","Bol.X","Bol.Y","Vel.X","Vel.Y","Gol.X","Gol.Y"]
        for i, (x, val) in enumerate(zip(pos_entrada, entrada)):
            v = int((val + 1) / 2 * 255)
            cor = (v, v, min(255, v + 50))
            pygame.draw.circle(s, cor, (x, y_e), r_e)
            pygame.draw.circle(s, (150,150,200), (x, y_e), r_e, 1)

        # ── Neurônios ocultos ──
        for i, (x, val) in enumerate(zip(pos_oculta, oculta)):
            v = int((val + 1) / 2 * 255)
            cor = COR_NEURONIO_POS if val > 0 else COR_NEURONIO_NEG
            brilho = int(abs(val) * 200)
            cor = tuple(min(255, c * brilho // 200) for c in cor)
            pygame.draw.circle(s, cor, (x, y_o), r_o)
            pygame.draw.circle(s, (150,150,200), (x, y_o), r_o, 1)

        # ── Neurônios de saída (ações) ──
        acao_escolhida = int(np.argmax(saida))
        for i, (x, val) in enumerate(zip(pos_saida, saida_norm)):
            v = int(val * 220)
            cor = (v, v, 50) if i == acao_escolhida else (50, 50, int(val*180))
            pygame.draw.circle(s, cor, (x, y_s), r_s)
            borda = COR_ALERTA if i == acao_escolhida else (100,100,150)
            pygame.draw.circle(s, borda, (x, y_s), r_s, 2)
            label = self.font_p.render(NOMES_ACOES[i], True,
                                       COR_ALERTA if i == acao_escolhida else (100,100,150))
            s.blit(label, (x - label.get_width()//2, y_s + r_s + 2))

        # ── Labels das camadas ──
        for texto, y_pos in [("Entradas", y_e - 16), ("Oculta", y_o - 16), ("Ações", y_s - 16)]:
            t = self.font_p.render(texto, True, (80, 90, 130))
            s.blit(t, (px + 10, y_pos))

        return y_s + r_s + 22

    def _desenhar_grafico(self, s, px, y_inicio, hist_max, hist_med):
        """Gráfico simples de linha com o fitness ao longo das gerações."""
        t = self.font_m.render("FITNESS POR GERAÇÃO", True, COR_TEXTO)
        s.blit(t, (px + LARGURA_PAINEL//2 - t.get_width()//2, y_inicio))
        y_inicio += 18

        gw, gh = 340, 110
        gx, gy = px + 20, y_inicio

        # Fundo do gráfico
        pygame.draw.rect(s, (20, 20, 35), (gx, gy, gw, gh))
        pygame.draw.rect(s, (50, 50, 80), (gx, gy, gw, gh), 1)

        if len(hist_max) < 2:
            msg = self.font_p.render("Aguardando dados...", True, (80,80,120))
            s.blit(msg, (gx + gw//2 - msg.get_width()//2, gy + gh//2 - 8))
            return y_inicio + gh + 30

        # Grid horizontal
        for i in range(1, 4):
            yg = gy + i * gh // 4
            pygame.draw.line(s, COR_GRID, (gx, yg), (gx + gw, yg), 1)

        todos = hist_max + hist_med
        vmin, vmax = min(todos), max(todos)
        if vmax == vmin: vmax = vmin + 1

        def to_px(val, idx, total):
            x = gx + int(idx / max(total - 1, 1) * gw)
            y = gy + gh - int((val - vmin) / (vmax - vmin) * (gh - 4)) - 2
            return x, y

        # Linha do fitness máximo
        pts_max = [to_px(v, i, len(hist_max)) for i, v in enumerate(hist_max)]
        if len(pts_max) >= 2:
            pygame.draw.lines(s, COR_GRAFICO, False, pts_max, 2)

        # Linha do fitness médio
        pts_med = [to_px(v, i, len(hist_med)) for i, v in enumerate(hist_med)]
        if len(pts_med) >= 2:
            pygame.draw.lines(s, COR_MEDIA, False, pts_med, 1)

        # Ponto atual
        pygame.draw.circle(s, COR_ALERTA, pts_max[-1], 4)

        # Eixo Y: min e max
        t_max = self.font_p.render(f"{vmax:.0f}", True, (100,100,160))
        t_min = self.font_p.render(f"{vmin:.0f}", True, (100,100,160))
        s.blit(t_max, (gx - t_max.get_width() - 2, gy))
        s.blit(t_min, (gx - t_min.get_width() - 2, gy + gh - 10))

        # Legenda
        y_leg = gy + gh + 6
        pygame.draw.line(s, COR_GRAFICO, (gx, y_leg+5), (gx+16, y_leg+5), 2)
        s.blit(self.font_p.render("Melhor", True, COR_GRAFICO), (gx+18, y_leg))
        pygame.draw.line(s, COR_MEDIA, (gx+75, y_leg+5), (gx+91, y_leg+5), 1)
        s.blit(self.font_p.render("Média",  True, COR_MEDIA),   (gx+93, y_leg))

        return y_inicio + gh + 28

    # ── Campo com o melhor agente ────────────────────────────────────────────

    def mostrar_melhor_jogando(self, rede, geracao, melhor_fit):
        """Roda o melhor agente no campo por PASSOS_EPISODIO frames."""
        obs = self.env_vis.reset()

        # Cria uma cópia do env para renderizar no nosso surf_campo
        env = self.env_vis
        env.render_mode  = True
        env.screen       = self.surf_campo
        env._load_fonts() if not hasattr(env, 'font_xs') else None

        for passo in range(PASSOS_EPISODIO):
            self.processar_eventos()

            acao = rede.pensar(obs)
            obs, _, done, info = env.step(acao)

            # Desenha campo na superfície parcial
            env._draw_field()
            env._draw_goal()
            env._draw_ball()
            env._draw_player()
            env._draw_hud()
            if env.msg_timer > 0:
                env._draw_message()

            # Overlay de info
            t = env.font_xs.render(
                f"Geração {geracao}  |  Melhor fitness: {melhor_fit:.1f}  |  Gols: {info['score']}",
                True, (255, 255, 200)
            )
            self.surf_campo.blit(t, (10, 10))

            # Cola o campo e o painel na tela
            self.tela.blit(self.surf_campo, (0, 0))
            self.desenhar_painel(
                rede,
                [],      # histórico vazio durante visualização
                [],
                geracao,
                melhor_fit,
                melhor_fit,
                0, 0,
                obs
            )

            pygame.display.flip()
            self.clock.tick(FPS * VELOCIDADE_SIM)

            if done:
                obs = env.reset()

        env.render_mode = False

