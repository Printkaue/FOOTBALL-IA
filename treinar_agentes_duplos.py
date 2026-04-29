import pygame
from football_env import FootballEnv, N_ACTIONS
from algoritimoGenetico import AlgoritmoGenetico
from viwer import *
from rede_neural import RedeNeural
from settings import *
from agloritimo_genetico_vs import AlgoritmoGeneticoVS

def treinar():
    ag  = AlgoritmoGeneticoVS()
    vis = Visualizador()

    print("=" * 55)
    print("  ⚽  FOOTBALL AI — Treinamento Genético")
    print("=" * 55)
    print(f"  População:     {POPULACAO} agentes")
    print(f"  Rede neural:   {N_INPUTS} → {NEURONIOS_OCULTO} → {N_ACTIONS}")
    print(f"  Passos/ep:     {PASSOS_EPISODIO}")
    print(f"  Mutação:       {TAXA_MUTACAO*100:.0f}%  força {FORCA_MUTACAO}")
    print(f"  Elite:         {ELITE} sobrevivem direto")
    print("=" * 55)

    while True:
        print(f"\n[Geração {ag.geracao}] Avaliando {POPULACAO} agentes...")

        # ── Fase de avaliação (sem janela, rápido) ──
        agente_avaliado = [0]

        def atualizar_progresso(i):
            agente_avaliado[0] = i + 1

            # Desenha a tela de progresso enquanto avalia
            vis.processar_eventos()
            vis.tela.fill((10, 10, 20))

            melhor = max(ag.fitness) if any(f > 0 for f in ag.fitness) else 0
            media  = sum(ag.fitness) / POPULACAO

            # Barra de progresso
            bw = 600
            bx = (LARGURA_TOTAL - bw) // 2
            by = ALTURA_TOTAL // 2
            pct = (i + 1) / POPULACAO
            pygame.draw.rect(vis.tela, (30, 30, 50), (bx, by, bw, 28), border_radius=6)
            pygame.draw.rect(vis.tela, COR_DESTAQUE, (bx, by, int(bw * pct), 28), border_radius=6)
            pygame.draw.rect(vis.tela, (80, 80, 120), (bx, by, bw, 28), 2, border_radius=6)

            t1 = vis.font_gg.render(f"Geração {ag.geracao} — Avaliando agentes", True, COR_TEXTO)
            t2 = vis.font_m.render(f"Agente {i+1} / {POPULACAO}  ({int(pct*100)}%)", True, COR_ALERTA)
            t3 = vis.font_p.render(f"Melhor fitness até agora: {melhor:.1f}", True, COR_DESTAQUE)

            vis.tela.blit(t1, (LARGURA_TOTAL//2 - t1.get_width()//2, by - 50))
            vis.tela.blit(t2, (LARGURA_TOTAL//2 - t2.get_width()//2, by + 36))
            vis.tela.blit(t3, (LARGURA_TOTAL//2 - t3.get_width()//2, by + 58))

            pygame.display.flip()

        ag.avaliar_todos(callback_progresso=atualizar_progresso)

        # ── Estatísticas ──
        ordem       = sorted(range(POPULACAO), key=lambda i: ag.fitness[i], reverse=True)
        melhor_fit  = ag.fitness[ordem[0]]
        media_fit   = sum(ag.fitness) / POPULACAO
        gols_melhor = int(melhor_fit // 20)   # estimativa

        print(f"  Melhor fitness : {melhor_fit:.2f}")
        print(f"  Fitness médio  : {media_fit:.2f}")

        melhor_atk = ag.melhor_atacante()
        melhor_def = ag.melhor_defensor()

        # ── Mostra o melhor agente jogando ──
        print(f"  Mostrando melhor agente por {PASSOS_EPISODIO} passos...")
        ag.historico_fitness.append(melhor_fit)
        ag.historico_media.append(media_fit)

        vis.env_vis.render_mode = True
        vis.env_vis.screen      = vis.surf_campo
        if not hasattr(vis.env_vis, 'font_xs'):
            vis.env_vis._load_fonts()

        obs = vis.env_vis.reset()
        for passo in range(PASSOS_EPISODIO):
            vis.processar_eventos()

            acao = melhor_atk.pensar(obs)
            acao2 = melhor_def.pensar(obs)
            obs, _, done, info = vis.env_vis.step(acao)

            vis.env_vis._draw_field()
            vis.env_vis._draw_goal()
            vis.env_vis._draw_ball()
            vis.env_vis._draw_player()
            vis.env_vis._draw_hud()
            if vis.env_vis.msg_timer > 0:
                vis.env_vis._draw_message()

            overlay = vis.font_p.render(
                f"Geração {ag.geracao}  |  Fitness: {melhor_fit:.1f}  |  Gols: {info['score']}  |  [Melhor agente ao vivo]",
                True, (255, 255, 200)
            )
            vis.surf_campo.blit(overlay, (10, 10))

            vis.tela.blit(vis.surf_campo, (0, 0))
            vis.desenhar_painel(
                melhor_atk,
                ag.historico_fitness,
                ag.historico_media,
                ag.geracao,
                melhor_fit,
                media_fit,
                0, 0,
                obs
            )

            pygame.display.flip()
            vis.clock.tick(FPS * VELOCIDADE_SIM)

            if done:
                obs = vis.env_vis.reset()

        vis.env_vis.render_mode = False

        # ── Evolui para a próxima geração ──
        ag.nova_geracao()



if __name__ == "__main__":
    treinar()
