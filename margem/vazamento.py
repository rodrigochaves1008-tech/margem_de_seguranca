"""Teste de vazamento: para meses sorteados, troca por valores aleatórios tudo o que foi divulgado
depois da data e refaz do zero o processo inteiro (seleção de variáveis, normalização, escolha do
modelo). Na mesma base de ajuste, a estimativa daquela data tem de ser igual, ao centavo, à guardada
no backtest (a que alimenta o gráfico)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import caminhos, config, painel
from .modelo import walkforward as WF


def testar(ticker: str, meses: int = 3, semente: int = 2026, progresso=print) -> list[dict]:
    ticker = ticker.upper()
    ur = caminhos.pasta_ultima_rodada(ticker)
    wf_path = ur / "walkforward.parquet"
    if not wf_path.exists():
        raise RuntimeError(f"{ticker} ainda não foi rodado; não há backtest guardado para comparar.")
    guardado = pd.read_parquet(wf_path)
    info = config.info_ativo(ticker)
    setor = config.info_setor(info["setor"])
    cfg = config.cfg()
    progresso(f"{ticker}: carregando dados para o teste de vazamento")
    dados = painel.carregar(ticker)
    candidatas = [v for fam in painel.preditores_do_setor(setor).values() for v in fam]
    rng = np.random.default_rng(semente)
    elegiveis = [d for d in guardado.index if np.isfinite(guardado.at[d, "V"])]
    sorteados = sorted(rng.choice(elegiveis, size=min(meses, len(elegiveis)), replace=False))
    resultados = []
    for i, t in enumerate(sorteados):
        t = pd.Timestamp(t)
        progresso(f"  {i + 1}/{len(sorteados)}: {t:%m/%Y} (dados depois da data trocados por valores aleatórios)")
        corr = painel.corromper_depois(dados, t, semente + i + 1)
        base = painel.montar(corr, painel.datas_da_base(corr, t))
        res = WF.rodar(base, setor["alvos"], candidatas, cfg, corr.cvm["documentos"]["divulgacao"],
                       corr.precos.desdobramentos, t, None, None, progresso=None)
        refeito = float(res.mensal.at[t, "V"]) if t in res.mensal.index else np.nan
        orig = float(guardado.at[t, "V"])
        resultados.append({"mes": t, "guardado": orig, "refeito": refeito, "diferenca": refeito - orig,
                           "passou": bool(np.isfinite(refeito) and round(refeito, 2) == round(orig, 2)),
                           "candidato_guardado": guardado.at[t, "escolhido"],
                           "candidato_refeito": res.mensal.at[t, "escolhido"] if t in res.mensal.index else None})
    return resultados
