"""Teste fora da amostra de regras de escolha e critérios de estimativa confiável.

Para cada ativo, usa as previsões de todos os candidatos guardadas no walk-forward (elas não
dependem da regra de escolha) e refaz, com o mesmo código de produção, a escolha em cada retreino
e o critério de estimativa confiável de cada variante. Não treina nada de novo.

As variantes ficam definidas antes de rodar (ver VARIANTES), para não ajustar o critério aos
ativos já vistos.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import avaliacao, config
from .modelo import walkforward as WF

VARIANTES = {
    "A": {"nome": "atual: escolha por 6 e 12 meses; erro ≤ 100% do passeio em 12 meses e ≤ 110% em 6",
          "escolha": [6, 12], "limites": {12: 1.00, 6: 1.10}},
    "B": {"nome": "alternativo: escolha por 12 e 24 meses; erro ≤ 100% do passeio em 24 meses e ≤ 110% em 12",
          "escolha": [12, 24], "limites": {24: 1.00, 12: 1.10}},
}


def _fim_de_mes(d) -> bool:
    d = pd.Timestamp(d)
    return d == (d + pd.offsets.MonthEnd(0)).normalize()


def reconstruir(mensal: pd.DataFrame, retreinos: list[pd.Timestamp], preco_mensal: pd.Series,
                desd: pd.DataFrame, cfg_escolha: dict, minimo_faixa: int):
    """Série publicada com a regra de escolha `cfg_escolha`, a partir das previsões de cada candidato."""
    cands = [c[3:] for c in mensal.columns if c.startswith("V__")]
    previsoes = mensal[[f"V__{c}" for c in cands]].copy()
    previsoes.columns = cands
    escolhido_em = {}
    for r in retreinos:
        c, _ = WF.escolher(r, previsoes[previsoes.index < r], preco_mensal, desd, cfg_escolha, cands)
        escolhido_em[r] = c
    out = mensal.copy()
    for m in out.index:
        rs = [r for r in retreinos if r <= m]
        if not rs:
            continue
        c = escolhido_em[rs[-1]]
        out.at[m, "escolhido"] = c
        out.at[m, "V"] = out.at[m, f"V__{c}"]
    out = WF._completar_faixa(out.drop(columns=["q10", "q90", "P10", "P90"], errors="ignore"), minimo_faixa)
    return out, escolhido_em, previsoes, cands


def avaliar_ativo(pasta_saida: Path, ticker: str) -> dict:
    ur = pasta_saida / "ultima_rodada" / ticker
    a = json.loads((ur / "analise.json").read_text(encoding="utf-8"))
    mensal = pd.read_parquet(ur / "walkforward.parquet")
    escolhas = pd.read_parquet(ur / "escolhas.parquet")
    desd = pd.read_parquet(ur / "desdobramentos.parquet")
    hoje_c = json.loads((ur / "hoje_candidatos.json").read_text(encoding="utf-8")) if (ur / "hoje_candidatos.json").exists() else None
    rel = pasta_saida / Path(a["relatorio"]).parent / "dados"
    base = pd.read_parquet(rel / "base_mensal.parquet")
    hoje = pd.Timestamp(a["data_analise"][:10])
    meses = [d for d in base.index if _fim_de_mes(d)]
    preco_mensal = base.loc[meses, "preco"]
    info = config.info_ativo(ticker)
    setor = config.info_setor(info["setor"])
    retreinos = sorted(pd.to_datetime(escolhas["data"]))
    cfg0 = config.cfg()
    saida = {"ticker": ticker, "setor": setor.get("nome")}
    for k, v in VARIANTES.items():
        cfg = copy.deepcopy(cfg0)
        cfg["modelo"]["escolha"]["horizontes_meses"] = v["escolha"]
        cfg["estimativa_confiavel"]["limites_contra_passeio"] = v["limites"]
        serie, escolhido_em, previsoes, cands = reconstruir(mensal, retreinos, preco_mensal, desd, cfg["modelo"]["escolha"],
                                                            int(cfg["estimativa_confiavel"]["meses_faixa_min"]))
        if k == "A":
            iguais = np.allclose(serie["V"].to_numpy(float), mensal["V"].to_numpy(float), equal_nan=True, rtol=0, atol=1e-9)
            saida["reconstrucao_confere"] = bool(iguais)
        # Último passo (hoje): a escolha usa os meses fechados até hoje, como no walk-forward.
        hj = {"data": hoje, "preco": a["preco"], "variacao_sementes": 0.0}
        if hoje_c and hoje_c.get("V_candidatos"):
            if pd.Timestamp(hoje_c["retreino"][:10]) == hoje:
                c_hoje, _ = WF.escolher(hoje, previsoes, preco_mensal, desd, cfg["modelo"]["escolha"], cands)
            else:
                c_hoje = escolhido_em[retreinos[-1]]
            hj["V"] = float(hoje_c["V_candidatos"].get(c_hoje, np.nan))
            hj["escolhido"] = c_hoje
        else:
            hj["V"] = float(serie["V"].dropna().iloc[-1])
        razoes = (serie["preco"] / serie["V"]).to_numpy()
        q10, q90 = WF.faixa(razoes, int(cfg["estimativa_confiavel"]["meses_faixa_min"]))
        hj["q10"], hj["q90"] = q10, q90
        aval = avaliacao.avaliar(serie, hj, base.loc[hoje], base, preco_mensal, desd, cfg, setor.get("obrigatorios", []))
        saida[k] = {
            "confiavel": aval["confiavel"], "motivos": aval["motivos"], "erro_tipico": aval["erro_tipico"],
            "cobertura_faixa": aval["cobertura_faixa"],
            "razoes": {h: aval["contra_passeio"].get(h, {}).get("razao") for h in v["limites"]},
            "n": {h: aval["contra_passeio"].get(h, {}).get("n") for h in v["limites"]},
            "V": hj["V"], "preco": a["preco"], "escolhido_hoje": hj.get("escolhido"),
            "margem": (hj["V"] - a["preco"]) / hj["V"] if hj["V"] and np.isfinite(hj["V"]) else None,
        }
    return saida
