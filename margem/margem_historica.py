"""Ranking da margem de segurança dos últimos anos e a posição da margem de hoje.

A margem de cada mês vem da estimativa walk-forward daquele mês (o que o modelo teria estimado só com os
dados até a data). É um retrato do próprio modelo: se ele não é confiável para o ativo, a posição de hoje
também não é.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import formatos
from . import precos as P

NOTA = ("Margem de cada mês = (preço justo walk-forward − preço) ÷ preço justo, com o que o modelo teria estimado "
        "naquela data só com os dados divulgados até ela. Preço e preço justo na base de ações de hoje. É um retrato do "
        "próprio modelo: se ele não é confiável, a posição de hoje também não é. Estimativa de modelo, não é "
        "recomendação de investimento.")


def ranking(mensal: pd.DataFrame | None, desd: pd.DataFrame, V_hoje, preco_hoje, data_preco, anos: int = 10,
            top: int = 10) -> dict:
    """Meses dos últimos `anos` ordenados pela margem, com a margem de hoje na sua posição."""
    try:
        V_hoje, preco_hoje = float(V_hoje), float(preco_hoje)
    except (TypeError, ValueError):
        return {"ok": False, "motivo": "sem preço justo hoje"}
    if mensal is None or not len(mensal) or not np.isfinite(V_hoje) or V_hoje <= 0 or not np.isfinite(preco_hoje):
        return {"ok": False, "motivo": "sem estimativas"}
    base = pd.Timestamp(data_preco)
    ini = base - pd.DateOffset(years=anos)
    m = mensal[(mensal.index > ini) & (mensal.index <= base) & mensal["V"].notna() & (mensal["V"] > 0)].copy()
    if len(m) < 12:
        return {"ok": False, "motivo": "menos de 12 meses de estimativas"}
    m["margem"] = (m["V"] - m["preco"]) / m["V"]
    margem_hoje = (V_hoje - preco_hoje) / V_hoje
    itens = [{"data": d, "rotulo": formatos.mes(d), "margem": float(r["margem"]),
              "preco": float(r["preco"]) / P.fator_entre(desd, d, base), "V": float(r["V"]) / P.fator_entre(desd, d, base),
              "hoje": False} for d, r in m.iterrows()]
    itens.append({"data": base, "rotulo": f"hoje ({formatos.data(base)})", "margem": margem_hoje, "preco": preco_hoje,
                  "V": V_hoje, "hoje": True})
    itens.sort(key=lambda x: (-x["margem"], x["data"]))
    for i, x in enumerate(itens):
        x["posicao"] = i + 1
    linhas = itens[:top]
    hoje = next(x for x in itens if x["hoje"])
    if hoje not in linhas:
        linhas = linhas + [hoje]
    q = m["margem"].quantile([0.5, 0.75, 0.9])
    return {"ok": True, "anos": anos, "n": len(m), "inicio": m.index[0], "fim": m.index[-1], "total": len(itens),
            "margem_hoje": margem_hoje, "posicao": hoje["posicao"], "percentil": float((m["margem"] < margem_hoje).mean()),
            "max": float(m["margem"].max()), "data_max": m["margem"].idxmax(), "mediana": float(q.loc[0.5]),
            "p75": float(q.loc[0.75]), "p90": float(q.loc[0.9]), "linhas": linhas}


def texto(rk: dict | None) -> str:
    if not rk or not rk.get("ok"):
        return ""
    p = rk["percentil"]
    if p >= 0.9:
        nivel = "perto do topo do histórico (entre os 10% maiores)"
    elif p >= 0.75:
        nivel = "acima da maioria dos meses, mas fora dos 10% maiores"
    elif p >= 0.5:
        nivel = "acima da mediana do histórico"
    else:
        nivel = "abaixo da mediana do histórico"
    return (f"Margem de hoje: {formatos.pct(rk['margem_hoje'], 1, True)}, a {rk['posicao']}ª maior em {rk['total']} "
            f"(os {rk['n']} meses de {formatos.mes(rk['inicio'])} a {formatos.mes(rk['fim'])} e hoje), maior que a de "
            f"{formatos.pct(p, 0)} dos meses: {nivel}. Topo: {formatos.pct(rk['max'], 1, True)} em "
            f"{formatos.mes(rk['data_max'])}; mediana: {formatos.pct(rk['mediana'], 1, True)}; 90% dos meses abaixo de "
            f"{formatos.pct(rk['p90'], 1, True)}.")
