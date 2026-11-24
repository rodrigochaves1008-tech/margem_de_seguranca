"""Dados dos dois gráficos de 10 anos (preço e preço justo ao longo do tempo)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import precos as P


def ajuste_pos_analise(V: float, data_analise, ate, diario: pd.DataFrame, desd: pd.DataFrame,
                       prov: pd.DataFrame) -> tuple[float, bool]:
    """Preço justo da análise levado para a base de `ate`: desdobramentos/grupamentos e proventos
    com data ex depois da análise, com os mesmos fatores do histórico de preços."""
    a, ate = pd.Timestamp(data_analise).normalize(), pd.Timestamp(ate).normalize()
    fator = 1.0
    houve = False
    if prov is not None and not prov.empty:
        for d, v in zip(prov["data"], prov["valor"]):
            if a < d <= ate:
                ant = diario[diario["data"] < d]
                if len(ant) and ant["fechamento"].iloc[-1] > 0:
                    fator *= 1.0 - v / float(ant["fechamento"].iloc[-1])
                    houve = True
    f_desd = P.fator_entre(desd, a, ate)
    if f_desd != 1.0:
        houve = True
    return V * fator / f_desd, houve


def montar(diario: pd.DataFrame, desd: pd.DataFrame, prov: pd.DataFrame, mensal: pd.DataFrame | None,
           analise: dict, historico_ativo: list[dict], ate, tracejado: bool, eventos: list[dict],
           mostrar_sem_confiavel: bool = False) -> dict:
    ate = pd.Timestamp(ate).normalize()
    base = ate
    d = diario[diario["data"] <= ate]
    ini = ate - pd.DateOffset(years=10)
    d = d[d["data"] > ini]
    aj = P.serie_ajustada(d, desd, base)
    rt = P.retorno_total(diario[diario["data"] <= ate], desd, prov, base)
    rt = rt[rt.index > ini]
    if len(rt) and len(aj):
        rt = rt * (aj.iloc[0] / rt.iloc[0])
    out = {
        "preco": [[x.strftime("%Y-%m-%d"), round(float(v), 4)] for x, v in aj.items()],
        "retorno": [[x.strftime("%Y-%m-%d"), round(float(v), 4)] for x, v in rt.items()],
        "desde": d["data"].min().strftime("%Y-%m-%d") if len(d) else None,
        "confiavel": bool(analise.get("estimativa")),
        # Decisão 10 do usuário: o preço justo do modelo aparece mesmo sem estimativa confiável, com aviso.
        "mostrar_justo": bool(analise.get("estimativa")) or (mostrar_sem_confiavel and bool(analise.get("preco_justo_modelo"))),
        "eventos": numerar_marcas([e for e in eventos if ini < pd.Timestamp(e["data"]) <= ate]),
        "justo": [], "historico": [], "tracejado": [],
    }
    if not out["mostrar_justo"]:
        return out
    if mensal is not None and len(mensal):
        m = mensal[(mensal.index > ini) & (mensal.index <= ate) & mensal["V"].notna()]
        for s, r in m.iterrows():
            f = P.fator_entre(desd, s, base)
            out["justo"].append([s.strftime("%Y-%m-%d"), r["V"] / f,
                                 None if pd.isna(r.get("P10")) else r["P10"] / f,
                                 None if pd.isna(r.get("P90")) else r["P90"] / f, r["preco"] / f])
    da = pd.Timestamp(analise["data_analise"]).normalize()
    fa = P.fator_entre(desd, da, base)
    if analise.get("estimativa"):
        V, p10, p90 = analise["preco_justo"], analise.get("p10"), analise.get("p90")
    else:
        V, p10, p90 = analise["preco_justo_modelo"], analise.get("p10_modelo"), analise.get("p90_modelo")
    pa = analise.get("preco")
    out["justo"].append([da.strftime("%Y-%m-%d"), V / fa, None if p10 is None else p10 / fa,
                         None if p90 is None else p90 / fa, None if pa is None else pa / fa])
    out["ultima"] = {"data": da.strftime("%Y-%m-%d"), "v": V / fa}
    if tracejado and ate > da:
        pontos = [da] + sorted({pd.Timestamp(x) for x in list(prov["data"]) + list(desd["data"])
                                if da < pd.Timestamp(x) <= ate}) + [ate]
        for x in pontos:
            vx, _ = ajuste_pos_analise(V, da, x, diario, desd, prov)
            r = vx / V if V else np.nan
            fx = P.fator_entre(desd, x, base)
            out["tracejado"].append([x.strftime("%Y-%m-%d"), vx / fx,
                                     None if p10 is None else p10 * r / fx, None if p90 is None else p90 * r / fx])
    for h in historico_ativo:
        dh = pd.Timestamp(str(h["data_analise"])[:10])
        if dh >= da or not h.get("estimativa") or h.get("preco_justo") is None or dh <= ini:
            continue
        out["historico"].append([dh.strftime("%Y-%m-%d"), h["preco_justo"] / P.fator_entre(desd, dh, base)])
    return out


def numerar_marcas(eventos: list[dict], dias_grupo: int = 75) -> list[dict]:
    """Numera as marcas em ordem de data. No gráfico vai só o número; marcas próximas (até `dias_grupo` dias
    da primeira do grupo) dividem um rótulo, como "6–10", para os textos não se sobreporem. A lista
    numerada fica embaixo do gráfico."""
    ev = sorted((dict(e) for e in eventos), key=lambda e: (e["data"], e.get("tipo") != "quebra", e.get("rotulo", "")))
    grupos: list[list[dict]] = []
    for i, e in enumerate(ev):
        e["n"] = i + 1
        if grupos and (pd.Timestamp(e["data"]) - pd.Timestamp(grupos[-1][0]["data"])).days <= dias_grupo:
            grupos[-1].append(e)
        else:
            grupos.append([e])
    for g in grupos:
        for e in g:
            e["marca"] = ""
        g[0]["marca"] = str(g[0]["n"]) if len(g) == 1 else f"{g[0]['n']}–{g[-1]['n']}"
    return ev


def eventos_de_preco(desd: pd.DataFrame) -> list[dict]:
    ev = []
    for d, f in zip(desd["data"], desd["fator"]):
        if f >= 1:
            rot = (f"desdobramento {f:g}:1" if float(f).is_integer() else f"bonificação (fator {f:g})").replace(".", ",")
        else:
            rot = f"grupamento 1:{1 / f:g}".replace(".", ",")
        ev.append({"data": pd.Timestamp(d).strftime("%Y-%m-%d"), "rotulo": rot, "tipo": "evento"})
    return ev
