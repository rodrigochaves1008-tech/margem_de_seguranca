"""Perspectivas públicas dos analistas (preços-alvo) e estudo das épocas de margem de segurança alta.

Os preços-alvo são de terceiros (bancos e corretoras), publicados com data e fonte. A pesquisa na web é
feita pelo Claude na sessão, a cada rodada, e gravada em saida/ultima_rodada/<TICKER>/precos_alvo.json
(o arquivo acumula: alvos antigos ficam). Não são estimativas do modelo e não entram no preço justo.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import caminhos
from . import precos as P


def ler_alvos(ticker: str) -> dict | None:
    p = caminhos.pasta_ultima_rodada(ticker) / "precos_alvo.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _itens(alvos: list[dict] | None, desd: pd.DataFrame, base) -> list[tuple]:
    """(data, casa, alvo na base de ações de `base`, registro), em ordem de data."""
    out = []
    for a in alvos or []:
        try:
            d, v = pd.Timestamp(str(a["data"])[:10]), float(a["preco_alvo"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        out.append((d, str(a.get("casa") or "?"), v / P.fator_entre(desd, d, base), a))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def media_em(itens: list[tuple], t, validade_meses: int) -> tuple[float, int, list[tuple]]:
    """Média do último preço-alvo de cada casa publicado até `t` e nos `validade_meses` anteriores."""
    t = pd.Timestamp(t)
    ini = t - pd.DateOffset(months=validade_meses)
    ultimo: dict[str, tuple] = {}
    for x in itens:
        if ini < x[0] <= t:
            ultimo[x[1]] = x
    if not ultimo:
        return np.nan, 0, []
    vigentes = sorted(ultimo.values(), key=lambda x: (x[0], x[1]))
    return float(np.mean([x[2] for x in vigentes])), len(vigentes), vigentes


def linha(alvos: list[dict] | None, desd: pd.DataFrame, ini, ate, base, validade_meses: int = 12) -> list[list]:
    """Pontos [data, média, casas] da linha dos analistas entre `ini` e `ate` (degraus: fins de mês, datas
    de publicação e de vencimento dos alvos)."""
    itens = _itens(alvos, desd, base)
    if not itens:
        return []
    ini, ate = pd.Timestamp(ini), pd.Timestamp(ate)
    datas = set(pd.date_range(ini, ate, freq="ME"))
    for d, *_ in itens:
        for x in (d, d + pd.DateOffset(months=validade_meses)):
            if ini < x <= ate:
                datas.add(pd.Timestamp(x))
    datas.add(ate)
    out = []
    for t in sorted(datas):
        v, n, _ = media_em(itens, t, validade_meses)
        out.append([t.strftime("%Y-%m-%d"), None if not np.isfinite(v) else round(v, 4), n])
    # Sem nada antes do primeiro alvo: começa no primeiro ponto com valor.
    while out and out[0][1] is None:
        out.pop(0)
    return out


def resumo_hoje(alvos: dict | None, desd: pd.DataFrame, hoje, preco: float, validade_meses: int = 12) -> dict | None:
    """Média de hoje, casas vigentes e a tabela de todos os alvos pesquisados (na base de hoje)."""
    if not alvos or not alvos.get("itens"):
        return None
    itens = _itens(alvos["itens"], desd, hoje)
    v, n, vigentes = media_em(itens, hoje, validade_meses)
    vig = {(x[0], x[1]) for x in vigentes}
    tabela = [{"data": d, "casa": casa, "alvo_base_hoje": vb, "alvo_original": float(a["preco_alvo"]),
               "recomendacao": a.get("recomendacao"), "aproximada": bool(a.get("data_aproximada")), "vigente": (d, casa) in vig, "fontes": a.get("fontes") or []}
              for d, casa, vb, a in sorted(itens, key=lambda x: (x[0], x[1]), reverse=True)]
    return {"v": v, "n": n, "potencial": (v / preco - 1) if np.isfinite(v) and preco else np.nan,
            "data_pesquisa": alvos.get("data_pesquisa"), "tabela": tabela,
            "inicio": itens[0][0] if itens else None, "validade_meses": validade_meses}


# ------------------------------------------------------------------ épocas de margem alta

def epocas_de_margem_alta(mensal: pd.DataFrame, diario: pd.DataFrame, desd: pd.DataFrame, prov: pd.DataFrame,
                          hoje: dict, cfg: dict) -> dict:
    """Épocas em que a margem de segurança do walk-forward ficou no quinto superior do próprio histórico do
    ativo: o pico de cada época, o preço e o preço justo da época na base de ações de hoje, o que o preço fez
    6 e 12 meses depois e a comparação com o preço e a margem de hoje."""
    c = cfg.get("estudo_margem", {})
    quantil = float(c.get("quantil", 0.80))
    juntar = int(c.get("meses_para_juntar", 3))
    m = mensal[mensal["V"].notna() & (mensal["V"] > 0)].copy() if mensal is not None and len(mensal) else pd.DataFrame()
    if len(m) < 24:
        return {"ok": False, "motivo": "menos de 24 meses de estimativas walk-forward"}
    base = pd.Timestamp(hoje["data_preco"])
    m["margem"] = (m["V"] - m["preco"]) / m["V"]
    limiar = float(m["margem"].quantile(quantil))
    altos = list(m.index[m["margem"] >= limiar])
    grupos: list[list[pd.Timestamp]] = []
    for d in altos:
        if grupos and (d.to_period("M") - grupos[-1][-1].to_period("M")).n <= juntar:
            grupos[-1].append(d)
        else:
            grupos.append([d])
    d_ok = diario[diario["data"] <= base]
    aj = P.serie_ajustada(d_ok, desd, base)
    rt = P.retorno_total(d_ok, desd, prov, base)

    def depois(serie: pd.Series, t: pd.Timestamp, meses: int) -> float:
        alvo = t + pd.DateOffset(months=meses)
        if alvo > serie.index.max():
            return np.nan
        p0 = serie[serie.index <= t]
        p1 = serie[serie.index >= alvo]
        if p0.empty or p1.empty:
            return np.nan
        return float(p1.iloc[0] / p0.iloc[-1] - 1)

    preco_hoje = float(aj.iloc[-1]) if len(aj) else float(hoje.get("preco", np.nan))
    epocas = []
    for g in grupos:
        sub = m.loc[g]
        pico = sub["margem"].idxmax()
        f = P.fator_entre(desd, pico, base)
        p_pico = float(sub.at[pico, "preco"]) / f
        epocas.append({
            "inicio": g[0], "fim": g[-1], "meses": len(g), "pico": pico, "margem_pico": float(sub.at[pico, "margem"]),
            "preco_pico": p_pico, "V_pico": float(sub.at[pico, "V"]) / f,
            "ret6": depois(aj, pico, 6), "ret12": depois(aj, pico, 12),
            "ret6_total": depois(rt, pico, 6), "ret12_total": depois(rt, pico, 12),
            "hoje_vs_pico": preco_hoje / p_pico - 1 if p_pico else np.nan})
    V_hoje = float(hoje.get("V", np.nan))
    margem_hoje = (V_hoje - preco_hoje) / V_hoje if np.isfinite(V_hoje) and V_hoje > 0 else np.nan
    pct_hoje = float((m["margem"] < margem_hoje).mean()) if np.isfinite(margem_hoje) else np.nan
    todos12 = [e["ret12_total"] for e in epocas if np.isfinite(e["ret12_total"])]
    todos6 = [e["ret6_total"] for e in epocas if np.isfinite(e["ret6_total"])]
    return {"ok": True, "quantil": quantil, "limiar": limiar, "n_meses": len(m), "inicio": m.index[0], "fim": m.index[-1],
            "epocas": epocas, "preco_hoje": preco_hoje, "V_hoje": V_hoje, "margem_hoje": margem_hoje,
            "percentil_hoje": pct_hoje, "hoje_entre_altas": bool(np.isfinite(margem_hoje) and margem_hoje >= limiar),
            "media_ret6_total": float(np.mean(todos6)) if todos6 else np.nan, "n6": len(todos6),
            "media_ret12_total": float(np.mean(todos12)) if todos12 else np.nan, "n12": len(todos12)}
