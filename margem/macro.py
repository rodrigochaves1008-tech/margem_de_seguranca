"""Séries de macro, exterior e setor: carga com data de divulgação e valor point-in-time por mês."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .fontes import bcb, outras, yahoo


def _divulgacao(datas: pd.Series, defasagem: dict) -> pd.Series:
    if "meses" in defasagem:
        n = int(defasagem["meses"])
        fim_mes = datas + pd.offsets.MonthEnd(0)
        return fim_mes + pd.offsets.MonthEnd(n) if n > 0 else fim_mes
    return datas + pd.Timedelta(days=int(defasagem.get("dias", 0)))


def carregar_serie(sid: str, meta: dict) -> pd.DataFrame | None:
    """data (referência), valor, divulgacao. None para séries derivadas."""
    f = meta["fonte"]
    if f == "derivada":
        return None
    if f == "sgs":
        df = bcb.sgs(int(meta["codigo"]))
    elif f == "focus":
        df = bcb.focus(meta["conjunto"], meta["indicador"], meta.get("suavizada"), meta.get("ano_relativo"))
    elif f == "tesouro":
        df = outras.tesouro_curva(meta["tipo"], float(meta["prazo_anos"]))
    elif f == "fred":
        df = outras.fred(meta["codigo"])
    elif f == "ipea":
        df = outras.ipea(meta["codigo"])
    elif f == "sidra":
        df = outras.sidra(int(meta["tabela"]), int(meta["variavel"]), int(meta["classificacao"]), int(meta["categoria"]))
    elif f == "ons":
        df = outras.ons(meta["conjunto"], meta["subsistema"])
    elif f == "yahoo":
        df = yahoo.serie(meta["codigo"], max_idade_horas=20)["precos"].rename(columns={"fechamento": "valor"})
    else:
        raise ValueError(f"Fonte desconhecida: {f}")
    df = df[["data", "valor"]].dropna().sort_values("data").reset_index(drop=True)
    df["divulgacao"] = _divulgacao(df["data"], meta.get("defasagem", {"dias": 0}))
    return df


def ids_necessarios(setor_cfg: dict) -> list[str]:
    """Preditores não-empresa do setor + séries auxiliares (derivadas, sinais)."""
    cat = config.catalogo_series()
    p = setor_cfg.get("preditores", {})
    ids = [x for fam in ("setor", "macro_brasil", "exterior") for x in (p.get(fam) or [])]
    extra = ["cdi_mes", "ibovespa", "embi"]
    todos = []
    for sid in ids + extra:
        if sid in cat and sid not in todos:
            todos.append(sid)
            meta = cat[sid]
            if meta["fonte"] == "derivada":
                for parte in _partes_formula(meta["formula"]):
                    if parte not in todos:
                        todos.insert(0, parte)
    return todos


def _partes_formula(formula: str) -> list[str]:
    return [t.strip() for t in formula.replace("+", " ").replace("-", " ").split() if t.strip()]


def carregar_todas(ids: list[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    cat = config.catalogo_series()
    out, faltas = {}, []
    for sid in ids:
        meta = cat[sid]
        try:
            df = carregar_serie(sid, meta)
            if df is not None:
                out[sid] = df
        except Exception as e:
            faltas.append(f"{meta.get('nome', sid)}: indisponível ({e.__class__.__name__})")
    return out, faltas


def _limite_dias(meta: dict) -> int:
    d = meta.get("defasagem", {})
    if "meses" in d:
        return int((int(d["meses"]) + 3) * 31)
    return 45


def valores_pit(df: pd.DataFrame, datas: list[pd.Timestamp], meta: dict) -> np.ndarray:
    """Valor de cada data com o último dado divulgado até ela, já transformado."""
    out = np.full(len(datas), np.nan)
    if df is None or df.empty:
        return out
    div = df["divulgacao"].to_numpy(dtype="datetime64[ns]")
    dat = df["data"].to_numpy(dtype="datetime64[ns]")
    val = df["valor"].to_numpy(dtype=float)
    transf = meta.get("transformacao", "nivel")
    mensal = "meses" in meta.get("defasagem", {})
    limite = np.timedelta64(_limite_dias(meta), "D")
    for k, d in enumerate(datas):
        d64 = np.datetime64(pd.Timestamp(d), "ns")
        i = int(np.searchsorted(div, d64, side="right")) - 1
        if i < 0 or d64 - div[i] > limite:
            continue
        if transf == "nivel":
            out[k] = val[i]
        elif transf == "var12m":
            alvo = np.datetime64(pd.Timestamp(dat[i]) - pd.DateOffset(years=1), "ns")
            j = int(np.searchsorted(dat, alvo, side="right")) - 1
            if j < 0 or val[j] == 0:
                continue
            tolerancia = np.timedelta64(5 if mensal else 10, "D")
            if alvo - dat[j] > tolerancia:
                continue
            out[k] = val[i] / val[j] - 1
        elif transf == "acum12m":
            ini = np.datetime64(pd.Timestamp(dat[i]) - pd.DateOffset(months=11), "ns")
            janela = val[(dat >= ini) & (dat <= dat[i])]
            if len(janela) == 12:
                out[k] = 100 * (np.prod(1 + janela / 100) - 1)
        else:
            raise ValueError(f"Transformação desconhecida: {transf}")
    return out


def tabela_pit(series: dict[str, pd.DataFrame], ids: list[str], datas: list[pd.Timestamp]) -> pd.DataFrame:
    cat = config.catalogo_series()
    tab = pd.DataFrame(index=pd.DatetimeIndex(datas))
    for sid in ids:
        meta = cat[sid]
        if meta["fonte"] == "derivada":
            continue
        tab[sid] = valores_pit(series.get(sid), datas, meta)
    for sid in ids:
        meta = cat[sid]
        if meta["fonte"] == "derivada":
            tab[sid] = _avaliar(meta["formula"], tab)
    return tab


def _avaliar(formula: str, tab: pd.DataFrame) -> pd.Series:
    tokens = formula.split()
    res = tab[tokens[0]].astype(float).copy()
    for op, nome in zip(tokens[1::2], tokens[2::2]):
        res = res + tab[nome] if op == "+" else res - tab[nome]
    return res


def cobertura(series: dict[str, pd.DataFrame], ids: list[str], faltas: list[str]) -> list[dict]:
    cat = config.catalogo_series()
    linhas = []
    for sid in ids:
        meta = cat[sid]
        df = series.get(sid)
        fonte = {"sgs": "BCB SGS", "focus": "BCB Focus", "tesouro": "Tesouro Direto", "fred": "FRED",
                 "ipea": "Ipeadata", "yahoo": "Yahoo Finance", "derivada": "calculada", "sidra": "IBGE SIDRA",
                 "ons": "ONS dados abertos"}[meta["fonte"]]
        if meta["fonte"] == "sidra":
            codigo = f"tabela {meta['tabela']}"
        elif meta["fonte"] == "ons":
            codigo = f"{meta['conjunto'].upper()} {meta['subsistema']}"
        elif meta["fonte"] == "tesouro":
            n = meta.get("prazo_anos")
            codigo = f"{'prefixados' if meta.get('tipo') == 'prefixado' else 'IPCA+'}, {n} {'ano' if n == 1 else 'anos'}"
        else:
            codigo = meta.get("codigo") or meta.get("indicador") or meta.get("formula")
        linhas.append({
            "serie": meta.get("nome", sid), "id": sid, "familia": meta.get("familia"), "fonte": fonte,
            "codigo": str(codigo),
            "inicio": None if df is None or df.empty else df["data"].min(),
            "fim": None if df is None or df.empty else df["data"].max(),
            "defasagem": meta.get("defasagem"), "revisada": bool(meta.get("revisada", False)),
            "preditor": bool(meta.get("preditor", True)),
        })
    return linhas
