"""Ranking dos preditores: importância por SHAP, conferida por permutação, em % do total.

A explicação é feita sobre o log do valor de mercado estimado pelo candidato escolhido. Quando o
alvo é um múltiplo (P/L ou P/VP), a conversão para preço multiplica pelo lucro ou pelo patrimônio;
esse termo entra como um preditor a mais ("lucro 12 meses (conversão do P/L)", por exemplo).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .fundamentos import EXPLICACAO_EMPRESA, PREDITORES_EMPRESA
from .modelo import candidatos as C
from .painel import familia_de

CONVERSAO = {"log_pl": ("_conv_lucro", "lucro_12m", "Lucro de 12 meses (conversão do P/L)"),
             "log_pvp": ("_conv_patrimonio", "patrimonio", "Patrimônio (conversão do P/VP)")}
FAMILIAS = {"empresa": "empresa", "setor": "setor", "macro_brasil": "macro Brasil", "exterior": "exterior",
            "conversao": "empresa", "outro": "outro"}


def _componentes(passo) -> list[tuple[str, str]]:
    """(alvo, modelo) que compõem o candidato escolhido."""
    modelo, alvo = passo.escolhido.split("|")
    alvos = list(passo.ajustes) if alvo == "todos" else [alvo]
    if modelo == "ensemble_geral":
        return [(a, m) for a in alvos for m in C.MODELOS_BASE]
    membros = C.ENSEMBLES.get(modelo, (modelo,))
    return [(a, m) for a in alvos for m in membros]


def funcao_explicada(passo):
    comps = [(a, m) for a, m in _componentes(passo) if a in passo.ajustes]
    variaveis: list[str] = []
    for a, _ in comps:
        for v in passo.ajustes[a].variaveis:
            if v not in variaveis:
                variaveis.append(v)
    conv = []
    for a, _ in comps:
        if a in CONVERSAO and CONVERSAO[a][0] not in conv:
            conv.append(CONVERSAO[a][0])
    colunas = variaveis + conv

    def f(M: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(M, columns=colunas)
        partes = []
        for a, m in comps:
            aj = passo.ajustes[a]
            X = df[aj.variaveis].fillna(aj.medianas).to_numpy(float)
            y = np.asarray(aj.modelos[m].predict(X), float)
            if a in CONVERSAO:
                y = y + df[CONVERSAO[a][0]].to_numpy(float)
            partes.append(y)
        return np.mean(partes, axis=0)

    return colunas, f


def _matriz(base: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(index=base.index)
    for c in colunas:
        if c == "_conv_lucro":
            m[c] = np.log(base["lucro_12m"].where(base["lucro_12m"] > 0))
        elif c == "_conv_patrimonio":
            m[c] = np.log(base["patrimonio"].where(base["patrimonio"] > 0))
        else:
            m[c] = base[c]
    return m


def nome_preditor(v: str) -> str:
    if v == "_conv_lucro":
        return CONVERSAO["log_pl"][2]
    if v == "_conv_patrimonio":
        return CONVERSAO["log_pvp"][2]
    if v in PREDITORES_EMPRESA:
        return PREDITORES_EMPRESA[v]
    return config.catalogo_series().get(v, {}).get("nome", v)


def explicacao_preditor(v: str) -> str:
    if v == "_conv_lucro":
        return "o lucro dos últimos 12 meses, que multiplica o P/L estimado para virar preço"
    if v == "_conv_patrimonio":
        return "o patrimônio do banco, que multiplica o P/VP estimado para virar preço"
    if v in EXPLICACAO_EMPRESA:
        return EXPLICACAO_EMPRESA[v]
    return config.catalogo_series().get(v, {}).get("explicacao", "")


def familia_preditor(v: str, setor: dict) -> str:
    if v.startswith("_conv"):
        return "empresa"
    return FAMILIAS.get(familia_de(v, setor), "outro")


def importancia(passo, base: pd.DataFrame, data_hoje: pd.Timestamp, setor: dict, n_fundo: int = 80) -> pd.DataFrame:
    import shap

    colunas, f = funcao_explicada(passo)
    treino = base[(base.index < pd.Timestamp(passo.retreino))]
    alvo_ok = treino[[a for a in passo.ajustes]].notna().all(axis=1) if passo.ajustes else treino.index == treino.index
    treino = treino[alvo_ok]
    M_treino = _matriz(treino, colunas)
    meds = M_treino.median()
    M_treino = M_treino.fillna(meds)
    M_hoje = _matriz(base.loc[[pd.Timestamp(data_hoje)]], colunas).fillna(meds)
    passo_fundo = max(1, len(M_treino) // n_fundo)
    fundo = M_treino.iloc[::passo_fundo].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expl = shap.explainers.Permutation(f, fundo, seed=0)
        sv_treino = expl(M_treino.to_numpy(float), max_evals=2 * len(colunas) + 1, silent=True).values
        sv_hoje = expl(M_hoje.to_numpy(float), max_evals=2 * len(colunas) + 1, silent=True).values[0]
    imp = np.abs(sv_treino).mean(axis=0)
    # Conferência por permutação: quanto piora o ajuste ao log do valor de mercado ao embaralhar.
    alvo_vm = np.log(treino["valor_mercado"]).to_numpy(float)
    base_err = float(np.mean((f(M_treino.to_numpy(float)) - alvo_vm) ** 2))
    rng = np.random.default_rng(0)
    perm = []
    for j in range(len(colunas)):
        aumentos = []
        for _ in range(5):
            M = M_treino.to_numpy(float).copy()
            M[:, j] = rng.permutation(M[:, j])
            aumentos.append(float(np.mean((f(M) - alvo_vm) ** 2)) - base_err)
        perm.append(max(0.0, float(np.mean(aumentos))))
    perm = np.array(perm)
    tab = pd.DataFrame({
        "variavel": colunas,
        "nome": [nome_preditor(c) for c in colunas],
        "familia": [familia_preditor(c, setor) for c in colunas],
        "importancia": imp / imp.sum() if imp.sum() > 0 else imp,
        "importancia_permutacao": perm / perm.sum() if perm.sum() > 0 else perm,
        "shap_hoje": sv_hoje,
        "valor_hoje": M_hoje.iloc[0].to_numpy(float),
        "mediana_treino": meds.to_numpy(float),
    }).sort_values("importancia", ascending=False).reset_index(drop=True)
    tab["direcao"] = np.where(tab["shap_hoje"] >= 0, "para cima", "para baixo")
    tab["explicacao"] = [explicacao_preditor(v) for v in tab["variavel"]]
    rk = tab[["importancia", "importancia_permutacao"]].rank()
    tab.attrs["concordancia"] = float(rk.corr().iloc[0, 1]) if len(tab) > 2 else np.nan
    return tab
