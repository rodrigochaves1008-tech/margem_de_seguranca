"""Seleção de variáveis e ajuste dos modelos de um alvo, só com os meses de treino."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .rede import RedeNeural

MODELOS_BASE = ("rn", "ridge", "gb")
ENSEMBLES = {
    "ens_rn_ridge": ("rn", "ridge"),
    "ens_rn_gb": ("rn", "gb"),
    "ens_ridge_gb": ("ridge", "gb"),
    "ensemble_todos": ("rn", "ridge", "gb"),
}
NOMES_MODELOS = {
    "rn": "rede neural", "ridge": "regressão linear regularizada (ridge)", "gb": "gradient boosting",
    "ens_rn_ridge": "média: rede neural + ridge", "ens_rn_gb": "média: rede neural + boosting",
    "ens_ridge_gb": "média: ridge + boosting", "ensemble_todos": "média dos três modelos",
    "ensemble_geral": "média de todos os modelos e alvos", "passeio": "passeio aleatório (último preço)",
}
NOMES_ALVOS = {"log_valor_mercado": "valor de mercado", "log_pl": "P/L", "log_pvp": "P/VP", "todos": "todos os alvos"}


@dataclass
class Ajuste:
    alvo: str
    variaveis: list[str]
    medianas: dict
    modelos: dict = field(default_factory=dict)
    n_treino: int = 0
    info_selecao: dict = field(default_factory=dict)


class _RidgeEscalado:
    def __init__(self):
        self.esc = StandardScaler()
        self.m = RidgeCV(alphas=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0), cv=TimeSeriesSplit(n_splits=4))

    def fit(self, X, y):
        self.m.fit(self.esc.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.esc.transform(X))


def selecionar_variaveis(treino: pd.DataFrame, alvo: str, candidatas: list[str], cfg_sel: dict) -> tuple[list[str], dict, dict]:
    y = treino[alvo]
    cols = [c for c in candidatas if c in treino.columns]
    cob = treino[cols].notna().mean()
    cols = [c for c in cols if cob[c] >= cfg_sel["cobertura_min"]]
    medianas = treino[cols].median().to_dict()
    X = treino[cols].fillna(medianas)
    cols = [c for c in cols if X[c].std() > 0]
    X = X[cols]
    if not cols:
        return [], {}, {"motivo": "nenhuma variável com cobertura suficiente"}
    rk = pd.concat([X, y], axis=1).rank()
    corr_alvo = rk[cols].corrwith(rk[alvo]).abs().fillna(0)
    ordem = sorted(cols, key=lambda c: (-corr_alvo[c], c))
    corr = rk[cols].corr().abs()
    mantidas: list[str] = []
    for c in ordem:
        if all(corr.loc[c, m] < cfg_sel["correlacao_max"] for m in mantidas):
            mantidas.append(c)
    n = len(treino)
    k_max = max(2, min(int(cfg_sel["max_variaveis"]), n // int(cfg_sel["obs_por_variavel"])))
    Xm = StandardScaler().fit_transform(X[mantidas])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LassoCV(cv=TimeSeriesSplit(n_splits=4), alphas=30, max_iter=5000).fit(Xm, y.to_numpy())
    coef = pd.Series(np.abs(lasso.coef_), index=mantidas)
    escolhidas = [c for c in sorted(mantidas, key=lambda c: (-coef[c], c)) if coef[c] > 0][:k_max]
    if len(escolhidas) < 2:
        escolhidas = mantidas[: max(2, min(3, k_max))]
    info = {"candidatas": len(candidatas), "com_cobertura": len(cols), "nao_redundantes": len(mantidas),
            "escolhidas": len(escolhidas), "k_max": k_max}
    return escolhidas, {c: medianas[c] for c in escolhidas}, info


def ajustar(treino: pd.DataFrame, alvo: str, candidatas: list[str], cfg_modelo: dict) -> Ajuste | None:
    treino = treino[treino[alvo].notna()]
    if len(treino) < int(cfg_modelo["meses_treino_min"]):
        return None
    variaveis, medianas, info = selecionar_variaveis(treino, alvo, candidatas, cfg_modelo["selecao_variaveis"])
    if not variaveis:
        return None
    X = treino[variaveis].fillna(medianas).to_numpy(float)
    y = treino[alvo].to_numpy(float)
    rn_cfg = cfg_modelo["rede_neural"]
    gb_cfg = cfg_modelo["boosting"]
    modelos = {
        "rn": RedeNeural(ocultos=int(rn_cfg["camadas_ocultas"][0]), alphas=rn_cfg["alphas"],
                         taxa=float(rn_cfg["taxa_aprendizado"]), epocas_max=int(rn_cfg["epocas_max"]),
                         paciencia=int(rn_cfg["paciencia"]), fracao_validacao=float(rn_cfg["fracao_validacao"]),
                         intervalo_validacao=int(rn_cfg["intervalo_validacao_meses"]),
                         sementes=cfg_modelo["sementes"]).fit(X, y),
        "ridge": _RidgeEscalado().fit(X, y),
        "gb": HistGradientBoostingRegressor(
            max_iter=int(gb_cfg["iteracoes"]), learning_rate=float(gb_cfg["taxa_aprendizado"]),
            max_depth=int(gb_cfg["profundidade_max"]), min_samples_leaf=int(gb_cfg["folhas_min"]),
            l2_regularization=float(gb_cfg["l2"]), early_stopping=False, random_state=0,
        ).fit(X, y),
    }
    return Ajuste(alvo=alvo, variaveis=variaveis, medianas=medianas, modelos=modelos, n_treino=len(treino),
                  info_selecao=info)


def prever(aj: Ajuste, linhas: pd.DataFrame) -> dict[str, np.ndarray]:
    """Previsões no espaço do alvo (log) de cada modelo e dos ensembles."""
    X = linhas[aj.variaveis].fillna(aj.medianas).to_numpy(float)
    p = {m: np.asarray(aj.modelos[m].predict(X), float) for m in MODELOS_BASE}
    for nome, membros in ENSEMBLES.items():
        p[nome] = np.mean([p[m] for m in membros], axis=0)
    return p
