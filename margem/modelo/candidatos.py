"""Seleção de variáveis e ajuste dos modelos de um alvo, só com os meses de treino."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .rede import RedeNeural

MODELOS_BASE = ("rn", "ridge", "gb")          # sobre as variáveis escolhidas pelo LASSO; formam o ensemble_geral
MODELOS_PCA = ("pca_ridge", "pca_rn")         # sobre os componentes principais de todas as candidatas
MODELOS = MODELOS_BASE + MODELOS_PCA
REDES = ("rn", "pca_rn")
ENSEMBLES = {
    "ens_rn_ridge": ("rn", "ridge"),
    "ens_rn_gb": ("rn", "gb"),
    "ens_ridge_gb": ("ridge", "gb"),
    "ensemble_todos": ("rn", "ridge", "gb"),
    "ens_pca": ("pca_ridge", "pca_rn"),
    "ensemble_todos_pca": ("rn", "ridge", "gb", "pca_ridge", "pca_rn"),
}
NOMES_MODELOS = {
    "rn": "rede neural", "ridge": "regressão linear regularizada (ridge)", "gb": "gradient boosting",
    "ens_rn_ridge": "média: rede neural + ridge", "ens_rn_gb": "média: rede neural + boosting",
    "ens_ridge_gb": "média: ridge + boosting", "ensemble_todos": "média dos três modelos",
    "pca_ridge": "ridge sobre componentes principais (PCA)", "pca_rn": "rede neural sobre componentes principais (PCA)",
    "ens_pca": "média: ridge + rede sobre componentes principais",
    "ensemble_todos_pca": "média dos cinco modelos (três + dois sobre componentes principais)",
    "ensemble_geral": "média de todos os modelos e alvos",
    "media_todos": "média simples de todos os modelos e alvos (com PCA)", "passeio": "passeio aleatório (último preço)",
}
NOMES_ALVOS = {"log_valor_mercado": "valor de mercado", "log_pl": "P/L", "log_pvp": "P/VP",
               "log_ev_ebitda": "EV/EBITDA", "todos": "todos os alvos"}


def membros(modelo: str) -> tuple[str, ...]:
    """Modelos simples que compõem um candidato (ele mesmo, se não for média)."""
    if modelo == "ensemble_geral":
        return MODELOS_BASE
    if modelo == "media_todos":
        return MODELOS
    return ENSEMBLES.get(modelo, (modelo,))


@dataclass
class Ajuste:
    alvo: str
    variaveis: list[str]                 # escolhidas pelo LASSO (rede, ridge e boosting)
    medianas: dict
    modelos: dict = field(default_factory=dict)
    n_treino: int = 0
    info_selecao: dict = field(default_factory=dict)
    variaveis_pca: list[str] = field(default_factory=list)   # todas as candidatas com cobertura (PCA)
    medianas_pca: dict = field(default_factory=dict)

    def colunas(self, modelo: str) -> list[str]:
        return self.variaveis_pca if modelo in MODELOS_PCA else self.variaveis

    def matriz(self, linhas: pd.DataFrame, modelo: str) -> np.ndarray:
        if modelo in MODELOS_PCA:
            return linhas[self.variaveis_pca].fillna(self.medianas_pca).to_numpy(float)
        return linhas[self.variaveis].fillna(self.medianas).to_numpy(float)


def divisoes_no_tempo(datas, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validação cruzada em ordem de tempo, por data (linhas da mesma data ficam do mesmo lado).
    Com uma linha por data, é igual ao TimeSeriesSplit."""
    d = pd.to_datetime(pd.Series(datas)).to_numpy()
    unicas = np.unique(d)
    out = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(unicas):
        out.append((np.where(np.isin(d, unicas[tr]))[0], np.where(np.isin(d, unicas[te]))[0]))
    return out


class _RidgeEscalado:
    def __init__(self, cv=None):
        self.esc = StandardScaler()
        self.m = RidgeCV(alphas=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0), cv=cv if cv is not None else TimeSeriesSplit(n_splits=4))

    def fit(self, X, y):
        self.m.fit(self.esc.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self.esc.transform(X))


class Fatores:
    """Componentes principais das candidatas, padronizadas com o treino. k = menor número de componentes que
    explica `variancia` da variação delas, até `k_max`. Ajustados só com os dados de treino."""

    def __init__(self, variancia: float = 0.90, k_max: int = 12):
        self.variancia = float(variancia)
        self.k_max = int(k_max)

    def fit(self, X: np.ndarray) -> "Fatores":
        X = np.asarray(X, float)
        self.esc = StandardScaler().fit(X)
        self.pca = PCA(svd_solver="full").fit(self.esc.transform(X))
        acum = np.cumsum(self.pca.explained_variance_ratio_)
        k = int(np.searchsorted(acum, self.variancia - 1e-12) + 1)
        self.k = max(1, min(k, self.k_max, X.shape[1], len(acum)))
        return self

    def transformar(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.esc.transform(np.asarray(X, float)))[:, : self.k]


class _SobreFatores:
    """Modelo ajustado sobre os componentes principais; recebe as candidatas originais."""

    def __init__(self, fatores: Fatores, modelo):
        self.fatores = fatores
        self.modelo = modelo

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.modelo.predict(self.fatores.transformar(X))

    def prever_por_semente(self, X: np.ndarray) -> np.ndarray:
        return self.modelo.prever_por_semente(self.fatores.transformar(X))


def com_cobertura(treino: pd.DataFrame, candidatas: list[str], cobertura_min: float) -> tuple[list[str], dict]:
    """Candidatas com cobertura mínima no treino e que variam; medianas do treino para preencher as faltas."""
    cols = [c for c in candidatas if c in treino.columns]
    cob = treino[cols].notna().mean()
    cols = [c for c in cols if cob[c] >= cobertura_min]
    medianas = treino[cols].median().to_dict()
    X = treino[cols].fillna(medianas)
    cols = [c for c in cols if X[c].std() > 1e-12]
    return cols, {c: medianas[c] for c in cols}


def selecionar_variaveis(treino: pd.DataFrame, alvo: str, candidatas: list[str],
                         cfg_sel: dict) -> tuple[list[str], dict, dict]:
    """Cobertura, redundância e LASSO com validação no tempo, só com o histórico do próprio ativo."""
    y = treino[alvo]
    cols, medianas = com_cobertura(treino, candidatas, cfg_sel["cobertura_min"])
    X = treino[cols].fillna(medianas)
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
    k_max = limite_variaveis(len(treino), cfg_sel)
    Xm = StandardScaler().fit_transform(X[mantidas])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LassoCV(cv=divisoes_no_tempo(treino.index), alphas=30, max_iter=5000).fit(Xm, y.to_numpy())
    coef = pd.Series(np.abs(lasso.coef_), index=mantidas)
    escolhidas = [c for c in sorted(mantidas, key=lambda c: (-coef[c], c)) if coef[c] > 0][:k_max]
    if len(escolhidas) < 2:
        escolhidas = mantidas[: max(2, min(3, k_max))]
    info = {"candidatas": len(candidatas), "com_cobertura": len(cols), "nao_redundantes": len(mantidas),
            "escolhidas": len(escolhidas), "k_max": k_max}
    return escolhidas, {c: medianas[c] for c in escolhidas}, info


def limite_variaveis(n: int, cfg_sel: dict) -> int:
    return max(2, min(int(cfg_sel["max_variaveis"]), n // int(cfg_sel["obs_por_variavel"])))


def _rede(cfg_modelo: dict) -> RedeNeural:
    rn_cfg = cfg_modelo["rede_neural"]
    return RedeNeural(ocultos=int(rn_cfg["camadas_ocultas"][0]), alphas=rn_cfg["alphas"],
                      taxa=float(rn_cfg["taxa_aprendizado"]), epocas_max=int(rn_cfg["epocas_max"]),
                      paciencia=int(rn_cfg["paciencia"]), fracao_validacao=float(rn_cfg["fracao_validacao"]),
                      intervalo_validacao=int(rn_cfg["intervalo_validacao_meses"]), sementes=cfg_modelo["sementes"])


def ajustar(treino: pd.DataFrame, alvo: str, candidatas: list[str], cfg_modelo: dict) -> Ajuste | None:
    """Ajusta rede neural, ridge e boosting para um alvo, só com o histórico do próprio ativo."""
    treino = treino[treino[alvo].notna()]
    if treino.index.nunique() < int(cfg_modelo["meses_treino_min"]):
        return None
    variaveis, medianas, info = selecionar_variaveis(treino, alvo, candidatas, cfg_modelo["selecao_variaveis"])
    if not variaveis:
        return None
    X = treino[variaveis].fillna(medianas).to_numpy(float)
    y = treino[alvo].to_numpy(float)
    divisoes = divisoes_no_tempo(treino.index)
    gb_cfg = cfg_modelo["boosting"]
    modelos = {
        "rn": _rede(cfg_modelo).fit(X, y, datas=treino.index),
        "ridge": _RidgeEscalado(cv=divisoes).fit(X, y),
        "gb": HistGradientBoostingRegressor(
            max_iter=int(gb_cfg["iteracoes"]), learning_rate=float(gb_cfg["taxa_aprendizado"]),
            max_depth=int(gb_cfg["profundidade_max"]), min_samples_leaf=int(gb_cfg["folhas_min"]),
            l2_regularization=float(gb_cfg["l2"]), early_stopping=False, random_state=0,
        ).fit(X, y),
    }
    aj = Ajuste(alvo=alvo, variaveis=variaveis, medianas=medianas, modelos=modelos, n_treino=len(treino),
                info_selecao=info)
    cfg_pca = cfg_modelo.get("componentes_principais")
    if cfg_pca:
        cfg_sel = cfg_modelo["selecao_variaveis"]
        vp, medp = com_cobertura(treino, candidatas, cfg_sel["cobertura_min"])
        if len(vp) >= 2:
            Xp = treino[vp].fillna(medp).to_numpy(float)
            fat = Fatores(cfg_pca["variancia_explicada"], limite_variaveis(len(treino), cfg_sel)).fit(Xp)
            S = fat.transformar(Xp)
            modelos["pca_ridge"] = _SobreFatores(fat, _RidgeEscalado(cv=divisoes).fit(S, y))
            modelos["pca_rn"] = _SobreFatores(fat, _rede(cfg_modelo).fit(S, y, datas=treino.index))
            aj.variaveis_pca, aj.medianas_pca = vp, medp
            info.update({"pca_variaveis": len(vp), "pca_k": fat.k,
                         "pca_variancia": float(np.sum(fat.pca.explained_variance_ratio_[: fat.k]))})
    return aj


def prever(aj: Ajuste, linhas: pd.DataFrame) -> dict[str, np.ndarray]:
    """Previsões no espaço do alvo (log) de cada modelo e dos ensembles."""
    p = {m: np.asarray(aj.modelos[m].predict(aj.matriz(linhas, m)), float) for m in MODELOS if m in aj.modelos}
    for nome, ms in ENSEMBLES.items():
        if all(m in p for m in ms):
            p[nome] = np.mean([p[m] for m in ms], axis=0)
    return p
