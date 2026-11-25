"""Testes dos candidatos e do diagnóstico de componentes principais (PCA), sem internet."""
import numpy as np
import pandas as pd

from margem import pca_preditores as PCA_P
from margem.modelo import candidatos as C
from margem.modelo import walkforward as WF

CFG = {"meses_treino_min": 36, "sementes": [11, 23],
       "selecao_variaveis": {"cobertura_min": 0.9, "correlacao_max": 0.9, "max_variaveis": 12, "obs_por_variavel": 10},
       "rede_neural": {"camadas_ocultas": [4], "alphas": [1.0], "taxa_aprendizado": 0.01, "epocas_max": 40,
                       "paciencia": 10, "fracao_validacao": 0.2, "intervalo_validacao_meses": 6},
       "boosting": {"iteracoes": 20, "taxa_aprendizado": 0.1, "profundidade_max": 2, "folhas_min": 5, "l2": 1.0},
       "componentes_principais": {"variancia_explicada": 0.90}}


def _base(n=120, semente=3):
    rng = np.random.default_rng(semente)
    f1, f2 = np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=n))
    d = {}
    for i in range(4):
        d[f"a{i}"] = f1 * (1 + 0.2 * i) + rng.normal(scale=0.05, size=n)
        d[f"b{i}"] = f2 * (1 - 0.1 * i) + rng.normal(scale=0.05, size=n)
    d["ruido"] = rng.normal(size=n) * 0.01
    d["log_valor_mercado"] = 1 + 0.5 * f1 - 0.3 * f2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame(d, index=pd.date_range("2012-01-31", periods=n, freq="ME"))


CANDIDATAS = [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)] + ["ruido"]


def test_fatores_acham_dois_componentes():
    b = _base()
    X = b[[f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]].to_numpy(float)
    f = C.Fatores(0.90, 12).fit(X)
    assert f.k == 2 and f.transformar(X).shape == (len(b), 2)
    assert C.Fatores(0.90, 1).fit(X).k == 1                  # respeita o limite


def test_ajustar_com_componentes_principais():
    b = _base()
    aj = C.ajustar(b.iloc[:100], "log_valor_mercado", CANDIDATAS, CFG)
    assert set(C.MODELOS) <= set(aj.modelos)
    assert set(aj.variaveis) <= set(aj.variaveis_pca) and aj.colunas("pca_rn") == aj.variaveis_pca
    p = C.prever(aj, b.iloc[100:])
    for m in list(C.MODELOS) + list(C.ENSEMBLES):
        assert p[m].shape == (20,) and np.isfinite(p[m]).all()
    assert np.allclose(p["ens_pca"], (p["pca_ridge"] + p["pca_rn"]) / 2)
    assert C.membros("ensemble_geral") == C.MODELOS_BASE and C.membros("pca_rn") == ("pca_rn",)


def test_pca_na_data_so_usa_o_passado():
    b = _base()
    r = b.index[90]
    d = PCA_P.pca_na_data(b, r, CANDIDATAS, CFG)
    b2 = b.copy()
    b2.loc[b2.index >= r, CANDIDATAS] = np.random.default_rng(0).normal(size=(len(b2) - 90, len(CANDIDATAS))) * 100
    d2 = PCA_P.pca_na_data(b2, r, CANDIDATAS, CFG)
    assert d["k"] == d2["k"] == 2 and np.allclose(d["comunalidade"], d2["comunalidade"])
    assert sorted(map(len, d["grupos"])) == [4, 4]           # os dois blocos redundantes
    assert d["comunalidade"]["ruido"] < 0.5 and d["comunalidade"]["a0"] > 0.9


def test_recomendacao():
    b = _base()
    d = PCA_P.pca_na_data(b, b.index[-1], CANDIDATAS, CFG)
    imp = pd.DataFrame({"variavel": ["a0", "b0"], "importancia": [0.6, 0.4], "importancia_permutacao": [0.5, 0.5]})
    shap_m, perm_m = PCA_P.importancia_media([imp, imp])
    freq = pd.Series({"a0": 1.0, "b0": 0.8, "a1": 0.2})
    tab = PCA_P.recomendar(d, imp, shap_m, perm_m, freq, str, lambda v: "macro")
    rec = tab.set_index("variavel")["recomendacao"]
    assert rec["a0"] == "manter" and rec["b0"] == "manter"
    assert rec["a1"] == "reciclar no fator" and rec["b3"] == "reciclar no fator"
    assert rec["ruido"] == "candidato a sair"
    fr, n = PCA_P.frequencia_lasso([{"data": "2020-01-31", "alvo": "x", "variaveis": ["a0"]},
                                    {"data": "2021-01-31", "alvo": "x", "variaveis": ["a0", "b0"]},
                                    {"data": "2021-01-31", "alvo": "painel:x", "variaveis": ["ruido"]}], None, ["x"])
    assert n == 2 and fr["a0"] == 1.0 and fr["b0"] == 0.5 and "ruido" not in fr


def test_media_simples_e_variacao_entre_sementes():
    b = _base()
    b["acoes"], b["preco"] = 1.0, np.exp(b["log_valor_mercado"])
    aj = C.ajustar(b.iloc[:100], "log_valor_mercado", CANDIDATAS, CFG)
    out = WF._todas_previsoes({"log_valor_mercado": aj}, b.iloc[100:], ["log_valor_mercado"])
    logs = np.mean([np.log(out[f"{m}|log_valor_mercado"]) for m in C.MODELOS], axis=0)
    assert np.allclose(out["media_todos|todos"], np.exp(logs)) and C.membros("media_todos") == C.MODELOS
    linha = b.iloc[[105]]

    def var(esc):
        return WF._variacao_sementes(WF.Passo(data=b.index[100], retreino=b.index[100],
                                              ajustes={"log_valor_mercado": aj}, escolhido=esc, pontuacoes={}), linha)
    assert var("ridge|log_valor_mercado") == 0.0
    v_redes = max(var("rn|log_valor_mercado"), var("pca_rn|log_valor_mercado"))
    assert v_redes > 0 and 0 < var("media_todos|todos") < v_redes
    cfg = {"metodo": "media_simples", "candidato_fixo": "media_todos|todos", "horizontes_meses": [6, 12],
           "janela_meses": 60, "meses_fechados_min": 12, "candidato_padrao": "ensemble_todos|log_pvp"}
    cands = WF.nomes_candidatos(["log_valor_mercado"])
    sem_desd = pd.DataFrame({"data": pd.to_datetime([]), "fator": pd.Series([], dtype=float)})
    assert WF.escolher(b.index[50], pd.DataFrame(columns=cands), b["preco"], sem_desd, cfg, cands)[0] == "media_todos|todos"
    cfg["metodo"] = "menor_erro"
    assert WF.escolher(b.index[50], pd.DataFrame(columns=cands), b["preco"], sem_desd, cfg, cands)[0] == cands[0]
