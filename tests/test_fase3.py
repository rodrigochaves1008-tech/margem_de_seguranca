"""Testes da Fase 3, sem internet: quebras estruturais e eventos."""
import numpy as np
import pandas as pd

from margem import estruturais as E


def _dados(quebra: bool, n: int = 200, semente: int = 7):
    rng = np.random.default_rng(semente)
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=n)
    if quebra:
        y[n // 2:] = -1.0 + 0.5 * x[n // 2:] + rng.normal(scale=0.3, size=n - n // 2)
    X = np.column_stack([np.ones(n), x])
    datas = pd.date_range("2010-01-31", periods=n, freq="ME")
    return y, X, datas


def test_bai_perron_acha_a_quebra():
    y, X, _ = _dados(True)
    r = E.bai_perron(y, X, min_seg=24, max_quebras=3)
    assert r["k"] == 1
    assert abs(r["quebras"][0] - 100) <= 3


def test_bai_perron_sem_quebra():
    y, X, _ = _dados(False)
    assert E.bai_perron(y, X, min_seg=24, max_quebras=3)["k"] == 0


def test_bai_perron_com_coluna_constante_no_trecho():
    y, X, _ = _dados(True)
    X = np.column_stack([X, np.r_[np.zeros(100), np.ones(100)]])  # vira constante dentro de cada trecho
    r = E.bai_perron(y, X, min_seg=24, max_quebras=3)
    assert r["k"] >= 1


def test_chow_robusto():
    y, X, datas = _dados(True)
    t = E.chow_robusto(y, X, datas, datas[100], min_lado=24, lags=12)
    assert t["testavel"] and t["p_valor"] < 0.01
    y, X, datas = _dados(False)
    t = E.chow_robusto(y, X, datas, datas[100], min_lado=24, lags=12)
    assert t["testavel"] and t["p_valor"] > 0.05
    t = E.chow_robusto(y, X, datas, datas[190], min_lado=24, lags=12)
    assert not t["testavel"] and t["n_depois"] == 10


def test_cusum():
    y, X, _ = _dados(True)
    assert E.cusum(y, X)["p_valor"] < 0.05


def test_capta():
    hoje = pd.Timestamp("2026-11-24")
    base = {"data": "2025-01-01", "vigencia": "2025-01-01"}
    assert E.capta({**base, "vigencia": "2027-01-01", "canais": ["selic"]}, ["selic"], hoje).startswith("Ainda não")
    assert E.capta({**base, "canais": []}, ["selic"], hoje).startswith("Não: nenhum")
    assert E.capta({**base, "canais": ["selic", "roe"]}, ["selic"], hoje).startswith("Em parte")
    assert E.capta({**base, "canais": ["roe"]}, ["selic"], hoje).startswith("Não: o modelo de hoje")


def test_normalizar_evento_da_configuracao():
    e = E._normalizar({"data": "2023-12-20", "titulo": "Reforma", "rotulo": "EC 132", "tipo": "tributario",
                       "fonte": "https://exemplo"}, "eventos.yaml")
    assert e["vigencia"] == "2023-12-20" and e["fontes"][0]["url"] == "https://exemplo"
    assert e["efeito_esperado"] == "incerto" and e["canais"] == []
