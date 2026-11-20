"""Testes rápidos, sem internet: point-in-time, fatores de ajuste, formatos e determinismo."""
import numpy as np
import pandas as pd

from margem import formatos, graficos, macro
from margem import precos as P
from margem.modelo.rede import RedeNeural


def test_formatos_brasileiros():
    assert formatos.moeda(1234.5) == "R$ 1.234,50"
    assert formatos.pct(0.264, 1, True) == "+26,4%"
    assert formatos.pct(-0.15, 0) == "−15%"
    assert formatos.data("2026-11-20") == "20/11/2026"


def test_valor_pit_nao_usa_o_futuro():
    df = pd.DataFrame({"data": pd.to_datetime(["2026-08-01", "2026-09-01", "2026-10-01"]), "valor": [1.0, 2.0, 3.0]})
    meta = {"defasagem": {"meses": 1}, "transformacao": "nivel"}
    df["divulgacao"] = macro._divulgacao(df["data"], meta["defasagem"])
    v = macro.valores_pit(df, [pd.Timestamp("2026-10-31"), pd.Timestamp("2026-11-20"), pd.Timestamp("2026-11-30")], meta)
    # Setembro só entra no fim de outubro; outubro só no fim de novembro.
    assert v[0] == 2.0 and v[1] == 2.0 and v[2] == 3.0


def test_var12m_e_acum12m():
    datas = pd.date_range("2025-01-01", "2026-01-01", freq="MS")
    df = pd.DataFrame({"data": datas, "valor": np.linspace(100, 112, len(datas))})
    df["divulgacao"] = macro._divulgacao(df["data"], {"meses": 0})
    v = macro.valores_pit(df, [pd.Timestamp("2026-01-31")], {"defasagem": {"meses": 0}, "transformacao": "var12m"})
    assert abs(v[0] - 0.12) < 1e-12
    dm = pd.DataFrame({"data": pd.date_range("2025-01-01", periods=12, freq="MS"), "valor": [1.0] * 12})
    dm["divulgacao"] = macro._divulgacao(dm["data"], {"meses": 0})
    a = macro.valores_pit(dm, [pd.Timestamp("2025-12-31")], {"defasagem": {"meses": 0}, "transformacao": "acum12m"})
    assert abs(a[0] - (1.01 ** 12 - 1) * 100) < 1e-9


def test_fatores_de_desdobramento():
    desd = pd.DataFrame({"data": pd.to_datetime(["2024-04-16"]), "fator": [2.0]})
    assert P.fator_entre(desd, "2024-03-31", "2024-04-30") == 2.0
    assert P.fator_entre(desd, "2024-04-16", "2024-04-30") == 1.0
    assert list(P.fatores_ate(pd.to_datetime(["2024-04-15", "2024-04-16"]), desd, "2026-01-01")) == [2.0, 1.0]


def test_ajuste_do_preco_justo_depois_da_analise():
    diario = pd.DataFrame({"data": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
                           "fechamento": [20.0, 20.0, 10.0]})
    desd = pd.DataFrame({"data": pd.to_datetime(["2026-01-06"]), "fator": [2.0]})
    prov = pd.DataFrame({"data": pd.to_datetime(["2026-01-05"]), "valor": [1.0]})
    v, houve = graficos.ajuste_pos_analise(40.0, "2026-01-02", "2026-01-06", diario, desd, prov)
    # Provento de R$ 1 sobre R$ 20 (fator 0,95) e desdobramento 2:1.
    assert houve and abs(v - 40.0 * 0.95 / 2) < 1e-12


def test_rede_neural_deterministica():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    y = X @ np.array([0.5, -0.2, 0.1, 0.0]) + rng.normal(scale=0.1, size=80)
    a = RedeNeural(sementes=(1, 2, 3)).fit(X, y).predict(X[:5])
    b = RedeNeural(sementes=(1, 2, 3)).fit(X, y).predict(X[:5])
    assert np.array_equal(a, b)
