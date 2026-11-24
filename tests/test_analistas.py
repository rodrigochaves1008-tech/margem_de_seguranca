"""Testes da linha dos preços-alvo dos analistas e do estudo das épocas de margem alta, sem internet."""
import numpy as np
import pandas as pd

from margem import analistas as A

DESD = pd.DataFrame({"data": pd.to_datetime(["2024-04-16"]), "fator": [2.0]})  # desdobramento 2:1
HOJE = pd.Timestamp("2026-11-24")


def test_media_do_ultimo_alvo_de_cada_casa_com_validade():
    alvos = [{"data": "2023-01-10", "casa": "X", "preco_alvo": 60.0},   # antes do 2:1: R$ 30 na base de hoje
             {"data": "2024-06-01", "casa": "X", "preco_alvo": 34.0},
             {"data": "2024-07-01", "casa": "Y", "preco_alvo": 36.0}]
    itens = A._itens(alvos, DESD, HOJE)
    v, n, _ = A.media_em(itens, pd.Timestamp("2023-06-30"), 12)
    assert n == 1 and abs(v - 30.0) < 1e-9
    v, n, _ = A.media_em(itens, pd.Timestamp("2024-12-31"), 12)
    assert n == 2 and abs(v - 35.0) < 1e-9           # da casa X vale só o alvo mais recente
    v, n, _ = A.media_em(itens, pd.Timestamp("2025-07-02"), 12)
    assert n == 0 and np.isnan(v)                    # todos com mais de 12 meses


def test_alvo_publicado_depois_da_data_nao_entra():
    itens = A._itens([{"data": "2025-03-01", "casa": "X", "preco_alvo": 10.0}], DESD, HOJE)
    assert A.media_em(itens, pd.Timestamp("2025-02-28"), 12)[1] == 0
    pontos = A.linha([{"data": "2025-03-01", "casa": "X", "preco_alvo": 10.0}], DESD, "2024-11-30", HOJE, HOJE)
    assert pontos[0][0] == "2025-03-01" and pontos[0][1] == 10.0


def test_epocas_de_margem_alta():
    idx = pd.date_range("2018-01-31", periods=60, freq="ME")
    V = np.full(60, 9.0)
    V[10:13], V[40:42] = 14.0, 15.0                  # duas épocas de margem alta
    mensal = pd.DataFrame({"V": V, "preco": np.full(60, 10.0)}, index=idx)
    dias = pd.bdate_range("2018-01-01", "2023-01-31")
    diario = pd.DataFrame({"data": dias, "fechamento": np.linspace(10.0, 12.0, len(dias))})
    sem_desd = pd.DataFrame({"data": pd.to_datetime([]), "fator": pd.Series([], dtype=float)})
    prov = pd.DataFrame({"data": pd.to_datetime([]), "valor": pd.Series([], dtype=float)})
    e = A.epocas_de_margem_alta(mensal, diario, sem_desd, prov, {"data_preco": "2023-01-31", "V": 13.0, "preco": 12.0},
                                {"estudo_margem": {"quantil": 0.95, "meses_para_juntar": 3}})
    assert e["ok"] and len(e["epocas"]) == 2
    assert e["epocas"][0]["pico"] == idx[10] and e["epocas"][1]["pico"] == idx[40]
    assert np.isfinite(e["epocas"][0]["ret12_total"]) and e["epocas"][0]["ret12_total"] > 0
    assert not e["hoje_entre_altas"]
