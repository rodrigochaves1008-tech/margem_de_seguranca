"""Teste do ranking da margem de segurança dos últimos anos, sem internet."""
import numpy as np
import pandas as pd

from margem import margem_historica as M


def test_ranking_da_margem_e_posicao_de_hoje():
    idx = pd.date_range("2016-12-31", periods=120, freq="ME")
    V = np.linspace(10.0, 20.0, 120)
    margem = np.linspace(-0.2, 0.3, 120)                 # margem cresce de −20% a +30%
    mensal = pd.DataFrame({"V": V, "preco": V * (1 - margem)}, index=idx)
    sem_desd = pd.DataFrame({"data": pd.to_datetime([]), "fator": pd.Series([], dtype=float)})
    rk = M.ranking(mensal, sem_desd, V_hoje=20.0, preco_hoje=15.0, data_preco="2026-11-30", anos=10, top=5)
    assert rk["ok"] and rk["n"] == 120 and rk["total"] == 121
    assert rk["posicao"] == 13                            # 12 meses com margem acima de 25%
    assert abs(rk["percentil"] - 0.9) < 1e-12
    assert len(rk["linhas"]) == 6 and rk["linhas"][-1]["hoje"] and rk["linhas"][0]["posicao"] == 1
    assert "perto do topo" in M.texto(rk)
    assert not M.ranking(mensal, sem_desd, V_hoje=None, preco_hoje=15.0, data_preco="2026-11-30")["ok"]
