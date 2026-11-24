"""Testes da Fase 2, sem internet: units, mapa de setores, validação no tempo, escala de ações."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from margem import config, fundamentos
from margem.fontes.cadastro import _acoes_por_unidade
from margem.modelo.candidatos import divisoes_no_tempo


def test_acoes_por_unidade():
    assert _acoes_por_unidade("Units", "1 ON + 2 PN") == 3
    assert _acoes_por_unidade("Units", "1 ação ordinária e 4 ações preferenciais") == 5
    assert _acoes_por_unidade("Ações Ordinárias", None) == 1


def test_mapa_de_setores_da_cvm():
    assert config.setor_do_cvm("Bancos") == "bancos"
    assert config.setor_do_cvm("Emp. Adm. Part. - Energia Elétrica") == "energia_eletrica"
    assert config.setor_do_cvm("Emp. Adm. Part. - Máqs., Equip., Veíc. e Peças") == "industria"
    assert config.setor_do_cvm("Telecomunicações") is None


def test_divisoes_no_tempo_igual_ao_timeseriessplit_com_uma_empresa():
    datas = pd.date_range("2015-01-31", periods=60, freq="ME")
    a = divisoes_no_tempo(datas)
    b = list(TimeSeriesSplit(n_splits=4).split(np.arange(60)))
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_divisoes_no_tempo_no_painel_nao_partem_uma_data():
    datas = np.repeat(pd.date_range("2015-01-31", periods=40, freq="ME"), 3)
    for tr, te in divisoes_no_tempo(datas):
        assert set(datas[tr]).isdisjoint(set(datas[te]))
        assert datas[tr].max() < datas[te].min()


def test_escala_das_acoes_em_milhares():
    c = pd.DataFrame({"dt_refer": pd.to_datetime(["2021-03-31"]), "divulgacao": pd.to_datetime(["2021-05-01"]),
                      "acoes_circulacao": [1_458_191.0], "acoes_total": [1_458_191.0]})
    fre = pd.DataFrame({"dt_refer": pd.to_datetime(["2021-01-01"]), "acoes_total": [1_458_191_000.0]})
    out = fundamentos._corrigir_escala(c, fre)
    assert out["acoes_total"].iloc[0] == 1_458_191_000.0
