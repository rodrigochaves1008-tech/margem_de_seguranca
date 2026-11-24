"""Preços diários sem ajuste, desdobramentos/grupamentos e proventos, e os fatores de ajuste.

- Fechamento bruto: B3 (COTAHIST) quando o arquivo do ano estiver disponível; senão, o
  fechamento do Yahoo (que já vem ajustado por desdobramentos) volta para a base da data
  multiplicando pelos desdobramentos posteriores.
- O preço usado no modelo e nos gráficos é ajustado SÓ por desdobramentos e grupamentos.
- Proventos (dividendos e JCP, valor bruto por ação na base da data "ex") servem para o
  retorno com proventos reinvestidos, o payout e o ajuste do preço justo no ranking dos ativos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .fontes import b3, yahoo


@dataclass
class Precos:
    diario: pd.DataFrame          # data, fechamento (bruto), negocios, origem
    desdobramentos: pd.DataFrame  # data (ex), fator (2.0 = desdobramento 2:1; 0.1 = grupamento 10:1)
    proventos: pd.DataFrame       # data (ex), valor (R$ por ação, base da data ex)
    notas: list[str] = field(default_factory=list)


def fator_entre(desd: pd.DataFrame, de, ate) -> float:
    """Produto dos fatores com data ex em (de, ate]. Preço na base de `ate` = bruto em `de` ÷ fator."""
    if desd is None or desd.empty:
        return 1.0
    m = (desd["data"] > pd.Timestamp(de)) & (desd["data"] <= pd.Timestamp(ate))
    return float(np.prod(desd.loc[m, "fator"])) if m.any() else 1.0


def fatores_ate(datas: pd.Series, desd: pd.DataFrame, ate) -> np.ndarray:
    """Vetor de fatores (de cada data até `ate`)."""
    datas = pd.to_datetime(pd.Series(datas)).to_numpy()
    out = np.ones(len(datas))
    if desd is None or desd.empty:
        return out
    for d, f in zip(desd["data"], desd["fator"]):
        if d <= pd.Timestamp(ate):
            out[datas < np.datetime64(d)] *= f
    return out


def carregar(ticker: str, ano_inicio: int = 2010, max_idade_horas: float = 4, acoes_por_unidade: float = 1.0) -> Precos:
    """acoes_por_unidade > 1 (units): preços e proventos divididos para virar valor por ação."""
    notas: list[str] = []
    yh = yahoo.serie(f"{ticker.upper()}.SA", max_idade_horas=max_idade_horas)
    desd = yh["desdobramentos"].copy()
    hoje = pd.Timestamp(date.today()) + pd.Timedelta(days=1)

    py = yh["precos"].copy()
    py = py[py["data"] >= pd.Timestamp(f"{ano_inicio}-01-01")]
    py["bruto_yahoo"] = py["fechamento"].to_numpy() * fatores_ate(py["data"], desd, hoje)

    pb, faltaram = b3.precos_brutos(ticker, ano_inicio)
    if faltaram:
        notas.append(
            "Cotações da B3 indisponíveis em " + ", ".join(map(str, faltaram))
            + " (a B3 recusou o download); nesses anos o fechamento veio do Yahoo, trazido de volta "
              "para a base da data."
        )
    base = py[["data", "bruto_yahoo", "volume"]].merge(
        pb[["data", "fechamento", "negocios"]] if not pb.empty else pd.DataFrame(columns=["data", "fechamento", "negocios"]),
        on="data", how="outer",
    ).sort_values("data")
    tem_b3 = base["fechamento"].notna()
    # Conferência B3 × Yahoo nos dias em comum.
    comum = tem_b3 & base["bruto_yahoo"].notna()
    if comum.any():
        dif = (base.loc[comum, "bruto_yahoo"] / base.loc[comum, "fechamento"] - 1).abs()
        n_ruim = int((dif > 0.01).sum())
        notas.append(
            f"Conferência B3 × Yahoo: {int(comum.sum())} pregões em comum; diferença mediana "
            f"{100 * dif.median():.3f}%; {n_ruim} pregões com diferença acima de 1%."
        )
    base["origem"] = np.where(tem_b3, "B3", "Yahoo")
    base["fechamento"] = base["fechamento"].where(tem_b3, base["bruto_yahoo"])
    base["negocios"] = base["negocios"].where(tem_b3, np.where(base["volume"].fillna(0) > 0, 1, 0))
    diario = base[["data", "fechamento", "negocios", "origem"]].dropna(subset=["fechamento"]).reset_index(drop=True)

    # Saltos de mais de 35% num dia sem desdobramento: possível evento não registrado.
    r = diario["fechamento"].pct_change()
    datas_desd = set(desd["data"]) if not desd.empty else set()
    saltos = diario.loc[(r.abs() > 0.35) & ~diario["data"].isin(datas_desd), "data"]
    if len(saltos):
        notas.append("Saltos de preço acima de 35% sem desdobramento registrado em: "
                     + ", ".join(d.strftime("%d/%m/%Y") for d in saltos) + ".")
    for d, f in zip(desd["data"], desd["fator"]):
        antes = diario.loc[diario["data"] < d, "fechamento"].tail(1)
        depois = diario.loc[diario["data"] >= d, "fechamento"].head(1)
        if len(antes) and len(depois):
            razao = float(antes.iloc[0] / depois.iloc[0])
            if abs(np.log(razao / f)) > np.log(1.5):
                notas.append(f"Desdobramento de {d:%d/%m/%Y} (fator {f:g}) não aparece no preço "
                             f"(queda observada {razao:.2f}×); conferir.")

    prov = yh["proventos"].copy()
    if not prov.empty:
        # O Yahoo divide os proventos pelos desdobramentos posteriores; volta para a base da data ex.
        prov["valor"] = prov["valor"].to_numpy() * fatores_ate(prov["data"], desd, hoje)
        prov = prov[prov["data"] >= pd.Timestamp(f"{ano_inicio}-01-01")].reset_index(drop=True)
    notas.append("Proventos do Yahoo: dividendos e JCP pelo valor bruto, antes do imposto retido.")
    if acoes_por_unidade and acoes_por_unidade != 1.0:
        diario["fechamento"] = diario["fechamento"] / acoes_por_unidade
        if len(prov):
            prov["valor"] = prov["valor"] / acoes_por_unidade
        notas.append(f"{ticker} é unit ({acoes_por_unidade:g} ações): preços e proventos divididos para virar valor por ação.")
    return Precos(diario=diario, desdobramentos=desd, proventos=prov, notas=notas)


def preco_em(diario: pd.DataFrame, d) -> tuple[float, pd.Timestamp | None]:
    x = diario[diario["data"] <= pd.Timestamp(d)]
    if x.empty:
        return np.nan, None
    u = x.iloc[-1]
    return float(u["fechamento"]), pd.Timestamp(u["data"])


def serie_ajustada(diario: pd.DataFrame, desd: pd.DataFrame, base) -> pd.Series:
    """Fechamentos na base de ações da data `base` (só desdobramentos e grupamentos)."""
    f = fatores_ate(diario["data"], desd, base)
    return pd.Series(diario["fechamento"].to_numpy() / f, index=pd.DatetimeIndex(diario["data"]))


def retorno_total(diario: pd.DataFrame, desd: pd.DataFrame, prov: pd.DataFrame, base) -> pd.Series:
    """Índice de retorno com proventos reinvestidos, na escala do preço ajustado em `base`."""
    aj = serie_ajustada(diario, desd, base)
    fator = pd.Series(1.0, index=aj.index)
    if prov is not None and not prov.empty:
        for d, v in zip(prov["data"], prov["valor"]):
            anteriores = diario[diario["data"] < d]
            if anteriores.empty:
                continue
            p_ant = float(anteriores["fechamento"].iloc[-1])
            if p_ant > 0:
                fator[fator.index >= d] *= 1.0 / (1.0 - v / p_ant)
    return aj * fator


def proventos_12m(prov: pd.DataFrame, desd: pd.DataFrame, d) -> float:
    """Proventos por ação com data ex nos 12 meses até d, na base de ações de d."""
    if prov is None or prov.empty:
        return 0.0
    d = pd.Timestamp(d)
    x = prov[(prov["data"] > d - pd.DateOffset(years=1)) & (prov["data"] <= d)]
    total = 0.0
    for de, v in zip(x["data"], x["valor"]):
        total += v / fator_entre(desd, de, d)
    return total
