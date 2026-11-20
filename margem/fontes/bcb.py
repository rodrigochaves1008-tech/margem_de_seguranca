"""Banco Central: SGS (séries temporais) e Boletim Focus (expectativas de mercado)."""
from __future__ import annotations

import time
from datetime import date
from urllib.parse import quote

import pandas as pd

from .. import caminhos, rede

URL_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
URL_FOCUS = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/{conjunto}"


def _fresco(p, horas: float) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) / 3600 < horas


def sgs(codigo: int, inicio: str = "2000-01-01", max_idade_horas: float = 20) -> pd.DataFrame:
    """Série do SGS: data (de referência), valor. A API limita séries diárias a 10 anos por pedido."""
    destino = caminhos.tratado("bcb", "sgs") / f"{codigo}.parquet"
    if _fresco(destino, max_idade_horas):
        return pd.read_parquet(destino)
    partes = []
    ini = pd.Timestamp(inicio)
    fim_total = pd.Timestamp(date.today())
    while ini <= fim_total:
        fim = min(ini + pd.DateOffset(years=9, months=11), fim_total)
        try:
            r = rede.get(URL_SGS.format(codigo=codigo), params={
                "formato": "json", "dataInicial": ini.strftime("%d/%m/%Y"), "dataFinal": fim.strftime("%d/%m/%Y")})
            dados = r.json()
        except Exception as e:
            if getattr(getattr(e, "response", None), "status_code", None) == 404:
                dados = []
            else:
                raise
        if dados:
            partes.append(pd.DataFrame(dados))
        ini = fim + pd.Timedelta(days=1)
    if not partes:
        df = pd.DataFrame(columns=["data", "valor"])
    else:
        df = pd.concat(partes, ignore_index=True)
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna().drop_duplicates("data", keep="last").sort_values("data").reset_index(drop=True)
    df.to_parquet(destino, index=False)
    return df


def focus(conjunto: str, indicador: str, suavizada: str | None = None, ano_relativo: int | None = None,
          max_idade_horas: float = 20) -> pd.DataFrame:
    """Mediana das expectativas do Focus por data da pesquisa: data, valor."""
    nome = f"{conjunto}_{indicador}_{suavizada or ''}_{ano_relativo if ano_relativo is not None else ''}"
    destino = caminhos.tratado("bcb", "focus") / (nome.replace(" ", "_") + ".parquet")
    if _fresco(destino, max_idade_horas):
        return pd.read_parquet(destino)
    filtro = f"Indicador eq '{indicador}' and baseCalculo eq 0"
    if suavizada:
        filtro += f" and Suavizada eq '{suavizada}'"
    campos = "Data,Mediana" + (",DataReferencia" if ano_relativo is not None else "")
    url = (URL_FOCUS.format(conjunto=conjunto) + "?$filter=" + quote(filtro, safe="'")
           + "&$select=" + campos + "&$format=json")
    r = rede.get(url, timeout=180)
    df = pd.DataFrame(r.json()["value"])
    if df.empty:
        out = pd.DataFrame(columns=["data", "valor"])
    else:
        df["Data"] = pd.to_datetime(df["Data"])
        if ano_relativo is not None:
            ref = pd.to_numeric(df["DataReferencia"], errors="coerce")
            df = df[ref == df["Data"].dt.year + ano_relativo]
        out = (df.rename(columns={"Data": "data", "Mediana": "valor"})[["data", "valor"]]
               .groupby("data", as_index=False)["valor"].last().sort_values("data").reset_index(drop=True))
    out.to_parquet(destino, index=False)
    return out
