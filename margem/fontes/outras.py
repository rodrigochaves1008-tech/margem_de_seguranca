"""Tesouro Direto (curvas de juros), FRED e Ipeadata."""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from .. import caminhos, rede

URL_TESOURO = ("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/"
               "resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv")
URL_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
URL_IPEA = "https://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{codigo}')"

TITULOS = {
    "prefixado": ("Tesouro Prefixado", "Tesouro Prefixado com Juros Semestrais"),
    "ipca": ("Tesouro IPCA+", "Tesouro IPCA+ com Juros Semestrais"),
}


def _fresco(p, horas: float) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) / 3600 < horas


def tesouro_taxas(max_idade_horas: float = 20) -> pd.DataFrame:
    destino = caminhos.tratado("tesouro") / "taxas.parquet"
    if _fresco(destino, max_idade_horas):
        return pd.read_parquet(destino)
    bruto = caminhos.bruto("tesouro") / "PrecoTaxaTesouroDireto.csv"
    rede.baixar_arquivo(URL_TESOURO, bruto, max_idade_horas=max_idade_horas, rotulo="taxas do Tesouro Direto")
    df = pd.read_csv(bruto, sep=";", decimal=",", encoding="latin1", dtype=str)
    out = pd.DataFrame({
        "tipo": df["Tipo Titulo"].str.strip(),
        "vencimento": pd.to_datetime(df["Data Vencimento"], format="%d/%m/%Y"),
        "data": pd.to_datetime(df["Data Base"], format="%d/%m/%Y"),
        "taxa": pd.to_numeric(df["Taxa Compra Manha"].str.replace(",", "."), errors="coerce"),
    }).dropna()
    out = out[out["taxa"] > 0]
    out.to_parquet(destino, index=False)
    return out


def tesouro_curva(tipo: str, prazo_anos: float, max_idade_horas: float = 20) -> pd.DataFrame:
    """Taxa interpolada no prazo constante pedido (linear entre os títulos do dia): data, valor.
    Fora da faixa de prazos do dia, usa o título mais próximo se estiver a até 2 anos do prazo."""
    destino = caminhos.tratado("tesouro") / f"curva_{tipo}_{prazo_anos:g}.parquet"
    if _fresco(destino, max_idade_horas):
        return pd.read_parquet(destino)
    t = tesouro_taxas(max_idade_horas)
    t = t[t["tipo"].isin(TITULOS[tipo])].copy()
    t["prazo"] = (t["vencimento"] - t["data"]).dt.days / 365.25
    t = t[t["prazo"] > 0.1]
    datas, valores = [], []
    for d, g in t.groupby("data"):
        g = g.groupby("prazo", as_index=False)["taxa"].mean().sort_values("prazo")
        x, y = g["prazo"].to_numpy(), g["taxa"].to_numpy()
        if x[0] <= prazo_anos <= x[-1]:
            v = float(np.interp(prazo_anos, x, y))
        else:
            i = int(np.argmin(np.abs(x - prazo_anos)))
            v = float(y[i]) if abs(x[i] - prazo_anos) <= 2 else np.nan
        datas.append(d)
        valores.append(v)
    out = pd.DataFrame({"data": datas, "valor": valores}).dropna().sort_values("data").reset_index(drop=True)
    out.to_parquet(destino, index=False)
    return out


def fred(codigo: str, max_idade_horas: float = 20) -> pd.DataFrame:
    destino = caminhos.tratado("fred") / f"{codigo}.parquet"
    if _fresco(destino, max_idade_horas):
        return pd.read_parquet(destino)
    r = rede.get(URL_FRED, params={"id": codigo}, timeout=120)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["data", "valor"]
    df["data"] = pd.to_datetime(df["data"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df.to_parquet(destino, index=False)
    return df


def ipea(codigo: str, max_idade_horas: float = 20) -> pd.DataFrame:
    destino = caminhos.tratado("ipea") / f"{codigo}.parquet"
    if _fresco(destino, max_idade_horas):
        return pd.read_parquet(destino)
    r = rede.get(URL_IPEA.format(codigo=codigo), timeout=120)
    v = pd.DataFrame(r.json()["value"])
    df = pd.DataFrame({
        "data": pd.to_datetime(v["VALDATA"].str[:10]),
        "valor": pd.to_numeric(v["VALVALOR"], errors="coerce"),
    }).dropna().sort_values("data").reset_index(drop=True)
    df.to_parquet(destino, index=False)
    return df
