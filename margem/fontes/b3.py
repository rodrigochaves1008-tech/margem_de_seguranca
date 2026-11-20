"""Cotações históricas da B3 (arquivos COTAHIST): fechamento oficial, sem ajuste nenhum."""
from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd

from .. import caminhos, rede

URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"


def _eh_zip(p) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(2) == b"PK"
    except OSError:
        return False


def baixar_ano(ano: int) -> "caminhos.Path | None":
    """Baixa o arquivo anual. Devolve None se a B3 recusar (ela bloqueia alguns servidores)."""
    destino = caminhos.bruto("b3", "cotahist") / f"COTAHIST_A{ano}.ZIP"
    ano_corrente = ano >= date.today().year
    if destino.exists() and not _eh_zip(destino):
        destino.unlink()
    try:
        rede.baixar_arquivo(
            URL.format(ano=ano),
            destino,
            max_idade_horas=12 if ano_corrente else None,
            rotulo=f"cotações B3 {ano}",
        )
    except Exception:
        return destino if _eh_zip(destino) else None
    if not _eh_zip(destino):
        destino.unlink(missing_ok=True)
        return None
    return destino


def _ler_zip(caminho) -> pd.DataFrame:
    # Lê linha a linha (o arquivo descompactado passa de 700 MB). Registro 01 = cotação;
    # mercado 010 = à vista.
    with zipfile.ZipFile(caminho) as z, z.open(z.namelist()[0]) as f:
        sel = [l.rstrip(b"\r\n") for l in f if l[:2] == b"01" and l[24:27] == b"010"]
    if not sel:
        return pd.DataFrame()
    txt = b"\n".join(sel).decode("latin1")
    larguras = [(2, 10), (10, 12), (12, 24), (24, 27), (39, 49), (56, 69), (69, 82), (82, 95), (108, 121),
                (147, 152), (152, 170), (170, 188), (210, 217), (230, 242)]
    nomes = ["data", "codbdi", "ticker", "tpmerc", "especi", "abertura", "maxima", "minima", "fechamento",
             "negocios", "quantidade", "volume", "fatcot", "isin"]
    df = pd.read_fwf(io.StringIO(txt), colspecs=larguras, names=nomes, dtype=str)
    df["data"] = pd.to_datetime(df["data"], format="%Y%m%d")
    fat = pd.to_numeric(df["fatcot"], errors="coerce").replace(0, 1).fillna(1)
    for c in ("abertura", "maxima", "minima", "fechamento"):
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0 / fat
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce") / 100.0
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce")
    df["negocios"] = pd.to_numeric(df["negocios"], errors="coerce")
    df["ticker"] = df["ticker"].str.strip()
    return df[["data", "ticker", "codbdi", "especi", "abertura", "maxima", "minima", "fechamento",
               "negocios", "quantidade", "volume", "isin"]]


def cotacoes_ano(ano: int) -> pd.DataFrame | None:
    """Cotações à vista do ano (todas as ações), lidas do cache tratado quando possível."""
    tratado = caminhos.tratado("b3", "cotahist") / f"{ano}.parquet"
    z = baixar_ano(ano)
    if z is None:
        return pd.read_parquet(tratado) if tratado.exists() else None
    if tratado.exists() and tratado.stat().st_mtime >= z.stat().st_mtime:
        return pd.read_parquet(tratado)
    df = _ler_zip(z)
    df.to_parquet(tratado, index=False)
    return df


def precos_brutos(ticker: str, ano_inicio: int, ano_fim: int | None = None) -> tuple[pd.DataFrame, list[int]]:
    """Fechamentos diários sem ajuste do ticker. Devolve (tabela, anos que faltaram)."""
    ano_fim = ano_fim or date.today().year
    partes, faltaram = [], []
    for ano in range(ano_inicio, ano_fim + 1):
        df = cotacoes_ano(ano)
        if df is None or df.empty:
            faltaram.append(ano)
            continue
        partes.append(df[df["ticker"] == ticker.upper()])
    if not partes:
        return pd.DataFrame(columns=["data", "fechamento"]), faltaram
    out = pd.concat(partes, ignore_index=True).sort_values("data")
    # Lote padrão (BDI 02) quando houver mais de um registro no dia.
    out = out.sort_values(["data", "codbdi"]).drop_duplicates("data", keep="first")
    out = out[np.isfinite(out["fechamento"]) & (out["fechamento"] > 0)]
    return out.reset_index(drop=True), faltaram
