"""Cadastro da CVM: ticker → empresa (CNPJ), setor de atividade e controle acionário.

- Tickers: FCA (Formulário Cadastral), tabela de valores mobiliários.
- Setor e controle: cadastro de companhias abertas (cad_cia_aberta.csv).
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date

import pandas as pd

from .. import caminhos, rede

URL_CAD = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"
URL_FCA = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_{ano}.zip"


def _fresco(p, horas: float) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) / 3600 < horas


def cadastro(max_idade_horas: float = 24 * 7) -> pd.DataFrame:
    destino = caminhos.tratado("cvm") / "cadastro.parquet"
    if _fresco(destino, max_idade_horas):
        return pd.read_parquet(destino)
    bruto = caminhos.bruto("cvm") / "cad_cia_aberta.csv"
    rede.baixar_arquivo(URL_CAD, bruto, max_idade_horas=max_idade_horas, rotulo="cadastro da CVM")
    df = pd.read_csv(bruto, sep=";", encoding="latin1", dtype=str)
    out = pd.DataFrame({
        "cnpj": df["CNPJ_CIA"], "nome": df["DENOM_SOCIAL"], "nome_comercial": df["DENOM_COMERC"],
        "codigo_cvm": df["CD_CVM"], "setor_cvm": df["SETOR_ATIV"].fillna(""), "situacao": df["SIT"],
        "controle": df["CONTROLE_ACIONARIO"].fillna(""), "cancelamento": df["DT_CANCEL"],
    })
    out.to_parquet(destino, index=False)
    return out


def tickers(max_idade_horas: float = 24 * 7) -> pd.DataFrame:
    """ticker, cnpj, nome, valor mobiliário, início e fim de negociação (FCA mais recente)."""
    destino = caminhos.tratado("cvm") / "tickers.parquet"
    if _fresco(destino, max_idade_horas):
        return pd.read_parquet(destino)
    partes = []
    ano = date.today().year
    for a in (ano - 1, ano):
        z = caminhos.bruto("cvm", "fca") / f"fca_cia_aberta_{a}.zip"
        try:
            rede.baixar_arquivo(URL_FCA.format(ano=a), z, max_idade_horas=max_idade_horas, rotulo=f"CVM FCA {a}")
        except Exception:
            continue
        with zipfile.ZipFile(z) as zz:
            nome = f"fca_cia_aberta_valor_mobiliario_{a}.csv"
            if nome not in zz.namelist():
                continue
            df = pd.read_csv(io.BytesIO(zz.read(nome)), sep=";", encoding="latin1", dtype=str)
        partes.append(df.assign(_ano=a))
    df = pd.concat(partes, ignore_index=True)
    df = df[df["Codigo_Negociacao"].notna() & (df["Mercado"].fillna("").str.contains("Bolsa"))]
    out = pd.DataFrame({
        "ticker": df["Codigo_Negociacao"].str.strip().str.upper(), "cnpj": df["CNPJ_Companhia"],
        "nome": df["Nome_Empresarial"], "valor_mobiliario": df["Valor_Mobiliario"],
        "inicio": pd.to_datetime(df["Data_Inicio_Negociacao"], errors="coerce"),
        "fim": pd.to_datetime(df["Data_Fim_Negociacao"], errors="coerce"), "_ano": df["_ano"],
        "acoes_por_unidade": [_acoes_por_unidade(v, c) for v, c in zip(df["Valor_Mobiliario"], df["Composicao_BDR_Unit"])],
    })
    out = out.sort_values(["ticker", "_ano"]).drop_duplicates("ticker", keep="last").drop(columns="_ano")
    out.to_parquet(destino, index=False)
    return out.reset_index(drop=True)


def _acoes_por_unidade(valor_mobiliario, composicao) -> float:
    """Units: soma das ações na composição ('1 ON + 2 PN' → 3). Demais: 1."""
    import re

    if not isinstance(valor_mobiliario, str) or "unit" not in valor_mobiliario.lower():
        return 1.0
    nums = [int(x) for x in re.findall(r"\d+", str(composicao or ""))]
    return float(sum(nums)) if nums else 1.0


def resolver(ticker: str) -> dict | None:
    """Empresa do ticker pela CVM: cnpj, nome, setor_cvm, controle. None se não achar."""
    t = ticker.upper()
    tk = tickers()
    x = tk[tk["ticker"] == t]
    if x.empty:
        return None
    r = x.iloc[0]
    cad = cadastro()
    c = cad[(cad["cnpj"] == r["cnpj"])]
    c = c[c["situacao"].fillna("") == "ATIVO"] if (c["situacao"].fillna("") == "ATIVO").any() else c
    info = {"ticker": t, "cnpj": r["cnpj"], "nome": r["nome"], "setor_cvm": "", "controle": "",
            "negociacao_encerrada": bool(pd.notna(r["fim"])),
            "acoes_por_unidade": float(r.get("acoes_por_unidade", 1.0) or 1.0)}
    if not c.empty:
        info["nome"] = (c.iloc[0]["nome_comercial"] or c.iloc[0]["nome"] or r["nome"]).strip()
        info["setor_cvm"] = c.iloc[0]["setor_cvm"]
        info["controle"] = c.iloc[0]["controle"]
    return info
