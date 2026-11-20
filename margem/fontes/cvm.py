"""CVM dados abertos: DFP (anual) e ITR (trimestral) das companhias abertas.

Point-in-time: cada documento entra na data de entrega à CVM (DT_RECEB). Os arquivos abertos
trazem só a versão mais recente de cada documento; quando ela é uma reapresentação (VERSAO > 1),
a data de entrega é a da reapresentação, e os números só entram a partir dela (antes, vale o
documento anterior). Isso é registrado em Limitações.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd

from .. import caminhos, rede

URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{tipo}/DADOS/{tipo_min}_cia_aberta_{ano}.zip"
ANO_INICIAL = {"DFP": 2010, "ITR": 2011}
DEMONSTRACOES = ("BPA", "BPP", "DRE", "DFC_MI")
ESCALA = {"MIL": 1000.0, "UNIDADE": 1.0}


def _zip(tipo: str, ano: int, max_idade_horas: float | None):
    destino = caminhos.bruto("cvm", tipo.lower()) / f"{tipo.lower()}_cia_aberta_{ano}.zip"
    rede.baixar_arquivo(
        URL.format(tipo=tipo, tipo_min=tipo.lower(), ano=ano),
        destino,
        max_idade_horas=max_idade_horas,
        rotulo=f"CVM {tipo} {ano}",
    )
    return destino


def _idade_padrao(ano: int, max_idade_horas: float) -> float:
    hoje = date.today().year
    # A CVM republica arquivos de anos anteriores quando há reapresentações.
    if ano >= hoje - 1:
        return max_idade_horas
    if ano >= hoje - 5:
        return 24 * 7
    return 24 * 30


def _ler_membro(z: zipfile.ZipFile, nome: str, prefixos: tuple[bytes, ...]) -> pd.DataFrame | None:
    if nome not in z.namelist():
        return None
    with z.open(nome) as f:
        cabecalho = f.readline()
        linhas = [l for l in f if l.startswith(prefixos)]
    texto = (cabecalho + b"".join(linhas)).decode("latin1")
    return pd.read_csv(io.StringIO(texto), sep=";", dtype=str)


def extrair(cnpjs: list[str], tipo: str, ano: int, max_idade_horas: float = 12) -> dict[str, pd.DataFrame]:
    """Linhas das companhias pedidas num ano de DFP/ITR, com cache em parquet."""
    z_path = _zip(tipo, ano, _idade_padrao(ano, max_idade_horas))
    pasta = caminhos.tratado("cvm", tipo.lower(), str(ano))
    chave = "_".join(sorted(c.replace(".", "").replace("/", "").replace("-", "") for c in cnpjs))[:120]
    marca = pasta / f"{chave}.ok"
    nomes = ["indice", "capital"] + [d.lower() for d in DEMONSTRACOES]
    if marca.exists() and marca.stat().st_mtime >= z_path.stat().st_mtime:
        return {n: pd.read_parquet(pasta / f"{chave}_{n}.parquet") for n in nomes}
    prefixos = tuple(c.encode("latin1") + b";" for c in cnpjs)
    t = tipo.lower()
    out: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(z_path) as z:
        out["indice"] = _ler_membro(z, f"{t}_cia_aberta_{ano}.csv", prefixos)
        out["capital"] = _ler_membro(z, f"{t}_cia_aberta_composicao_capital_{ano}.csv", prefixos)
        for d in DEMONSTRACOES:
            con = _ler_membro(z, f"{t}_cia_aberta_{d}_con_{ano}.csv", prefixos)
            ind = _ler_membro(z, f"{t}_cia_aberta_{d}_ind_{ano}.csv", prefixos)
            partes = []
            for df, nivel in ((con, "con"), (ind, "ind")):
                if df is not None and not df.empty:
                    df = df.copy()
                    df["NIVEL"] = nivel
                    partes.append(df)
            out[d.lower()] = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    for n in nomes:
        df = out.get(n)
        if df is None:
            df = pd.DataFrame()
        df.to_parquet(pasta / f"{chave}_{n}.parquet", index=False)
        out[n] = df
    marca.touch()
    return out


def carregar(cnpj: str, ano_fim: int | None = None, max_idade_horas: float = 12) -> dict[str, pd.DataFrame]:
    """Todos os DFP e ITR da companhia: documentos, capital e demonstrações (formato longo)."""
    ano_fim = ano_fim or date.today().year
    docs, caps, dems = [], [], []
    for tipo in ("DFP", "ITR"):
        for ano in range(ANO_INICIAL[tipo], ano_fim + 1):
            try:
                partes = extrair([cnpj], tipo, ano, max_idade_horas)
            except Exception as e:  # ano ainda não publicado, por exemplo
                print(f"  aviso: CVM {tipo} {ano} indisponível ({e.__class__.__name__})")
                continue
            idx = partes["indice"]
            if idx is None or idx.empty:
                continue
            idx = idx.assign(TIPO=tipo)
            docs.append(idx)
            if partes["capital"] is not None and not partes["capital"].empty:
                caps.append(partes["capital"].assign(TIPO=tipo))
            for d in DEMONSTRACOES:
                df = partes[d.lower()]
                if df is not None and not df.empty:
                    dems.append(df.assign(TIPO=tipo, DEMONSTRACAO=d))
    documentos = _documentos(docs)
    capital = _capital(caps, documentos)
    demonstracoes = _demonstracoes(dems, documentos)
    return {"documentos": documentos, "capital": capital, "demonstracoes": demonstracoes}


def _documentos(docs: list[pd.DataFrame]) -> pd.DataFrame:
    if not docs:
        return pd.DataFrame(columns=["tipo", "dt_refer", "versao", "divulgacao", "id_doc", "link"])
    d = pd.concat(docs, ignore_index=True)
    out = pd.DataFrame({
        "tipo": d["TIPO"],
        "dt_refer": pd.to_datetime(d["DT_REFER"]),
        "versao": pd.to_numeric(d["VERSAO"]).astype(int),
        "divulgacao": pd.to_datetime(d["DT_RECEB"]),
        "id_doc": d["ID_DOC"],
        "link": d["LINK_DOC"],
    })
    # Se aparecer mais de uma versão do mesmo documento, fica a de maior número.
    out = out.sort_values(["tipo", "dt_refer", "versao"]).drop_duplicates(["tipo", "dt_refer"], keep="last")
    return out.sort_values("divulgacao").reset_index(drop=True)


def _junta_divulgacao(df: pd.DataFrame, documentos: pd.DataFrame) -> pd.DataFrame:
    chave = documentos[["tipo", "dt_refer", "versao", "divulgacao"]]
    return df.merge(chave, on=["tipo", "dt_refer", "versao"], how="inner")


def _capital(caps: list[pd.DataFrame], documentos: pd.DataFrame) -> pd.DataFrame:
    if not caps:
        return pd.DataFrame(columns=["tipo", "dt_refer", "versao", "acoes_total", "acoes_tesouraria",
                                     "acoes_circulacao", "divulgacao"])
    c = pd.concat(caps, ignore_index=True)
    out = pd.DataFrame({
        "tipo": c["TIPO"],
        "dt_refer": pd.to_datetime(c["DT_REFER"]),
        "versao": pd.to_numeric(c["VERSAO"]).astype(int),
        "acoes_total": pd.to_numeric(c["QT_ACAO_TOTAL_CAP_INTEGR"], errors="coerce"),
        "acoes_tesouraria": pd.to_numeric(c["QT_ACAO_TOTAL_TESOURO"], errors="coerce").fillna(0),
    })
    out["acoes_circulacao"] = out["acoes_total"] - out["acoes_tesouraria"]
    return _junta_divulgacao(out, documentos).sort_values("divulgacao").reset_index(drop=True)


URL_FRE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{ano}.zip"


def capital_fre(cnpj: str, ano_fim: int | None = None, max_idade_horas: float = 12) -> pd.DataFrame:
    """Total de ações e free float pelo Formulário de Referência (FRE).

    Os arquivos abertos trazem só a última versão de cada FRE; o dado entra na data de entrega
    dessa versão (conservador). Serve para os anos em que o ITR/DFP não traz a composição do capital.
    """
    ano_fim = ano_fim or date.today().year
    prefixo = (cnpj.encode("latin1") + b";",)
    linhas = []
    for ano in range(2010, ano_fim + 1):
        destino = caminhos.bruto("cvm", "fre") / f"fre_cia_aberta_{ano}.zip"
        try:
            rede.baixar_arquivo(URL_FRE.format(ano=ano), destino,
                                max_idade_horas=_idade_padrao(ano, max_idade_horas), rotulo=f"CVM FRE {ano}")
        except Exception:
            continue
        with zipfile.ZipFile(destino) as z:
            idx = _ler_membro(z, f"fre_cia_aberta_{ano}.csv", prefixo)
            cap = _ler_membro(z, f"fre_cia_aberta_capital_social_{ano}.csv", prefixo)
            dist = _ler_membro(z, f"fre_cia_aberta_distribuicao_capital_{ano}.csv", prefixo)
        if idx is None or idx.empty or cap is None or cap.empty:
            continue
        idx = idx.assign(VERSAO=pd.to_numeric(idx["VERSAO"]))
        integ = cap[cap["Tipo_Capital"].str.contains("Integralizado", case=False, na=False)]
        if integ.empty:
            continue
        versao = int(pd.to_numeric(integ["Versao"]).max())
        receb = idx.loc[idx["VERSAO"] == versao, "DT_RECEB"]
        if receb.empty:
            continue
        reg = {
            "dt_refer": pd.Timestamp(integ["Data_Referencia"].iloc[0]),
            "versao": versao,
            "divulgacao": pd.Timestamp(receb.iloc[0]),
            "acoes_total": float(pd.to_numeric(integ["Quantidade_Total_Acoes"]).iloc[0]),
            "free_float": np.nan,
        }
        if dist is not None and not dist.empty:
            pct = pd.to_numeric(dist["Percentual_Total_Acoes_Circulacao"], errors="coerce")
            if pct.notna().any():
                reg["free_float"] = float(pct.iloc[-1]) / 100.0
        linhas.append(reg)
    return pd.DataFrame(linhas, columns=["dt_refer", "versao", "divulgacao", "acoes_total", "free_float"])


def _demonstracoes(dems: list[pd.DataFrame], documentos: pd.DataFrame) -> pd.DataFrame:
    colunas = ["tipo", "dt_refer", "versao", "demonstracao", "nivel", "ordem", "dt_ini", "dt_fim",
               "conta", "descricao", "valor", "divulgacao"]
    if not dems:
        return pd.DataFrame(columns=colunas)
    d = pd.concat(dems, ignore_index=True)
    escala = d["ESCALA_MOEDA"].map(ESCALA).fillna(1.0)
    out = pd.DataFrame({
        "tipo": d["TIPO"],
        "dt_refer": pd.to_datetime(d["DT_REFER"]),
        "versao": pd.to_numeric(d["VERSAO"]).astype(int),
        "demonstracao": d["DEMONSTRACAO"],
        "nivel": d["NIVEL"],
        "ordem": d["ORDEM_EXERC"].str.upper().str.startswith("Ú").map({True: "ultimo", False: "penultimo"}),
        "dt_ini": pd.to_datetime(d["DT_INI_EXERC"]) if "DT_INI_EXERC" in d else pd.NaT,
        "dt_fim": pd.to_datetime(d["DT_FIM_EXERC"]),
        "conta": d["CD_CONTA"],
        "descricao": d["DS_CONTA"],
        "valor": pd.to_numeric(d["VL_CONTA"], errors="coerce") * escala,
    })
    return _junta_divulgacao(out, documentos)[colunas].reset_index(drop=True)
