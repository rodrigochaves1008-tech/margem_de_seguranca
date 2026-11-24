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


def _limpo(cnpj: str) -> str:
    return cnpj.replace(".", "").replace("/", "").replace("-", "")


def _ler_membro(z: zipfile.ZipFile, nome: str, prefixos: tuple[bytes, ...]) -> pd.DataFrame | None:
    if nome not in z.namelist():
        return None
    with z.open(nome) as f:
        cabecalho = f.readline()
        linhas = [l for l in f if l.startswith(prefixos)]
    texto = (cabecalho + b"".join(linhas)).decode("latin1")
    return pd.read_csv(io.StringIO(texto), sep=";", dtype=str)


_NOMES = ["indice", "capital"] + [d.lower() for d in DEMONSTRACOES]


def _rastreadas() -> set[str]:
    p = caminhos.tratado("cvm") / "empresas_rastreadas.txt"
    return set(p.read_text(encoding="utf-8").split()) if p.exists() else set()


def _rastrear(cnpjs) -> None:
    p = caminhos.tratado("cvm") / "empresas_rastreadas.txt"
    todas = _rastreadas() | set(cnpjs)
    p.write_text("\n".join(sorted(todas)), encoding="utf-8")


def extrair(cnpjs: list[str], tipo: str, ano: int, max_idade_horas: float = 12) -> dict[str, dict[str, pd.DataFrame]]:
    """Linhas de cada companhia pedida num ano de DFP/ITR: {cnpj: {parte: tabela}}.

    Cache por empresa. Quando falta alguma, o arquivo do ano é lido uma vez só para todas as
    empresas que faltam (e para as já rastreadas, se o arquivo da CVM mudou)."""
    z_path = _zip(tipo, ano, _idade_padrao(ano, max_idade_horas))
    pasta = caminhos.tratado("cvm", tipo.lower(), str(ano))
    mt = z_path.stat().st_mtime

    def em_dia(c):
        m = pasta / f"{_limpo(c)}.ok"
        return m.exists() and m.stat().st_mtime >= mt

    faltam = [c for c in cnpjs if not em_dia(c)]
    if faltam:
        alvo = sorted(set(faltam) | {c for c in _rastreadas() if not em_dia(c)})
        prefixos = tuple(c.encode("latin1") + b";" for c in alvo)
        t = tipo.lower()
        bruto: dict[str, pd.DataFrame | None] = {}
        with zipfile.ZipFile(z_path) as z:
            bruto["indice"] = _ler_membro(z, f"{t}_cia_aberta_{ano}.csv", prefixos)
            bruto["capital"] = _ler_membro(z, f"{t}_cia_aberta_composicao_capital_{ano}.csv", prefixos)
            for d in DEMONSTRACOES:
                partes = []
                for nivel in ("con", "ind"):
                    df = _ler_membro(z, f"{t}_cia_aberta_{d}_{nivel}_{ano}.csv", prefixos)
                    if df is not None and not df.empty:
                        partes.append(df.assign(NIVEL=nivel))
                bruto[d.lower()] = pd.concat(partes, ignore_index=True) if partes else None
        for c in alvo:
            for n in _NOMES:
                df = bruto.get(n)
                parte = df[df.iloc[:, 0] == c] if df is not None and len(df) else pd.DataFrame()
                parte.to_parquet(pasta / f"{_limpo(c)}_{n}.parquet", index=False)
            (pasta / f"{_limpo(c)}.ok").touch()
        _rastrear(alvo)
    return {c: {n: pd.read_parquet(pasta / f"{_limpo(c)}_{n}.parquet") for n in _NOMES} for c in cnpjs}


def carregar_varias(cnpjs: list[str], ano_fim: int | None = None, max_idade_horas: float = 12) -> dict[str, dict]:
    """Todos os DFP e ITR de cada companhia: documentos, capital e demonstrações (formato longo)."""
    ano_fim = ano_fim or date.today().year
    acum = {c: {"docs": [], "caps": [], "dems": []} for c in cnpjs}
    for tipo in ("DFP", "ITR"):
        for ano in range(ANO_INICIAL[tipo], ano_fim + 1):
            try:
                por_empresa = extrair(list(cnpjs), tipo, ano, max_idade_horas)
            except Exception as e:  # ano ainda não publicado, por exemplo
                print(f"  aviso: CVM {tipo} {ano} indisponível ({e.__class__.__name__})")
                continue
            for c, partes in por_empresa.items():
                idx = partes["indice"]
                if idx is None or idx.empty:
                    continue
                acum[c]["docs"].append(idx.assign(TIPO=tipo))
                if partes["capital"] is not None and not partes["capital"].empty:
                    acum[c]["caps"].append(partes["capital"].assign(TIPO=tipo))
                for d in DEMONSTRACOES:
                    df = partes[d.lower()]
                    if df is not None and not df.empty:
                        acum[c]["dems"].append(df.assign(TIPO=tipo, DEMONSTRACAO=d))
    out = {}
    for c, a in acum.items():
        documentos = _documentos(a["docs"])
        out[c] = {"documentos": documentos, "capital": _capital(a["caps"], documentos),
                  "demonstracoes": _demonstracoes(a["dems"], documentos)}
    return out


def carregar(cnpj: str, ano_fim: int | None = None, max_idade_horas: float = 12) -> dict[str, pd.DataFrame]:
    return carregar_varias([cnpj], ano_fim, max_idade_horas)[cnpj]


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


def capital_fre_varias(cnpjs: list[str], ano_fim: int | None = None, max_idade_horas: float = 12) -> dict[str, pd.DataFrame]:
    """Total de ações e free float pelo Formulário de Referência (FRE), por empresa.

    Os arquivos abertos trazem só a última versão de cada FRE; o dado entra na data de entrega
    dessa versão (conservador). Serve para os anos em que o ITR/DFP não traz a composição do capital.
    """
    ano_fim = ano_fim or date.today().year
    prefixos = tuple(c.encode("latin1") + b";" for c in cnpjs)
    linhas: dict[str, list] = {c: [] for c in cnpjs}
    for ano in range(2010, ano_fim + 1):
        destino = caminhos.bruto("cvm", "fre") / f"fre_cia_aberta_{ano}.zip"
        try:
            rede.baixar_arquivo(URL_FRE.format(ano=ano), destino,
                                max_idade_horas=_idade_padrao(ano, max_idade_horas), rotulo=f"CVM FRE {ano}")
        except Exception:
            continue
        with zipfile.ZipFile(destino) as z:
            idx_all = _ler_membro(z, f"fre_cia_aberta_{ano}.csv", prefixos)
            cap_all = _ler_membro(z, f"fre_cia_aberta_capital_social_{ano}.csv", prefixos)
            dist_all = _ler_membro(z, f"fre_cia_aberta_distribuicao_capital_{ano}.csv", prefixos)
        if idx_all is None or idx_all.empty or cap_all is None or cap_all.empty:
            continue
        for c in cnpjs:
            idx = idx_all[idx_all.iloc[:, 0] == c].assign(VERSAO=lambda x: pd.to_numeric(x["VERSAO"]))
            cap = cap_all[cap_all.iloc[:, 0] == c]
            integ = cap[cap["Tipo_Capital"].str.contains("Integralizado", case=False, na=False)]
            if idx.empty or integ.empty:
                continue
            versao = int(pd.to_numeric(integ["Versao"]).max())
            receb = idx.loc[idx["VERSAO"] == versao, "DT_RECEB"]
            if receb.empty:
                continue
            reg = {"dt_refer": pd.Timestamp(integ["Data_Referencia"].iloc[0]), "versao": versao,
                   "divulgacao": pd.Timestamp(receb.iloc[0]),
                   "acoes_total": float(pd.to_numeric(integ["Quantidade_Total_Acoes"]).iloc[0]), "free_float": np.nan}
            if dist_all is not None and not dist_all.empty:
                dist = dist_all[dist_all.iloc[:, 0] == c]
                pct = pd.to_numeric(dist.get("Percentual_Total_Acoes_Circulacao"), errors="coerce") if len(dist) else pd.Series(dtype=float)
                if len(pct) and pct.notna().any():
                    reg["free_float"] = float(pct.dropna().iloc[-1]) / 100.0
            linhas[c].append(reg)
    cols = ["dt_refer", "versao", "divulgacao", "acoes_total", "free_float"]
    return {c: pd.DataFrame(v, columns=cols) for c, v in linhas.items()}


def capital_fre(cnpj: str, ano_fim: int | None = None, max_idade_horas: float = 12) -> pd.DataFrame:
    return capital_fre_varias([cnpj], ano_fim, max_idade_horas)[cnpj]


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
