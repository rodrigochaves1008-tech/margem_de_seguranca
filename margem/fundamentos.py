"""Fundamentos da empresa, point-in-time, a partir dos DFP/ITR da CVM.

Para cada data em que um documento foi entregue à CVM, calcula os indicadores com tudo o que
já tinha sido entregue até aquela data (e nada depois). A base mensal usa o último cálculo
com data de divulgação até o fim do mês.

Bancos: usa o nível de demonstração com cobertura completa. No BB, as consolidadas trimestrais
só existem na CVM a partir de 2022; as individuais (padrão Banco Central) cobrem todo o período.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- contas (bancos)

_RE_LUCRO_FINAL = re.compile(r"^Lucro.*Per[ií]odo", re.I)
_RE_LUCRO_EXCLUIR = re.compile(r"antes|Continuadas|Descontinuadas", re.I)
_RE_CONTROLADORA = re.compile(r"Atribu[ií]do a(os)? S[oó]cios da Empresa Controladora", re.I)
_RE_PDD = re.compile(
    r"provis[aã]o (?:p/|para) cr[eé]ditos de liquida[cç][aã]o duvidosa|perda esperada para risco de cr[eé]dito"
    r"|provis[aã]o para perdas em empr[eé]stimos|perdas em empr[eé]stimos|perdas em t[ií]tulos",
    re.I,
)
_RE_SERVICOS = re.compile(r"Presta[cç][aã]o de Servi[cç]os", re.I)
_RE_PESSOAL = re.compile(r"Pessoal", re.I)
_RE_ADMIN = re.compile(r"Administrativas", re.I)
_RE_PL = re.compile(r"^Patrim[oô]nio L[ií]quido", re.I)
_RE_PL_CONTROLADOR = re.compile(r"Atribu[ií]do ao Controlador", re.I)

FLUXOS_BANCO = ["lucro", "resultado_bruto_if", "pdd", "pdd_em_302", "servicos", "pessoal", "administrativas"]
ESTOQUES_BANCO = ["ativo_total", "patrimonio"]


def _nivel_conta(conta: str) -> int:
    return conta.count(".")


def _extrair_dre_banco(linhas: pd.DataFrame, nivel_demo: str) -> dict:
    """Linhas de uma DRE (um documento, um período) → valores das contas que interessam."""
    v = {}
    l1 = linhas[linhas["conta"].map(_nivel_conta) == 1]
    l2 = linhas[linhas["conta"].map(_nivel_conta) == 2]
    cand = l1[l1["descricao"].str.contains(_RE_LUCRO_FINAL) & ~l1["descricao"].str.contains(_RE_LUCRO_EXCLUIR)
              & (l1["conta"] != "3.99")]
    lucro = np.nan
    if not cand.empty:
        final = cand.sort_values("conta").iloc[-1]
        lucro = final["valor"]
        if nivel_demo == "con":
            sub = l2[l2["conta"].str.startswith(final["conta"] + ".") & l2["descricao"].str.contains(_RE_CONTROLADORA)]
            if not sub.empty and sub["valor"].iloc[0] != 0:
                lucro = sub["valor"].iloc[0]
    v["lucro"] = lucro
    rb = linhas[linhas["conta"] == "3.03"]
    v["resultado_bruto_if"] = rb["valor"].iloc[0] if not rb.empty else np.nan
    pdd = l2[l2["descricao"].str.contains(_RE_PDD)]
    v["pdd"] = pdd["valor"].sum() if not pdd.empty else np.nan
    v["pdd_em_302"] = pdd[pdd["conta"].str.startswith("3.02.")]["valor"].sum() if not pdd.empty else 0.0
    sob_304 = l2[l2["conta"].str.startswith("3.04.")]
    for nome, rx in (("servicos", _RE_SERVICOS), ("pessoal", _RE_PESSOAL), ("administrativas", _RE_ADMIN)):
        x = sob_304[sob_304["descricao"].str.contains(rx)]
        v[nome] = x["valor"].sum() if not x.empty else np.nan
    return v


def _extrair_bp_banco(bpa: pd.DataFrame, bpp: pd.DataFrame, nivel_demo: str) -> dict:
    v = {}
    a = bpa[bpa["conta"] == "1"]
    v["ativo_total"] = a["valor"].iloc[0] if not a.empty else np.nan
    l1 = bpp[bpp["conta"].map(_nivel_conta) == 1]
    pl = l1[l1["descricao"].str.contains(_RE_PL)]
    val = np.nan
    if not pl.empty:
        linha = pl.sort_values("conta").iloc[-1]
        val = linha["valor"]
        if nivel_demo == "con":
            l2 = bpp[bpp["conta"].str.startswith(linha["conta"] + ".") & bpp["descricao"].str.contains(_RE_PL_CONTROLADOR)]
            if not l2.empty and l2["valor"].iloc[0] != 0:
                val = l2["valor"].iloc[0]
    v["patrimonio"] = val
    return v


# ---------------------------------------------------------------- escolha do nível

def escolher_nivel(dem: pd.DataFrame, documentos: pd.DataFrame) -> str:
    """'con' se as consolidadas cobrem quase todos os documentos; senão 'ind'."""
    total = documentos[["tipo", "dt_refer"]].drop_duplicates().shape[0]
    cob = {}
    for nv in ("con", "ind"):
        x = dem[(dem["nivel"] == nv) & (dem["demonstracao"] == "DRE")][["tipo", "dt_refer"]].drop_duplicates()
        cob[nv] = len(x) / total if total else 0
    return "con" if cob["con"] >= 0.95 else "ind"


# ---------------------------------------------------------------- tabelas por período

@dataclass
class Periodos:
    fluxos: pd.DataFrame     # doc, divulgacao, dt_ini, dt_fim, métricas de fluxo
    estoques: pd.DataFrame   # doc, divulgacao, dt_fim, métricas de estoque
    capital: pd.DataFrame    # dt_refer, divulgacao, acoes_circulacao
    nivel: str


def montar_periodos(cvm: dict, contabilidade: str = "banco") -> Periodos:
    if contabilidade != "banco":
        raise NotImplementedError("Fundamentos de empresas não financeiras entram na Fase 2.")
    dem = cvm["demonstracoes"]
    docs = cvm["documentos"]
    nivel = escolher_nivel(dem, docs)
    dem = dem[dem["nivel"] == nivel]
    chave = ["tipo", "dt_refer", "versao", "divulgacao", "ordem"]

    fl = []
    dre = dem[dem["demonstracao"] == "DRE"]
    for k, linhas in dre.groupby(chave + ["dt_ini", "dt_fim"]):
        reg = dict(zip(chave + ["dt_ini", "dt_fim"], k))
        reg.update(_extrair_dre_banco(linhas, nivel))
        fl.append(reg)
    fluxos = _limpar_comparativos(pd.DataFrame(fl), "lucro")

    es = []
    bpa = dem[dem["demonstracao"] == "BPA"]
    bpp = dem[dem["demonstracao"] == "BPP"]
    for k, a in bpa.groupby(chave + ["dt_fim"]):
        p = bpp[(bpp["tipo"] == k[0]) & (bpp["dt_refer"] == k[1]) & (bpp["versao"] == k[2]) & (bpp["dt_fim"] == k[5])]
        reg = dict(zip(chave + ["dt_fim"], k))
        reg.update(_extrair_bp_banco(a, p, nivel))
        es.append(reg)
    estoques = _limpar_comparativos(pd.DataFrame(es), "ativo_total")

    cap = _capital(cvm)
    return Periodos(fluxos=fluxos, estoques=estoques, capital=cap, nivel=nivel)


def _limpar_comparativos(df: pd.DataFrame, metrica: str) -> pd.DataFrame:
    """Comparativos do ano anterior zerados (como nos documentos de 2025 do BB) contam como ausentes.
    Preferência na escolha: números do próprio documento do período (ordem 'ultimo')."""
    if df.empty:
        return df
    zerado = (df["ordem"] == "penultimo") & ((df[metrica] == 0) | df[metrica].isna())
    df = df[~zerado].copy()
    df["prioridade"] = (df["ordem"] == "ultimo").astype(int)
    return df.reset_index(drop=True)


def _capital(cvm: dict) -> pd.DataFrame:
    """Ações em circulação: composição do capital do ITR/DFP (com tesouraria) e, antes dela,
    o total de ações do Formulário de Referência (sem tesouraria, que fica como zero)."""
    partes = []
    c = cvm.get("capital")
    if c is not None and not c.empty:
        partes.append(c[["dt_refer", "divulgacao", "acoes_circulacao", "acoes_total"]].assign(origem="ITR/DFP"))
    fre = cvm.get("capital_fre")
    if fre is not None and not fre.empty:
        partes.append(pd.DataFrame({
            "dt_refer": fre["dt_refer"], "divulgacao": fre["divulgacao"],
            "acoes_circulacao": fre["acoes_total"], "acoes_total": fre["acoes_total"], "origem": "FRE",
        }))
    if not partes:
        return pd.DataFrame(columns=["dt_refer", "divulgacao", "acoes_circulacao", "acoes_total", "origem"])
    return pd.concat(partes, ignore_index=True).sort_values(["divulgacao", "origem"]).reset_index(drop=True)


# ---------------------------------------------------------------- fotografia numa data

def _fim_trimestre(d: pd.Timestamp) -> pd.Timestamp:
    return (d + pd.offsets.QuarterEnd(0)).normalize()


def _trimestres_atras(q: pd.Timestamp, n: int) -> pd.Timestamp:
    return (q - pd.offsets.QuarterEnd(n)).normalize()


def _valor_trimestral(fl: pd.DataFrame, metrica: str, q: pd.Timestamp) -> float:
    """Valor de 3 meses do trimestre que termina em q, com o documento mais recente que o informa."""
    inicio_tri = pd.Timestamp(year=q.year, month=q.month - 2, day=1)
    inicio_ano = pd.Timestamp(year=q.year, month=1, day=1)
    direto = fl[(fl["dt_fim"] == q) & (fl["dt_ini"] == inicio_tri)]
    if not direto.empty:
        return _mais_confiavel(direto)[metrica]
    ytd = fl[(fl["dt_fim"] == q) & (fl["dt_ini"] == inicio_ano)]
    if ytd.empty:
        return np.nan
    v = _mais_confiavel(ytd)[metrica]
    if q.month == 3:
        return v
    q_ant = _trimestres_atras(q, 1)
    ytd_ant = fl[(fl["dt_fim"] == q_ant) & (fl["dt_ini"] == inicio_ano)]
    if ytd_ant.empty:
        return np.nan
    return v - _mais_confiavel(ytd_ant)[metrica]


def _mais_confiavel(x: pd.DataFrame) -> pd.Series:
    """Números do próprio documento do período; entre eles, a entrega mais recente."""
    return x.sort_values(["prioridade", "divulgacao"]).iloc[-1]


def _ltm(fl: pd.DataFrame, metrica: str, q: pd.Timestamp) -> float:
    vals = [_valor_trimestral(fl, metrica, _trimestres_atras(q, i)) for i in range(4)]
    return float(np.sum(vals)) if all(np.isfinite(vals)) else np.nan


def _estoque(es: pd.DataFrame, metrica: str, q: pd.Timestamp) -> float:
    x = es[es["dt_fim"] == q]
    if x.empty:
        return np.nan
    return float(_mais_confiavel(x)[metrica])


def _media(a: float, b: float) -> float:
    if np.isfinite(a) and np.isfinite(b):
        return (a + b) / 2
    return a


def _cresc(a: float, b: float) -> float:
    if np.isfinite(a) and np.isfinite(b) and b > 0 and a > 0:
        return a / b - 1
    return np.nan


def fotografia(per: Periodos, data: pd.Timestamp) -> dict | None:
    """Indicadores com tudo o que foi entregue à CVM até `data` (inclusive)."""
    fl = per.fluxos[per.fluxos["divulgacao"] <= data]
    es = per.estoques[per.estoques["divulgacao"] <= data]
    cap = per.capital[per.capital["divulgacao"] <= data]
    if fl.empty or es.empty:
        return None
    q = es["dt_fim"].max()  # último período com balanço conhecido
    r: dict = {"dt_refer": q}
    lucro = _ltm(fl, "lucro", q)
    q_ltm = q
    if not np.isfinite(lucro):
        # O último DFP pode ainda não ter entrado (reapresentação): usa o último 12 meses completo.
        for i in range(1, 4):
            qi = _trimestres_atras(q, i)
            lucro = _ltm(fl, "lucro", qi)
            if np.isfinite(lucro):
                q_ltm = qi
                break
    r["dt_refer_12m"] = q_ltm if np.isfinite(lucro) else pd.NaT
    r["lucro_12m"] = lucro
    lucro_1a = _ltm(fl, "lucro", _trimestres_atras(q_ltm, 4))
    lucro_3a = _ltm(fl, "lucro", _trimestres_atras(q_ltm, 12))
    rb = _ltm(fl, "resultado_bruto_if", q_ltm)
    pdd = _ltm(fl, "pdd", q_ltm)
    pdd302 = _ltm(fl, "pdd_em_302", q_ltm)
    serv = _ltm(fl, "servicos", q_ltm)
    pes = _ltm(fl, "pessoal", q_ltm)
    adm = _ltm(fl, "administrativas", q_ltm)

    pl = _estoque(es, "patrimonio", q)
    pl_1a = _estoque(es, "patrimonio", _trimestres_atras(q, 4))
    at = _estoque(es, "ativo_total", q)
    at_1a = _estoque(es, "ativo_total", _trimestres_atras(q, 4))
    pl_ltm = _media(_estoque(es, "patrimonio", q_ltm), _estoque(es, "patrimonio", _trimestres_atras(q_ltm, 4)))
    at_ltm = _media(_estoque(es, "ativo_total", q_ltm), _estoque(es, "ativo_total", _trimestres_atras(q_ltm, 4)))

    r["patrimonio"] = pl
    r["ativo_total"] = at
    r["log_lucro_12m"] = np.log(lucro) if np.isfinite(lucro) and lucro > 0 else np.nan
    r["log_patrimonio"] = np.log(pl) if np.isfinite(pl) and pl > 0 else np.nan
    r["roe"] = lucro / pl_ltm if np.isfinite(pl_ltm) and pl_ltm > 0 else np.nan
    r["roa"] = lucro / at_ltm if np.isfinite(at_ltm) and at_ltm > 0 else np.nan
    r["cresc_lucro_12m"] = _cresc(lucro, lucro_1a)
    c3 = _cresc(lucro, lucro_3a)
    r["cresc_lucro_3a"] = (1 + c3) ** (1 / 3) - 1 if np.isfinite(c3) else np.nan
    margem_antes_pdd = rb - pdd302 if np.isfinite(rb) else np.nan
    r["margem_financeira"] = margem_antes_pdd / at_ltm if np.isfinite(at_ltm) and at_ltm > 0 else np.nan
    r["custo_credito"] = -pdd / at_ltm if np.isfinite(pdd) and np.isfinite(at_ltm) and at_ltm > 0 else np.nan
    receitas = margem_antes_pdd + (serv if np.isfinite(serv) else 0.0)
    despesas = -((pes if np.isfinite(pes) else 0.0) + (adm if np.isfinite(adm) else 0.0))
    r["eficiencia"] = despesas / receitas if receitas and np.isfinite(receitas) and receitas > 0 and despesas > 0 else np.nan
    r["cresc_ativo_12m"] = _cresc(at, at_1a)
    r["alavancagem"] = at / pl if np.isfinite(pl) and pl > 0 else np.nan
    cap = cap[cap["acoes_circulacao"] > 0]
    if not cap.empty:
        # A composição do capital do ITR/DFP (desconta a tesouraria) tem preferência; o FRE só
        # é usado enquanto o ITR/DFP ainda não traz esse dado (no BB, antes de 2020).
        itr = cap[cap["origem"] == "ITR/DFP"]
        c = (itr if not itr.empty else cap).sort_values(["divulgacao", "dt_refer"]).iloc[-1]
        r["acoes_circulacao"] = float(c["acoes_circulacao"])
        r["dt_refer_acoes"] = c["dt_refer"]
        r["origem_acoes"] = c["origem"]
    else:
        r["acoes_circulacao"] = np.nan
        r["dt_refer_acoes"] = pd.NaT
        r["origem_acoes"] = None
    return r


def serie_fundamentos(per: Periodos) -> pd.DataFrame:
    """Uma linha por data de entrega de documento, com os indicadores conhecidos naquela data."""
    datas = sorted(set(per.fluxos["divulgacao"]) | set(per.estoques["divulgacao"]) | set(per.capital["divulgacao"]))
    linhas = []
    for d in datas:
        f = fotografia(per, pd.Timestamp(d))
        if f is not None:
            f["divulgacao"] = pd.Timestamp(d)
            linhas.append(f)
    return pd.DataFrame(linhas)


PREDITORES_EMPRESA = {
    "log_lucro_12m": "Lucro dos últimos 12 meses (log)",
    "log_patrimonio": "Patrimônio líquido (log)",
    "roe": "ROE (lucro ÷ patrimônio)",
    "roa": "ROA (lucro ÷ ativo)",
    "cresc_lucro_12m": "Crescimento do lucro em 12 meses",
    "cresc_lucro_3a": "Crescimento do lucro em 3 anos (ao ano)",
    "eficiencia": "Índice de eficiência (despesas ÷ receitas)",
    "custo_credito": "Custo do crédito (PDD ÷ ativo)",
    "margem_financeira": "Margem financeira antes da PDD (÷ ativo)",
    "cresc_ativo_12m": "Crescimento do ativo em 12 meses",
    "alavancagem": "Alavancagem (ativo ÷ patrimônio)",
    "payout": "Payout (proventos ÷ lucro)",
}

EXPLICACAO_EMPRESA = {
    "log_lucro_12m": "o tamanho do lucro dos últimos 12 meses; lucro maior sustenta um valor de mercado maior",
    "log_patrimonio": "o tamanho do patrimônio do banco",
    "roe": "quanto o banco lucra para cada real de patrimônio; é a medida central de rentabilidade de um banco",
    "roa": "quanto o banco lucra para cada real de ativos",
    "cresc_lucro_12m": "se o lucro está crescendo ou caindo em relação a um ano antes",
    "cresc_lucro_3a": "a tendência do lucro em 3 anos",
    "eficiencia": "quanto das receitas vai para despesas de pessoal e administrativas; quanto menor, mais eficiente",
    "custo_credito": "quanto o banco provisiona para calotes em relação ao tamanho do balanço",
    "margem_financeira": "o ganho com juros antes das perdas com calote, em relação ao tamanho do balanço",
    "cresc_ativo_12m": "o ritmo de crescimento do balanço",
    "alavancagem": "quantas vezes os ativos superam o patrimônio",
    "payout": "a fatia do lucro devolvida aos acionistas como dividendos e JCP",
}
