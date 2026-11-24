"""Base mensal point-in-time: uma linha por fim de mês (e a data da análise), cada uma montada só
com o que já tinha sido divulgado naquela data.

Alvos (todos em log e independentes da base de ações):
- log_valor_mercado = log(preço × ações em circulação)
- log_pl = log(valor de mercado ÷ lucro de 12 meses)
- log_pvp = log(valor de mercado ÷ patrimônio)
- log_ev_ebitda = log((valor de mercado + dívida líquida + minoritários) ÷ EBITDA de 12 meses)
O preço justo por ação sai dividindo o valor de mercado estimado pelas ações da data.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, fundamentos, macro, precos
from .fontes import cvm

ALVOS = ("log_valor_mercado", "log_pl", "log_pvp", "log_ev_ebitda")


@dataclass
class DadosBrutos:
    ticker: str
    info: dict
    setor: dict
    precos: precos.Precos
    cvm: dict
    series: dict[str, pd.DataFrame]
    ids_series: list[str]
    notas: list[str] = field(default_factory=list)


def carregar(ticker: str) -> DadosBrutos:
    info = config.info_ativo(ticker)
    setor = config.info_setor(info["setor"])
    if not setor.get("ativo", False):
        raise RuntimeError(
            f"O setor '{setor.get('nome', info['setor'])}' ainda não está ativo no mapa de preditores. "
            "Os preditores precisam ser propostos e aprovados antes da rodada."
        )
    c = config.cfg()
    ini = pd.Timestamp(c["modelo"]["inicio_dados"]).year
    notas: list[str] = []
    print(f"  preços e proventos de {ticker}...", flush=True)
    pr = precos.carregar(ticker, ano_inicio=ini, acoes_por_unidade=float(info.get("acoes_por_unidade", 1.0) or 1.0))
    notas += pr.notas
    print("  balanços da CVM...", flush=True)
    dados_cvm = cvm.carregar(info["cnpj"])
    dados_cvm["capital_fre"] = cvm.capital_fre(info["cnpj"])
    reap = dados_cvm["documentos"][dados_cvm["documentos"]["versao"] > 1]
    if len(reap):
        notas.append(
            f"{len(reap)} documentos da CVM foram reapresentados; os arquivos abertos só trazem a última "
            "versão, que entra na data da reapresentação (antes dela, vale o documento anterior)."
        )
    print("  séries de macro, exterior e setor...", flush=True)
    ids = macro.ids_necessarios(setor)
    series, faltas = macro.carregar_todas(ids)
    notas += faltas
    return DadosBrutos(ticker=ticker.upper(), info=info, setor=setor, precos=pr, cvm=dados_cvm,
                       series=series, ids_series=ids, notas=notas)


def carregar_pares(dados: DadosBrutos, progresso=print) -> tuple[dict[str, "DadosBrutos"], list[str]]:
    """Dados dos pares do setor (config/setores.yaml, 'pares'), com as mesmas séries de macro e setor."""
    pares = [p.upper() for p in (dados.setor.get("pares") or []) if p.upper() != dados.ticker]
    if not pares:
        return {}, []
    c = config.cfg()
    ini = pd.Timestamp(c["modelo"]["inicio_dados"]).year
    notas, infos = [], {}
    for p in pares:
        try:
            infos[p] = config.info_ativo(p)
        except KeyError as e:
            notas.append(f"Par {p} ignorado: {e}")
    if progresso:
        progresso(f"  pares do setor: {', '.join(infos)}")
    cnpjs = sorted({i["cnpj"] for i in infos.values()})
    cvms = cvm.carregar_varias(cnpjs)
    fres = cvm.capital_fre_varias(cnpjs)
    out = {}
    for p, info in infos.items():
        try:
            pr = precos.carregar(p, ano_inicio=ini, acoes_por_unidade=float(info.get("acoes_por_unidade", 1.0) or 1.0))
        except Exception as e:
            notas.append(f"Par {p} sem preços ({e.__class__.__name__}); ignorado.")
            continue
        d = dict(cvms[info["cnpj"]])
        d["capital_fre"] = fres[info["cnpj"]]
        out[p] = DadosBrutos(ticker=p, info=info, setor=dados.setor, precos=pr, cvm=d, series=dados.series,
                             ids_series=dados.ids_series, notas=[])
    return out, notas


def datas_da_base(dados: DadosBrutos, ate: pd.Timestamp) -> list[pd.Timestamp]:
    """Fins de mês desde o início dos dados até `ate`, mais a própria data `ate` se não for fim de mês."""
    ini = pd.Timestamp(config.cfg()["modelo"]["inicio_dados"])
    ate = pd.Timestamp(ate).normalize()
    meses = list(pd.date_range(ini, ate, freq="ME"))
    if not meses or meses[-1] != ate:
        meses.append(ate)
    return meses


def montar(dados: DadosBrutos, datas: list[pd.Timestamp]) -> pd.DataFrame:
    contab = dados.setor.get("contabilidade", "geral")
    per = fundamentos.montar_periodos(dados.cvm, contab)
    fund = fundamentos.serie_fundamentos(per)
    diario = dados.precos.diario
    desd = dados.precos.desdobramentos
    prov = dados.precos.proventos

    linhas = []
    for d in datas:
        r: dict = {"data": d}
        p, dp = precos.preco_em(diario, d)
        r["preco"] = p
        r["data_preco"] = dp
        recentes = diario[(diario["data"] <= d)].tail(5)
        r["pregoes_sem_negocio"] = int((recentes["negocios"].fillna(0) <= 0).sum()) if len(recentes) else 5
        f = fund[fund["divulgacao"] <= d]
        if not f.empty:
            fr = f.iloc[-1].to_dict()
            for k, v in fr.items():
                if k not in ("divulgacao",):
                    r[k] = v
            r["divulgacao_balanco"] = fr["divulgacao"]
            if np.isfinite(fr.get("acoes_circulacao", np.nan)):
                r["acoes"] = fr["acoes_circulacao"] * precos.fator_entre(desd, fr["dt_refer_acoes"], d)
        acoes = r.get("acoes", np.nan)
        r["valor_mercado"] = p * acoes if np.isfinite(acoes) else np.nan
        dpa = precos.proventos_12m(prov, desd, d)
        r["proventos_12m"] = dpa
        lucro = r.get("lucro_12m", np.nan)
        r["payout"] = dpa * acoes / lucro if np.isfinite(acoes) and np.isfinite(lucro) and lucro > 0 else np.nan
        vm = r["valor_mercado"]
        pl = r.get("patrimonio", np.nan)
        r["log_valor_mercado"] = np.log(vm) if np.isfinite(vm) and vm > 0 else np.nan
        r["log_pl"] = np.log(vm / lucro) if np.isfinite(vm) and np.isfinite(lucro) and lucro > 0 else np.nan
        r["log_pvp"] = np.log(vm / pl) if np.isfinite(vm) and np.isfinite(pl) and pl > 0 else np.nan
        ebitda = r.get("ebitda_12m", np.nan)
        dl = r.get("divida_liquida", np.nan)
        ev = vm + dl + r.get("minoritarios", 0.0) if np.isfinite(vm) and np.isfinite(dl) else np.nan
        r["valor_firma"] = ev
        r["log_ev_ebitda"] = (np.log(ev / ebitda) if np.isfinite(ev) and np.isfinite(ebitda) and ebitda > 0 and ev > 0
                              else np.nan)
        linhas.append(r)
    base = pd.DataFrame(linhas).set_index("data")
    # Emissões e recompras: ações hoje contra 12 meses antes, na mesma base (sem desdobramentos).
    var = []
    for d in base.index:
        ant = d - pd.DateOffset(years=1)
        a0 = base["acoes"].get(d, np.nan) if "acoes" in base else np.nan
        x = base[(base.index <= ant)]
        if len(x) and "acoes" in x and np.isfinite(x["acoes"].iloc[-1]) and np.isfinite(a0):
            a1 = x["acoes"].iloc[-1] * precos.fator_entre(desd, x.index[-1], d)
            var.append(a0 / a1 - 1 if a1 > 0 else np.nan)
        else:
            var.append(np.nan)
    base["variacao_acoes_12m"] = var
    tab = macro.tabela_pit(dados.series, dados.ids_series, datas)
    base = base.join(tab)
    return base


def preditores_do_setor(setor: dict) -> dict[str, list[str]]:
    p = setor.get("preditores", {})
    return {fam: list(p.get(fam) or []) for fam in ("empresa", "setor", "macro_brasil", "exterior")}


def familia_de(nome: str, setor: dict) -> str:
    for fam, lista in preditores_do_setor(setor).items():
        if nome in lista:
            return fam
    return "outro"


def corromper_depois(dados: DadosBrutos, corte: pd.Timestamp, semente: int) -> DadosBrutos:
    """Cópia dos dados com tudo o que foi divulgado depois de `corte` trocado por valores aleatórios
    (preços, proventos, desdobramentos, balanços, macro). Usada no teste de vazamento."""
    rng = np.random.default_rng(semente)
    d = copy.deepcopy(dados)
    corte = pd.Timestamp(corte)

    def ruido(x: pd.Series) -> np.ndarray:
        return x.to_numpy(dtype=float) * rng.uniform(0.2, 5.0, size=len(x))

    def flutuante(df: pd.DataFrame, cols) -> None:
        for col in cols:
            df[col] = df[col].astype(float)

    pr = d.precos
    flutuante(pr.diario, ["fechamento"])
    flutuante(pr.proventos, ["valor"])
    flutuante(pr.desdobramentos, ["fator"])
    m = pr.diario["data"] > corte
    pr.diario.loc[m, "fechamento"] = ruido(pr.diario.loc[m, "fechamento"])
    m = pr.proventos["data"] > corte
    pr.proventos.loc[m, "valor"] = ruido(pr.proventos.loc[m, "valor"])
    m = pr.desdobramentos["data"] > corte
    pr.desdobramentos.loc[m, "fator"] = rng.choice([0.1, 0.5, 2.0, 3.0, 10.0], size=int(m.sum()))
    dem = d.cvm["demonstracoes"]
    flutuante(dem, ["valor"])
    m = dem["divulgacao"] > corte
    dem.loc[m, "valor"] = ruido(dem.loc[m, "valor"])
    for k in ("capital",):
        c = d.cvm[k]
        flutuante(c, ["acoes_circulacao", "acoes_total"])
        m = c["divulgacao"] > corte
        for col in ("acoes_circulacao", "acoes_total"):
            c.loc[m, col] = ruido(c.loc[m, col])
    fre = d.cvm.get("capital_fre")
    if fre is not None and not fre.empty:
        flutuante(fre, ["acoes_total", "free_float"])
        m = fre["divulgacao"] > corte
        fre.loc[m, "acoes_total"] = ruido(fre.loc[m, "acoes_total"])
        fre.loc[m, "free_float"] = rng.uniform(0.05, 0.95, size=int(m.sum()))
    for sid, s in d.series.items():
        flutuante(s, ["valor"])
        m = s["divulgacao"] > corte
        s.loc[m, "valor"] = ruido(s.loc[m, "valor"])
    return d
