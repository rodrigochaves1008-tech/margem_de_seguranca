"""Ranking dos ativos analisados e atualização rápida.

Atualização rápida: busca os últimos preços e os proventos, desdobramentos e grupamentos, confere
se saiu ITR ou DFP novo, recalcula margens e marcas e refaz a tela de cada ativo e a página do
ranking dos ativos. Não treina modelos, não baixa fundamentos nem macro e não grava no histórico.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import caminhos, config, formatos, graficos, historico
from . import precos as P
from .fontes import cvm, yahoo
from .saida import paginas


def faixa_da_margem(m: float) -> tuple[str, str]:
    f = config.cfg()["faixas_margem"]
    if not np.isfinite(m):
        return "—", "neu"
    if m > f["desconto_forte"]:
        return "desconto forte", "pos"
    if m > f["desconto_moderado"]:
        return "desconto moderado", "pos"
    if m >= f["premio_moderado"]:
        return "perto do preço justo", "neu"
    if m >= f["premio_forte"]:
        return "prêmio moderado", "neg"
    return "prêmio forte", "neg"


# ------------------------------------------------------------------ dados guardados de cada ativo

def ler_ultima_rodada(ticker: str) -> dict | None:
    ur = caminhos.pasta_ultima_rodada(ticker)
    p = ur / "analise.json"
    if not p.exists():
        return None
    out = {"analise": json.loads(p.read_text(encoding="utf-8"))}
    for nome in ("walkforward", "precos", "desdobramentos", "proventos", "preditores"):
        q = ur / f"{nome}.parquet"
        out[nome] = pd.read_parquet(q) if q.exists() else None
    q = ur / "eventos.json"
    out["eventos"] = json.loads(q.read_text(encoding="utf-8")) if q.exists() else []
    return out


def _atualizar_precos(ticker: str, ur: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    """Acrescenta pregões, proventos e desdobramentos novos (Yahoo). Sem internet, usa os guardados."""
    diario, desd, prov = ur["precos"], ur["desdobramentos"], ur["proventos"]
    try:
        yh = yahoo.serie(f"{ticker}.SA", max_idade_horas=1)
    except Exception as e:
        return diario, desd, prov, f"sem preços novos ({e.__class__.__name__}); usados os guardados"
    novos_desd = yh["desdobramentos"]
    desd = pd.concat([desd, novos_desd], ignore_index=True).drop_duplicates("data").sort_values("data").reset_index(drop=True)
    amanha = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
    py = yh["precos"]
    ult = diario["data"].max()
    novos = py[py["data"] > ult].copy()
    if len(novos):
        novos["fechamento"] = novos["fechamento"].to_numpy() * P.fatores_ate(novos["data"], desd, amanha)
        novos["negocios"] = np.where(novos["volume"].fillna(0) > 0, 1, 0)
        novos["origem"] = "Yahoo"
        diario = pd.concat([diario, novos[["data", "fechamento", "negocios", "origem"]]], ignore_index=True)
    pv = yh["proventos"].copy()
    if len(pv):
        pv["valor"] = pv["valor"].to_numpy() * P.fatores_ate(pv["data"], desd, amanha)
        prov = pd.concat([prov, pv[pv["data"] > (prov["data"].max() if len(prov) else pd.Timestamp("1900-01-01"))]],
                         ignore_index=True).sort_values("data").reset_index(drop=True)
    pasta = caminhos.pasta_ultima_rodada(ticker)
    diario.to_parquet(pasta / "precos.parquet", index=False)
    desd.to_parquet(pasta / "desdobramentos.parquet", index=False)
    prov.to_parquet(pasta / "proventos.parquet", index=False)
    return diario, desd, prov, None


def _docs_novos_cvm(analises: dict[str, dict]) -> dict[str, str]:
    """Ativos com ITR/DFP na CVM mais novo que o último usado na análise."""
    cnpjs = {t: a.get("cnpj") for t, a in analises.items() if a.get("cnpj")}
    if not cnpjs:
        return {}
    ano = date.today().year
    novos: dict[str, str] = {}
    docs = []
    for tipo in ("ITR", "DFP"):
        for a in (ano - 1, ano):
            try:
                por_empresa = cvm.extrair(sorted(set(cnpjs.values())), tipo, a,
                                          max_idade_horas=config.cfg()["atualizacao"]["cvm_idade_max_horas"])
                for x in (p["indice"] for p in por_empresa.values()):
                    if x is not None and len(x):
                        docs.append(x.assign(TIPO=tipo))
            except Exception:
                continue
    if not docs:
        return {}
    d = pd.concat(docs, ignore_index=True)
    d["DT_REFER"] = pd.to_datetime(d["DT_REFER"])
    for t, c in cnpjs.items():
        usado = analises[t].get("ultima_divulgacao_cvm") or {}
        ref_usada = pd.Timestamp(usado.get("data_referencia")) if usado.get("data_referencia") else None
        x = d[d["CNPJ_CIA"] == c]
        if ref_usada is not None and len(x) and x["DT_REFER"].max() > ref_usada:
            u = x.sort_values("DT_REFER").iloc[-1]
            novos[t] = f"{u['TIPO']} de {pd.Timestamp(u['DT_REFER']):%d/%m/%Y} na CVM"
    return novos


def _marcas(a: dict, idade: int, prazo: int, doc_novo: str | None, versao_atual: str | None,
            ajustado: bool, margem: float, fora: bool) -> list[dict]:
    m = []
    if fora:
        m.append({"texto": "fora do ranking dos ativos", "alerta": True, "dica": "retirado a pedido; rodar o ativo de novo o traz de volta"})
    motivos = []
    if idade >= prazo:
        motivos.append(f"análise com {idade} dias")
    if doc_novo:
        motivos.append(doc_novo)
    if versao_atual and a.get("versao_modelo") != versao_atual:
        motivos.append("feita com versão anterior do modelo")
    if motivos:
        m.append({"texto": "desatualizada", "alerta": True, "dica": "; ".join(motivos)})
    if a.get("estimativa") and np.isfinite(margem) and abs(margem) <= (a.get("erro_tipico") or 0):
        m.append({"texto": "dentro da margem de erro", "alerta": False,
                  "dica": "a margem, em módulo, não supera o erro típico do backtest"})
    if ajustado:
        m.append({"texto": "preço justo ajustado", "alerta": False,
                  "dica": "proventos, desdobramentos ou grupamentos depois da análise"})
    return m


def atualizar(novas: dict[str, dict] | None = None, progresso=print, refazer_telas: bool = True) -> dict:
    """Atualização rápida de todos os ativos do histórico. `novas`: análises desta rodada ainda não
    gravadas no histórico (entram no lugar da última de cada ativo)."""
    t0 = time.time()
    analises = historico.mais_recentes()
    for t, a in (novas or {}).items():
        analises[t] = a
    fora = historico.fora_do_ranking()
    for t in (novas or {}):
        fora.discard(t)
    cfg = config.cfg()
    prazo = int(cfg["ranking_dos_ativos"]["prazo_desatualizada_dias"])
    dados = {}
    for t in sorted(analises):
        ur = ler_ultima_rodada(t)
        if ur is None or ur["precos"] is None:
            continue
        if t in (novas or {}):
            ur["analise"] = analises[t]
        diario, desd, prov, aviso = _atualizar_precos(t, ur)
        dados[t] = {"ur": ur, "diario": diario, "desd": desd, "prov": prov, "aviso": aviso}
    if not dados:
        return {"linhas": [], "sem": [], "data_precos": None}
    # Mesma data de preço para todos: o último fechamento comum.
    data_precos = min(d["diario"]["data"].max() for d in dados.values())
    docs_novos = _docs_novos_cvm({t: dados[t]["ur"]["analise"] for t in dados})
    linhas, sem = [], []
    for t, d in dados.items():
        a = d["ur"]["analise"]
        preco, _ = P.preco_em(d["diario"], data_precos)
        idade = (pd.Timestamp(date.today()) - pd.Timestamp(str(a["data_analise"])[:10])).days
        try:
            versao_atual = config.versao_modelo(a["setor"])
        except Exception:
            versao_atual = None
        reg = {"ticker": t, "setor": a["setor"], "setor_nome": a.get("setor_nome", a["setor"]), "nome": a.get("nome", t),
               "preco": preco, "data_analise": pd.Timestamp(str(a["data_analise"])[:10]),
               "link_relatorio": a.get("relatorio"), "fora": t in fora, "motivo": a.get("motivo_sem_estimativa"),
               "estimativa": bool(a.get("estimativa"))}
        if a.get("estimativa"):
            V, ajustado = graficos.ajuste_pos_analise(a["preco_justo"], a["data_analise"][:10], data_precos,
                                                      d["diario"], d["desd"], d["prov"])
            r = V / a["preco_justo"]
            margem = (V - preco) / V
            faixa, classe = faixa_da_margem(margem)
            reg.update({"V": V, "P10": a["p10"] * r if a.get("p10") else None, "P90": a["p90"] * r if a.get("p90") else None,
                        "margem": margem, "margem_rs": V - preco, "faixa_margem": faixa, "classe_margem": classe,
                        "erro_tipico": a.get("erro_tipico"), "ajustado": ajustado,
                        "principal_preditor": a.get("principal_preditor_nome") or a.get("principal_preditor"),
                        "direcao": a.get("direcao")})
        else:
            ajustado, margem = False, np.nan
        reg["marcas"] = _marcas(a, idade, prazo, docs_novos.get(t), versao_atual, ajustado, margem, t in fora)
        reg["desatualizada"] = any(m["texto"] == "desatualizada" for m in reg["marcas"])
        if t in fora:
            continue
        (linhas if a.get("estimativa") else sem).append(reg)
    linhas.sort(key=lambda r: (-r["margem"], r["ticker"]))
    for i, r in enumerate(linhas, 1):
        r["posicao"] = i
    avisos = _avisos_concentracao(linhas, cfg)
    fora_lista = sorted(t for t in fora if t in dados)
    resultado = {"linhas": linhas, "sem": sem, "data_precos": data_precos, "avisos": avisos,
                 "n_posicao": len(linhas), "n_desatualizados": sum(r["desatualizada"] for r in linhas + sem),
                 "n_sem": len(sem), "fora": fora_lista,
                 "setores": sorted({r["setor_nome"] for r in linhas + sem}),
                 "gerado_em": formatos.data_hora(datetime.now())}
    paginas.escrever_ranking(resultado)
    if refazer_telas:
        por_ticker = {r["ticker"]: r for r in linhas + sem}
        for t, d in dados.items():
            escrever_tela_ativo(t, d, por_ticker.get(t), resultado, t in fora)
    if progresso:
        progresso(f"  atualização rápida: {len(dados)} ativos em {time.time() - t0:.0f} s")
    return resultado


def _avisos_concentracao(linhas: list[dict], cfg: dict) -> list[str]:
    n = int(cfg["ranking_dos_ativos"]["concentracao_primeiros"])
    frac = float(cfg["ranking_dos_ativos"]["concentracao_fracao"])
    top = linhas[:n]
    if len(top) < 3:
        return []
    av = []
    for campo, nome in (("setor_nome", "setor"), ("principal_preditor", "principal preditor")):
        c = Counter(r.get(campo) for r in top)
        valor, qtd = c.most_common(1)[0]
        if valor and qtd / len(top) >= frac:
            av.append(f"{qtd} dos {len(top)} primeiros têm o mesmo {nome} ({valor}): o desconto pode vir de um "
                      "fator comum, e não de cada empresa.")
    return av


def posicao_texto(t: str, resultado: dict, fora: bool) -> str:
    if fora:
        return "fora do ranking dos ativos"
    for r in resultado["linhas"]:
        if r["ticker"] == t:
            return f"{r['posicao']}º de {resultado['n_posicao']} ativos com estimativa"
    return "sem estimativa confiável (lista à parte)"


def escrever_tela_ativo(t: str, d: dict, reg: dict | None, resultado: dict, fora: bool) -> None:
    a = d["ur"]["analise"]
    ate = d["diario"]["data"].max()
    dg = graficos.montar(d["diario"], d["desd"], d["prov"], d["ur"]["walkforward"], a, historico.do_ativo(t),
                         ate, tracejado=True, eventos=d["ur"]["eventos"])
    preco, data_preco = P.preco_em(d["diario"], ate)
    ctx = {
        "ticker": t, "nome": a.get("nome", t), "setor_nome": a.get("setor_nome", a["setor"]),
        "preco": preco, "data_preco": data_preco, "data_analise": pd.Timestamp(str(a["data_analise"])[:10]),
        "confiavel": bool(a.get("estimativa")), "motivos": a.get("motivos") or [a.get("motivo_sem_estimativa") or ""],
        "link_relatorio": "../" + a["relatorio"] if a.get("relatorio") else "#",
        "posicao_texto": posicao_texto(t, resultado, fora),
        "marcas": (reg or {}).get("marcas", []) if not fora else [{"texto": "fora do ranking dos ativos", "alerta": True, "dica": ""}],
        "legenda": legenda(a, dg),
        "marcas_grafico": dg.get("eventos", []),
    }
    if a.get("estimativa"):
        V, ajustado = graficos.ajuste_pos_analise(a["preco_justo"], str(a["data_analise"])[:10], ate, d["diario"], d["desd"], d["prov"])
        r = V / a["preco_justo"]
        margem = (V - preco) / V
        faixa, classe = faixa_da_margem(margem)
        et = a.get("erro_tipico") or np.nan
        ctx.update({"V": V, "P10": a["p10"] * r if a.get("p10") else None, "P90": a["p90"] * r if a.get("p90") else None,
                    "ajustado": ajustado, "margem": margem, "margem_rs": V - preco, "faixa_margem": faixa,
                    "classe_margem": classe, "erro_tipico": et, "supera": abs(margem) > et,
                    "top5": a.get("top5", [])})
    paginas.escrever_tela(ctx, dg)


def legenda(a: dict, dg: dict) -> str:
    desde = dg.get("desde")
    partes = []
    if desde:
        partes.append(f"Dados desde {formatos.data(desde)}.")
    partes.append("Preço: fechamento diário ajustado só por desdobramentos e grupamentos.")
    if dg.get("eventos"):
        partes.append("Linhas verticais: pontilhadas para eventos (desdobramentos, grupamentos e mudanças de regras) e "
                      "tracejadas para quebras estruturais detectadas (diagnóstico; detalhes no relatório).")
    if a.get("estimativa"):
        partes.append(
            "Preço justo: estimativas walk-forward simuladas com a versão do modelo da análise mais recente "
            f"({a.get('versao_modelo')}), cada uma só com os dados divulgados até a data; não são análises feitas na época. "
            "As análises do histórico aparecem como pontos. O último ponto cheio é a análise mais recente; depois dele, "
            "a linha tracejada leva o preço justo até o último fechamento, ajustado por proventos, desdobramentos e grupamentos.")
    rev = a.get("series_revisadas") or []
    if rev:
        partes.append("Séries usadas com valores revisados depois (sem a versão da época): " + ", ".join(rev) + ".")
    return " ".join(partes)
