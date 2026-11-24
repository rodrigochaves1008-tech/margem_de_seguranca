"""Rodada completa de um ativo e lotes de ativos."""
from __future__ import annotations

import json
import shutil
import time
import traceback
from datetime import date, datetime

import numpy as np
import pandas as pd

from . import (AVISO, avaliacao, caminhos, config, formatos, graficos, historico, macro, painel,
               preditores, ranking_dos_ativos, valuation)
from .fontes import cvm as fonte_cvm
from .modelo import candidatos as C
from .modelo import walkforward as WF
from .saida import paginas


class SetorForaDoMapa(Exception):
    pass


def _fim_de_mes(d: pd.Timestamp) -> bool:
    return d == (d + pd.offsets.MonthEnd(0)).normalize()


def _ler_anteriores(ticker: str, versao: str):
    ur = caminhos.pasta_ultima_rodada(ticker)
    p = ur / "analise.json"
    if not p.exists():
        return None, None
    a = json.loads(p.read_text(encoding="utf-8"))
    if a.get("versao_modelo") != versao:
        return None, None
    wf, es = ur / "walkforward.parquet", ur / "escolhas.parquet"
    if not (wf.exists() and es.exists()):
        return None, None
    return pd.read_parquet(wf), pd.read_parquet(es)


def tempo_estimado_min(ticker: str) -> float:
    try:
        info = config.info_ativo(ticker)
        versao = config.versao_modelo(info["setor"])
    except Exception:
        return 1.0
    ant, _ = _ler_anteriores(ticker, versao)
    cache_ok = (caminhos.base_local() / "tratado" / "cvm").exists()
    return (2.0 if ant is not None else 8.0) + (0 if cache_ok else 20.0)


def rodar_ativo(ticker: str, hoje=None, progresso=print, lote: bool = False) -> dict:
    t0 = time.time()
    ticker = ticker.upper()
    try:
        info = config.info_ativo(ticker)
    except KeyError as e:
        raise SetorForaDoMapa(str(e)) from None
    try:
        setor = config.info_setor(info["setor"])
    except KeyError:
        raise SetorForaDoMapa(f"{ticker}: setor '{info['setor']}' fora do mapa de preditores") from None
    if not setor.get("ativo"):
        raise SetorForaDoMapa(
            f"{ticker}: o setor '{setor.get('nome', info['setor'])}' ainda não está ativo no mapa de preditores; "
            "os preditores precisam ser propostos e aprovados antes.")
    cfg = config.cfg()
    agora = datetime.now()
    hoje = pd.Timestamp(hoje or date.today()).normalize()
    progresso(f"{ticker}: carregando dados")
    dados = painel.carregar(ticker)
    datas = painel.datas_da_base(dados, hoje)
    base = painel.montar(dados, datas)
    pares, notas_pares = painel.carregar_pares(dados, progresso)
    dados.notas += notas_pares
    bases_pares = {p: painel.montar(d, datas) for p, d in pares.items()}
    versao = config.versao_modelo(info["setor"])
    anteriores, escolhas_ant = _ler_anteriores(ticker, versao)
    progresso(f"{ticker}: walk-forward ({'só os meses novos' if anteriores is not None else 'completo, pode levar alguns minutos'})")
    candidatas = [v for fam in painel.preditores_do_setor(setor).values() for v in fam]
    desd = dados.precos.desdobramentos
    res = WF.rodar(base, setor["alvos"], candidatas, cfg, dados.cvm["documentos"]["divulgacao"], desd, hoje,
                   anteriores, escolhas_ant, progresso=progresso, pares=bases_pares, ticker=ticker)
    mensal, h = res.mensal, res.hoje
    meses = [d for d in base.index if _fim_de_mes(d)]
    preco_mensal = base.loc[meses, "preco"]
    linha_hoje = base.loc[hoje]
    aval = avaliacao.avaliar(mensal, h, linha_hoje, base, preco_mensal, desd, cfg, setor.get("obrigatorios", []))
    progresso(f"{ticker}: ranking dos preditores e checagens")
    imp = preditores.importancia(res.passos_hoje, base, hoje, setor) if res.passos_hoje else None
    val = valuation.calcular(base, hoje, None, cfg, setor.get("contabilidade", "geral"), bases_pares)
    anos = int(cfg["modelo"]["anos_fora_amostra"])
    jan = avaliacao.janela_backtest(mensal, hoje, anos) if len(mensal) else mensal
    horizontes_bt = tuple(sorted(set(int(h) for h in cfg["modelo"]["escolha"]["horizontes_meses"])
                                 | set(avaliacao.limites_contra_passeio(cfg))))
    bt = (avaliacao.tabela_backtest(jan, preco_mensal, desd, hoje, setor["alvos"], horizontes=horizontes_bt)
          if len(jan) else pd.DataFrame())
    sinais = avaliacao.margem_como_sinal(jan, base, dados.precos.diario, desd, dados.precos.proventos, cfg) if len(jan) else []
    progresso(f"{ticker}: ranking dos preditores de 1, 2 e 3 anos atrás")
    historia_preditores = _preditores_no_tempo(res, base, hoje, setor, candidatas, cfg, bases_pares, ticker, imp)

    stamp = agora.strftime("%Y-%m-%d_%H%M")
    pasta_rel = caminhos.pasta_relatorios(ticker) / stamp
    rel_relativo = f"relatorios/{ticker}/{stamp}/relatorio.html"
    analise = _montar_analise(ticker, info, setor, versao, agora, h, aval, imp, dados, linha_hoje, rel_relativo)

    _salvar_ultima_rodada(ticker, analise, res, dados, imp)
    progresso(f"{ticker}: atualização rápida dos demais ativos")
    foto = ranking_dos_ativos.atualizar(novas={ticker: analise}, progresso=progresso)
    ctx = _contexto_relatorio(ticker, info, setor, analise, aval, imp, val, bt, jan, res, base, linha_hoje,
                              dados, foto, agora, hoje, cfg)
    ctx.update(_contexto_fase2(sinais, historia_preditores, pares, setor))
    dg = graficos.montar(dados.precos.diario, desd, dados.precos.proventos, mensal, analise,
                         historico.do_ativo(ticker), pd.Timestamp(h["data_preco"]), tracejado=False,
                         eventos=graficos.eventos_de_preco(desd))
    paginas.escrever_relatorio(pasta_rel, ctx, dg)
    _copiar_dados(pasta_rel, base, res, dados, imp, bt, val, foto, analise)
    if ticker in historico.fora_do_ranking():
        historico.registrar_fora(ticker, "volta", "rodado de novo")
    historico.registrar(_linha_historico(analise))
    # Refaz a tela com o histórico já atualizado (pontos das análises anteriores).
    foto2 = ranking_dos_ativos.atualizar(progresso=None)
    pos = ranking_dos_ativos.posicao_texto(ticker, foto2, False)
    dur = time.time() - t0
    progresso(f"{ticker}: pronto em {dur / 60:.1f} min · {pos}")
    return {"ticker": ticker, "analise": analise, "posicao": pos, "tela": caminhos.pasta_ativos() / f"{ticker}.html",
            "relatorio": pasta_rel / "relatorio.html", "ranking": foto2, "duracao_s": dur}


# ------------------------------------------------------------------ fase 2: preditores no tempo e sinais

def _preditores_no_tempo(res, base, hoje, setor, candidatas, cfg, bases_pares, ticker, imp_hoje) -> list[dict]:
    """Ranking dos preditores nos retreinos de 1, 2 e 3 anos atrás (modelos refeitos, iguais aos da época)."""
    if res.escolhas is None or not len(res.escolhas) or imp_hoje is None:
        return []
    def nome_modelo(c):
        m, a = (c or "|").split("|")
        return C.NOMES_MODELOS.get(m, m) + ("" if a in ("todos", "") else f", alvo {C.NOMES_ALVOS.get(a, a)}")

    out = [{"quando": "hoje", "data": hoje, "top": imp_hoje.head(5)[["nome", "importancia"]].to_dict("records"),
            "comuns": None, "correlacao": None, "modelo": nome_modelo(res.hoje.get("escolhido"))}]
    top_hoje = list(imp_hoje.head(5)["variavel"])
    esc = res.escolhas.sort_values("data")
    for anos in (1, 2, 3):
        alvo = hoje - pd.DateOffset(years=anos)
        x = esc[esc["data"] <= alvo]
        if x.empty:
            continue
        r = pd.Timestamp(x["data"].iloc[-1])
        passo = WF.refazer_passo(base, r, setor["alvos"], candidatas, cfg, x["escolhido"].iloc[-1], bases_pares, ticker)
        if passo is None or r not in base.index:
            continue
        try:
            imp = preditores.importancia(passo, base, r, setor)
        except Exception:
            continue
        comum = imp.set_index("variavel")["importancia"].reindex(imp_hoje["variavel"]).dropna()
        corr = (imp_hoje.set_index("variavel")["importancia"].reindex(comum.index).rank()
                .corr(comum.rank())) if len(comum) >= 4 else np.nan
        out.append({"quando": f"há {anos} {'ano' if anos == 1 else 'anos'}", "data": r,
                    "top": imp.head(5)[["nome", "importancia"]].to_dict("records"),
                    "comuns": len(set(imp.head(5)["variavel"]) & set(top_hoje)), "correlacao": corr,
                    "n_comuns": len(comum), "modelo": nome_modelo(x["escolhido"].iloc[-1])})
    return out


def _contexto_fase2(sinais, historia, pares, setor) -> dict:
    linhas_sinal = []
    for s in sinais:
        linhas_sinal.append({"faixa": s["faixa"], "horizonte": s["horizonte"], "n": s["n"],
                             "retorno_medio": s.get("retorno_medio"), "excesso_ibov": s.get("excesso_ibov"),
                             "excesso_cdi": s.get("excesso_cdi"), "acerto_ibov": s.get("acerto_ibov"),
                             "acerto_cdi": s.get("acerto_cdi")})
    n_max = max([s["n"] for s in sinais], default=0)
    texto = ("Poucos sinais: com prazos de 6 e 12 meses sem sobreposição, a amostra é pequena e o resultado não "
             "permite conclusão estatística." if n_max < 10 else "")
    hist_txt = []
    for h in historia[1:]:
        corr = (f"correlação de postos nos {h['n_comuns']} preditores em comum: {formatos.num(h['correlacao'], 2)}"
                if h.get("correlacao") is not None and np.isfinite(h["correlacao"])
                else f"só {h.get('n_comuns', 0)} preditores em comum, poucos para comparar a ordem")
        hist_txt.append(f"{h['quando']} ({formatos.data(h['data'])}, modelo: {h['modelo']}): {h['comuns']} dos 5 maiores de "
                        f"hoje já estavam entre os 5 maiores; {corr}.")
    return {"sinais": linhas_sinal, "texto_sinais": texto, "historia_preditores": historia,
            "texto_historia_preditores": hist_txt,
            "pares": sorted(pares), "nota_painel": (
                f"Painel com os pares do setor ({', '.join(sorted(pares))}): os candidatos em painel treinam o ativo e os "
                "pares juntos, com efeito fixo por empresa, e disputam a escolha do walk-forward como os demais. "
                "Esse painel foi incluído depois de ver o resultado da Fase 1 (aceito pelo usuário); o critério de "
                "estimativa confiável não mudou.") if pares else ""}


# ------------------------------------------------------------------ montagem dos registros

def _ultima_divulgacao(dados, hoje) -> dict:
    d = dados.cvm["documentos"]
    d = d[d["divulgacao"] <= hoje].sort_values(["divulgacao", "dt_refer"])
    if d.empty:
        return {}
    u = d.iloc[-1]
    return {"documento": u["tipo"], "data_referencia": u["dt_refer"].strftime("%Y-%m-%d"),
            "data_entrega": u["divulgacao"].strftime("%Y-%m-%d"), "versao": int(u["versao"])}


def _montar_analise(ticker, info, setor, versao, agora, h, aval, imp, dados, linha_hoje, rel_relativo) -> dict:
    confiavel = bool(aval["confiavel"])
    cat = config.catalogo_series()
    usados = [v for fam in painel.preditores_do_setor(setor).values() for v in fam]
    revisadas = [cat[s]["nome"] for s in usados if s in cat and cat[s].get("revisada")]
    top = imp.head(5) if imp is not None else pd.DataFrame()
    a = {
        "ticker": ticker, "nome": info.get("nome", ticker), "setor": info["setor"], "setor_nome": setor.get("nome"),
        "cnpj": info.get("cnpj"),
        "data_analise": agora.isoformat(timespec="seconds"),
        "data_preco": pd.Timestamp(h["data_preco"]).strftime("%Y-%m-%d"), "preco": h.get("preco"),
        "preco_justo": h.get("V") if confiavel else None,
        "p10": h.get("P10") if confiavel and np.isfinite(h.get("P10", np.nan)) else None,
        "p90": h.get("P90") if confiavel and np.isfinite(h.get("P90", np.nan)) else None,
        "preco_justo_modelo": h.get("V"), "p10_modelo": h.get("P10"), "p90_modelo": h.get("P90"),
        "erro_tipico": aval.get("erro_tipico"),
        "principal_preditor": imp.iloc[0]["variavel"] if imp is not None and len(imp) else None,
        "principal_preditor_nome": imp.iloc[0]["nome"] if imp is not None and len(imp) else None,
        "direcao": imp.iloc[0]["direcao"] if imp is not None and len(imp) else None,
        "estimativa": confiavel,
        "motivo_sem_estimativa": None if confiavel else "; ".join(aval["motivos"]),
        "motivos": aval["motivos"],
        "ultima_divulgacao_cvm": _ultima_divulgacao(dados, pd.Timestamp(agora).normalize()),
        "versao_modelo": versao,
        "relatorio": rel_relativo,
        "modelo_escolhido": h.get("escolhido"),
        "top5": [{"nome": r["nome"], "direcao": r["direcao"], "familia": r["familia"], "importancia": r["importancia"]}
                 for _, r in top.iterrows()],
        "series_revisadas": revisadas,
    }
    return _json_ok(a)


def _json_ok(x):
    if isinstance(x, dict):
        return {k: _json_ok(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_ok(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def _linha_historico(a: dict) -> dict:
    campos = ["ticker", "setor", "data_analise", "data_preco", "preco", "preco_justo", "p10", "p90", "erro_tipico",
              "principal_preditor", "principal_preditor_nome", "direcao", "estimativa", "motivo_sem_estimativa",
              "ultima_divulgacao_cvm", "versao_modelo", "relatorio"]
    return {k: a.get(k) for k in campos}


def _salvar_ultima_rodada(ticker, analise, res, dados, imp) -> None:
    ur = caminhos.pasta_ultima_rodada(ticker)
    (ur / "analise.json").write_text(json.dumps(analise, ensure_ascii=False, indent=1), encoding="utf-8")
    if len(res.mensal):
        res.mensal.to_parquet(ur / "walkforward.parquet")
    if len(res.escolhas):
        res.escolhas.to_parquet(ur / "escolhas.parquet", index=False)
    dados.precos.diario.to_parquet(ur / "precos.parquet", index=False)
    dados.precos.desdobramentos.to_parquet(ur / "desdobramentos.parquet", index=False)
    dados.precos.proventos.to_parquet(ur / "proventos.parquet", index=False)
    if imp is not None:
        imp.drop(columns=["explicacao"], errors="ignore").to_parquet(ur / "preditores.parquet", index=False)
    (ur / "eventos.json").write_text(json.dumps(graficos.eventos_de_preco(dados.precos.desdobramentos),
                                                ensure_ascii=False), encoding="utf-8")
    h = res.hoje
    (ur / "hoje_candidatos.json").write_text(json.dumps(_json_ok({
        "data": h.get("data"), "retreino": h.get("retreino"), "escolhido": h.get("escolhido"),
        "preco": h.get("preco"), "V_candidatos": h.get("V_candidatos", {})}), ensure_ascii=False), encoding="utf-8")
    vars_ = [{"data": v["data"].strftime("%Y-%m-%d"), "alvo": v["alvo"], "variaveis": v["variaveis"]}
             for v in res.variaveis_por_retreino]
    p = ur / "variaveis.json"
    antigos = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    vistos = {(v["data"], v["alvo"]) for v in vars_}
    juntos = [v for v in antigos if (v["data"], v["alvo"]) not in vistos] + vars_
    p.write_text(json.dumps(sorted(juntos, key=lambda v: (v["data"], v["alvo"])), ensure_ascii=False), encoding="utf-8")


def _copiar_dados(pasta, base, res, dados, imp, bt, val, foto, analise) -> None:
    d = pasta / "dados"
    (d / "series").mkdir(parents=True, exist_ok=True)
    base.to_parquet(d / "base_mensal.parquet")
    if len(res.mensal):
        res.mensal.to_parquet(d / "walkforward.parquet")
    if len(res.escolhas):
        res.escolhas.to_parquet(d / "escolhas.parquet", index=False)
    dados.precos.diario.to_parquet(d / "precos_diarios.parquet", index=False)
    dados.precos.desdobramentos.to_parquet(d / "desdobramentos.parquet", index=False)
    dados.precos.proventos.to_parquet(d / "proventos.parquet", index=False)
    dados.cvm["demonstracoes"].to_parquet(d / "cvm_demonstracoes.parquet", index=False)
    dados.cvm["documentos"].to_parquet(d / "cvm_documentos.parquet", index=False)
    for sid, s in dados.series.items():
        s.to_parquet(d / "series" / f"{sid}.parquet", index=False)
    if imp is not None:
        imp.to_parquet(d / "preditores.parquet", index=False)
    if len(bt):
        bt.to_parquet(d / "backtest.parquet", index=False)
    foto_json = {"data_precos": str(foto.get("data_precos"))[:10],
                 "linhas": [{k: r.get(k) for k in ("posicao", "ticker", "preco", "V", "P10", "P90", "margem", "faixa_margem")}
                            for r in foto.get("linhas", [])],
                 "sem": [{"ticker": r["ticker"], "motivo": r.get("motivo")} for r in foto.get("sem", [])]}
    (d / "foto_ranking_dos_ativos.json").write_text(json.dumps(_json_ok(foto_json), ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "valuation.json").write_text(json.dumps(_json_ok(val), ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "analise.json").write_text(json.dumps(analise, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copyfile(caminhos.CONFIG / "configuracao.yaml", d / "configuracao.yaml")
    shutil.copyfile(caminhos.CONFIG / "setores.yaml", d / "setores.yaml")
    shutil.copyfile(caminhos.CONFIG / "series.yaml", d / "series.yaml")


# ------------------------------------------------------------------ textos do relatório

def _fmt_valor(v: str, x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    cat = config.catalogo_series()
    un = cat.get(v, {}).get("unidade", "")
    tr = cat.get(v, {}).get("transformacao", "nivel")
    if v in ("roe", "roa", "roic", "cresc_lucro_12m", "cresc_lucro_3a", "eficiencia", "custo_credito",
             "margem_financeira", "cresc_ativo_12m", "payout", "margem_bruta", "margem_ebitda", "margem_ebit",
             "margem_liquida", "cresc_receita_12m", "cresc_ebitda_12m", "cresc_receita_3a", "cresc_ebitda_3a",
             "fcl_receita", "capex_receita", "variacao_acoes_12m", "free_float", "divida_liquida_pl") or tr == "var12m":
        return formatos.pct(x, 1)
    if v in ("divida_liquida_ebitda", "cobertura_juros", "liquidez_corrente", "giro_ativo"):
        return formatos.num(x, 2) + "×"
    if v == "ciclo_caixa":
        return formatos.num(x, 0) + " dias"
    if v == "_conv_divida":
        return "R$ " + formatos.num(x / 1e9, 1) + " bi"
    if v.startswith("_conv") or v in ("log_lucro_12m", "log_patrimonio"):
        return "R$ " + formatos.num(np.exp(x) / 1e9, 1) + " bi"
    if v == "alavancagem":
        return formatos.num(x, 1) + "×"
    if un.startswith("%") or un in ("p.p.",):
        return formatos.num(x, 2) + (" p.p." if un == "p.p." else "%")
    return formatos.num(x, 2)


def _contexto_relatorio(ticker, info, setor, a, aval, imp, val, bt, jan, res, base, linha_hoje, dados, foto,
                        agora, hoje, cfg) -> dict:
    h = res.hoje
    confiavel = a["estimativa"]
    V = h.get("V", np.nan)
    preco = h.get("preco", np.nan)
    margem = (V - preco) / V if confiavel and V else np.nan
    faixa, classe = ranking_dos_ativos.faixa_da_margem(margem) if confiavel else ("—", "neu")
    et = aval.get("erro_tipico", np.nan)
    cp = aval.get("contra_passeio", {})
    lim = aval.get("limites") or avaliacao.limites_contra_passeio(cfg)
    h_longo, h_curto = max(lim), min(lim)
    r12 = cp.get(h_longo, {}).get("razao", np.nan)
    r6 = cp.get(h_curto, {}).get("razao", np.nan)
    modelo, alvo = (h.get("escolhido") or "|").split("|")
    nome_modelo = (C.NOMES_MODELOS.get(modelo, modelo) if alvo in ("todos", "")
                   else f"{C.NOMES_MODELOS.get(modelo, modelo)}, alvo {C.NOMES_ALVOS.get(alvo, alvo)}")
    pos = ranking_dos_ativos.posicao_texto(ticker, foto, False)

    resumo = []
    if confiavel:
        resumo.append(f"O modelo estima o preço justo de {ticker} em {formatos.moeda(V)} (faixa de 80%: "
                      f"{formatos.moeda(h['P10'])} a {formatos.moeda(h['P90'])}), contra {formatos.moeda(preco)} "
                      f"em {formatos.data(h['data_preco'])}.")
        resumo.append(f"A margem de segurança é de {formatos.pct(margem, 1, True)} ({faixa}); "
                      + ("ela supera o erro típico do backtest" if abs(margem) > et else "ela não supera o erro típico do backtest")
                      + f" ({formatos.pct(et, 1)}).")
    else:
        resumo.append(f"Não há estimativa confiável para {ticker} nesta data: " + "; ".join(aval["motivos"]) + ".")
    if imp is not None and len(imp):
        p0 = imp.iloc[0]
        resumo.append(f"O principal preditor é {p0['nome']} ({formatos.pct(p0['importancia'], 0)} da importância), "
                      f"que hoje empurra o preço justo {p0['direcao']}.")
    if np.isfinite(r12):
        resumo.append(f"No backtest, o erro contra o preço {h_longo} meses depois foi {formatos.pct(r12, 0)} do erro do passeio "
                      f"aleatório nos últimos {cp[h_longo]['n']} meses ({'o modelo ganhou' if r12 < 1 else 'o passeio aleatório ganhou'}).")
    resumo.append(f"Posição: {pos}.")

    texto_margem = []
    if confiavel:
        sinal = "abaixo" if margem > 0 else "acima"
        texto_margem.append(
            f"O preço está {formatos.pct(abs(margem), 1)} {sinal} do preço justo (margem = (V − P) ÷ V); para chegar ao preço "
            f"justo, o preço teria de {'subir' if margem > 0 else 'cair'} {formatos.pct(abs(V / preco - 1), 1)}. "
            + ("A diferença é maior que o erro típico do modelo, então não se explica só pelo erro." if abs(margem) > et
               else "A diferença é menor que o erro típico do modelo: fica dentro da margem de erro."))
    texto_margem.append(f"Modelo escolhido pelo walk-forward na data: {nome_modelo}.")
    if not confiavel:
        texto_margem.append("Sem estimativa confiável, o número calculado pelo modelo não é mostrado; ele fica só na cópia "
                            "dos dados deste relatório (dados/analise.json), para auditoria.")

    # Preditores
    prds, explicacoes, concordancia, mudanca = [], [], np.nan, ""
    if imp is not None and len(imp):
        concordancia = imp.attrs.get("concordancia", np.nan)
        for _, p in imp.head(10).iterrows():
            prds.append({"nome": p["nome"], "familia": p["familia"], "importancia": p["importancia"],
                         "importancia_permutacao": p["importancia_permutacao"], "direcao": p["direcao"],
                         "valor_txt": _fmt_valor(p["variavel"], p["valor_hoje"]),
                         "mediana_txt": _fmt_valor(p["variavel"], p["mediana_treino"])})
        for _, p in imp.head(5).iterrows():
            vh, vm = _fmt_valor(p["variavel"], p["valor_hoje"]), _fmt_valor(p["variavel"], p["mediana_treino"])
            comp = ("igual à mediana" if vh == vm else
                    "acima da mediana" if p["valor_hoje"] > p["mediana_treino"] else "abaixo da mediana")
            efeito = (f"no modelo, isso empurra o preço justo {p['direcao']}" if p["direcao"] != "—"
                      else "o efeito de hoje não pôde ser calculado")
            explicacoes.append(
                f"{p['nome']}: {p['explicacao']}. Hoje está em {vh}, {comp} do período de treino ({vm}); {efeito}.")
        mudanca = _mudanca_preditores(res, hoje, alvo)

    macro_ids = ["selic", "juro_curto", "juro_longo", "juro_real_longo", "ipca_12m", "focus_ipca_12m",
                 "focus_selic_prox_ano", "focus_pib_prox_ano", "dolar_var12m", "ibc_br_var12m", "desemprego",
                 "inad_total", "premio_risco_brasil", "ibovespa_var12m", "fed_funds", "ust10", "vix"]
    antes = (hoje - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0)).normalize()
    cat = config.catalogo_series()
    macro_rows = []
    for m in macro_ids:
        if m in base.columns:
            macro_rows.append({"nome": cat.get(m, {}).get("nome", m), "hoje": _fmt_valor(m, linha_hoje.get(m)),
                               "antes": _fmt_valor(m, base.loc[antes, m] if antes in base.index else np.nan)})

    # Backtest
    bt_rows = []
    if len(bt):
        for i, r in bt.iterrows():
            d = r.to_dict()
            d["destaque"] = i < 2
            bt_rows.append(d)
    if len(bt):
        esc = bt.iloc[0]
        rw = bt.iloc[1]
        outros = bt.iloc[2:]
        L, Cc = f"mape_{h_longo}m", f"mape_{h_curto}m"
        melhor = outros.sort_values(L).iloc[0] if len(outros) else None
        venceu_l = "o modelo" if esc[L] < rw[L] else "o passeio aleatório"
        venceu_c = "o modelo" if esc[Cc] < rw[Cc] else "o passeio aleatório"
        texto_bt = (f"Nos {int(esc[f'n_{h_longo}m'])} meses fora da amostra com preço {h_longo} meses depois já conhecido, a "
                    f"escolha do walk-forward errou em média {formatos.pct(esc[L], 1)}, contra {formatos.pct(rw[L], 1)} "
                    f"do passeio aleatório: ganhou {venceu_l}. Em {h_curto} meses: {formatos.pct(esc[Cc], 1)} contra "
                    f"{formatos.pct(rw[Cc], 1)} ({venceu_c}).")
        if melhor is not None:
            texto_bt += (f" Olhando o período todo depois (sem uso na escolha), o candidato com menor erro em {h_longo} meses "
                         f"foi {melhor['modelo']} ({formatos.pct(melhor[L], 1)}).")
    else:
        texto_bt = "Sem meses fora da amostra suficientes para o backtest."
    cont = res.escolhas["escolhido"].value_counts() if len(res.escolhas) else pd.Series(dtype=int)
    contagem = "; ".join(f"{C.NOMES_MODELOS.get(k.split('|')[0], k)} ({C.NOMES_ALVOS.get(k.split('|')[1], '')}): {v}"
                         for k, v in cont.items())

    # Valuation
    ref = V if confiavel else preco
    val_rows = [{"metodo": m["metodo"], "valor": m["valor"],
                 "dif": (m["valor"] / ref - 1) if np.isfinite(m["valor"]) and ref else np.nan, "premissas": m["premissas"]}
                for m in val["metodos"]]
    texto_val = _texto_valuation(val, val_rows, confiavel, linha_hoje, base, hoje)

    limitacoes = _limitacoes(dados, aval, setor)
    cob = macro.cobertura(dados.series, dados.ids_series, [])
    familias = {"macro_brasil": "macro Brasil", "exterior": "exterior", "setor": "setor", "empresa": "empresa"}
    for c in cob:
        d = c["defasagem"] or {}
        if "meses" in d:
            n = int(d["meses"])
            c["defasagem_txt"] = "fim do mês" if n == 0 else f"fim do mês + {n} {'mês' if n == 1 else 'meses'}"
        elif d:
            n = int(d.get("dias", 0))
            c["defasagem_txt"] = "no dia" if n == 0 else f"{n} {'dia' if n == 1 else 'dias'} depois"
        else:
            c["defasagem_txt"] = "calculada"
        c["familia"] = familias.get(c["familia"], c["familia"])
        c["uso"] = "preditor" if c["preditor"] else "auxiliar"
    precos_cob = {"serie": "Preço de fechamento", "familia": "empresa", "fonte": "B3 (COTAHIST) e Yahoo Finance",
                  "codigo": ticker, "inicio": dados.precos.diario["data"].min(), "fim": dados.precos.diario["data"].max(),
                  "defasagem_txt": "no dia", "revisada": False, "uso": "alvo"}
    docs = dados.cvm["documentos"]
    cvm_cob = {"serie": "Balanços (DFP e ITR)", "familia": "empresa", "fonte": "CVM dados abertos",
               "codigo": info.get("cnpj"), "inicio": docs["dt_refer"].min(), "fim": docs["dt_refer"].max(),
               "defasagem_txt": "data de entrega à CVM", "revisada": False, "uso": "preditor"}
    return {
        "ticker": ticker, "nome": info.get("nome"), "setor_nome": setor.get("nome"), "data_analise": agora,
        "data_hora": formatos.data_hora(agora), "data_preco": h.get("data_preco"), "preco": preco,
        "versao_modelo": a["versao_modelo"], "confiavel": confiavel, "motivos": aval["motivos"],
        "V": V, "P10": h.get("P10"), "P90": h.get("P90"), "margem": margem, "margem_rs": V - preco if confiavel else np.nan,
        "faixa_margem": faixa, "classe_margem": classe, "erro_tipico": et, "supera": confiavel and abs(margem) > et,
        "potencial": V / preco - 1 if confiavel else np.nan,
        "resumo": resumo[:5], "texto_margem": texto_margem,
        "legenda": ranking_dos_ativos.legenda(a, {"desde": dados.precos.diario["data"].max() - pd.DateOffset(years=10)
                                                   if dados.precos.diario["data"].min() < dados.precos.diario["data"].max() - pd.DateOffset(years=10)
                                                   else dados.precos.diario["data"].min()}),
        "foto": {"data_precos": foto.get("data_precos"), "sem": foto.get("sem", []),
                 "linhas": [{**r, "marcas_texto": ", ".join(m["texto"] for m in r.get("marcas", []))} for r in foto.get("linhas", [])]},
        "preditores": prds, "explicacoes": explicacoes, "concordancia": concordancia, "mudanca_preditores": mudanca,
        "modelo_escolhido": nome_modelo,
        "macro": macro_rows,
        "nota_projecao": "Projeção do preço justo em 12 meses por cenários (base pelo Boletim Focus, otimista e "
                         "pessimista, variando juros, câmbio e atividade): entra na Fase 3.",
        "nota_estruturais": "Testes de quebras estruturais, marcação de eventos (reforma tributária do consumo, IR sobre "
                            "dividendos e JCP, novas regras de provisão dos bancos, regulação, controle estatal) e pesquisa "
                            "das mudanças dos últimos 24 meses com fonte e data: entram na Fase 3.",
        "texto_backtest": texto_bt, "backtest": bt_rows,
        "janela_backtest": (f"{formatos.mes(jan.index.min())} a {formatos.mes(jan.index.max())}" if len(jan) else "—"),
        "cobertura": aval.get("cobertura_faixa"), "meses_com_faixa": aval.get("meses_com_faixa"),
        "janela_comparacao": cfg["estimativa_confiavel"]["janela_comparacao_meses"], "razao12": r12, "razao6": r6,
        "h_longo": h_longo, "h_curto": h_curto,
        "bt_horizontes": sorted({int(c.split("_")[1][:-1]) for c in (bt.columns if len(bt) else []) if c.startswith("mape_")}),
        "n_retreinos": len(res.escolhas), "contagem_escolhas": contagem,
        "valuation": val_rows, "texto_valuation": texto_val,
        "rotulo_dif": "Diferença para o preço justo do modelo" if confiavel else "Diferença para o preço atual",
        "custo_ida_volta": float(cfg["custos"]["ida_e_volta"]),
        "limitacoes": limitacoes, "cobertura_series": [precos_cob, cvm_cob] + cob,
        "metodologia": _metodologia(setor, cfg),
    }


def _mudanca_preditores(res, hoje, alvo) -> str:
    vs = res.variaveis_por_retreino
    if not vs:
        return "Sem retreinos guardados."
    alvos = {v["alvo"] for v in vs}
    a = alvo if alvo in alvos else sorted(alvos)[0]
    x = sorted([v for v in vs if v["alvo"] == a], key=lambda v: v["data"])
    atual = x[-1]["variaveis"]
    partes = []
    for anos in (1, 2, 3):
        alvo_data = hoje - pd.DateOffset(years=anos)
        ant = [v for v in x if v["data"] <= alvo_data]
        if ant:
            comum = len(set(ant[-1]["variaveis"]) & set(atual))
            partes.append(f"há {anos} {'ano' if anos == 1 else 'anos'}, {comum} das {len(atual)} variáveis de hoje já "
                          "eram usadas")
    nomes = ", ".join(preditores.nome_preditor(v) for v in atual)
    return (f"Variáveis escolhidas no retreino mais recente (alvo {C.NOMES_ALVOS.get(a, a)}): {nomes}. "
            + ("; ".join(partes) + "." if partes else ""))


def _texto_valuation(val, rows, confiavel, linha, base, hoje) -> list[str]:
    txt = []
    difs = [r["dif"] for r in rows if np.isfinite(r["dif"])]
    ref = "o preço justo do modelo" if confiavel else "o preço atual (não há preço justo confiável)"
    if difs:
        txt.append(f"A coluna de diferença compara cada método com {ref}. Mediana das diferenças: "
                   f"{formatos.pct(float(np.median(difs)), 0, True)}.")
    if difs and sum(abs(d) > 0.30 for d in difs) >= len(difs) / 2:
        roe = linha.get("roe", np.nan)
        expl = [f"A maioria dos métodos diverge mais de 30%. O ROE dos últimos 12 meses é de {formatos.pct(roe, 1)}, contra "
                f"{formatos.pct(val['roe_sustentavel'], 1)} de mediana em 5 anos"]
        if np.isfinite(val.get("pl_atual", np.nan)):
            expl.append(f"o P/L atual é {formatos.num(val['pl_atual'], 1)} contra {formatos.num(val['pl_mediano'], 1)} de "
                        f"mediana em 10 anos e o P/VP é {formatos.num(val['pvp_atual'], 2)} contra {formatos.num(val['pvp_mediano'], 2)}")
        txt.append("; ".join(expl) + ". Métodos baseados no lucro recente (Graham, P/L, dividendos) reagem a lucros "
                   "atípicos; o modelo também pesa juros, risco e o histórico da relação entre fundamentos e preço.")
    return txt


def _limitacoes(dados, aval, setor) -> list[str]:
    lim = list(dados.notas)
    lim.append("Fundamentos: para bancos com consolidadas trimestrais incompletas na CVM, são usadas as demonstrações "
               "individuais (padrão Banco Central), disponíveis em todo o período; no BB, as consolidadas trimestrais "
               "só existem a partir de 2022.")
    lim.append("Ações em circulação: antes de a composição do capital aparecer no ITR/DFP (no BB, 2020), usa-se o total "
               "de ações do Formulário de Referência, sem descontar as ações em tesouraria (diferença abaixo de 1% no BB).")
    cat = config.catalogo_series()
    rev = [cat[s]["nome"] for s in dados.ids_series if cat.get(s, {}).get("revisada")]
    if rev:
        lim.append("Séries revisadas depois, sem a versão da época na fonte (usada a versão atual, com defasagem "
                   "conservadora): " + ", ".join(rev) + ".")
    lim.append("Risco-país: o EMBI+ parou de ser publicado em 30/07/2024 e o CDS de 5 anos não tem fonte gratuita com "
               "histórico; o modelo usa o prêmio de juro real do Brasil sobre os EUA (NTN-B 10 anos − TIPS 10 anos) "
               "como substituto.")
    lim.append("Séries do FRED na versão atual: as versões da época (ALFRED) precisam da chave gratuita do FRED.")
    lim.append("Juros da curva: títulos do Tesouro Direto (taxa de compra da manhã) interpolados nos prazos de 1, 5 e "
               "10 anos, no lugar dos contratos de DI futuro da B3.")
    lim.append("Amostra pequena: cerca de 190 meses e fundamentos trimestrais (umas 60 observações independentes). "
               "A rede neural é pequena e regularizada, mas o risco de sobreajuste continua alto.")
    if setor.get("contabilidade") != "banco":
        lim.append("Empresas não financeiras: EBITDA = EBIT + depreciação e amortização da DFC; capex = aquisições de "
                   "imobilizado e intangível. Em concessões (energia), parte do investimento vai para ativos de contrato "
                   "e não aparece como capex, o que superestima o fluxo de caixa livre.")
    if "reservatorios_seco" in dados.ids_series:
        lim.append("Energia: PLD e GSF são da CCEE, que recusa acesso automatizado a partir de servidores; o custo "
                   "marginal de operação (CMO) do ONS entra no lugar do PLD, e o GSF fica de fora.")
    lim.append("Valor de mercado: preço do ticker analisado × total de ações (ON + PN). Em units, o preço é dividido "
               "pelo número de ações da unit.")
    lim.append("Fase 3 (ainda não entra): mudanças estruturais e projeção por cenários.")
    return lim


def _metodologia(setor, cfg) -> list[str]:
    m = cfg["modelo"]
    return [
        "Alvo: valor de mercado, P/L e P/VP (em log), sempre convertidos para preço por ação na base de ações da data. "
        "O preço é o fechamento ajustado só por desdobramentos e grupamentos; nunca o ajustado por proventos.",
        "Base mensal point-in-time: cada balanço entra na data de entrega à CVM; cada série macro, no calendário de "
        "divulgação (defasagens conservadoras por série); o último dado já divulgado no fim de cada mês.",
        f"Modelos: rede neural (uma camada oculta com {m['rede_neural']['camadas_ocultas'][0]} neurônios, regularização L2, "
        f"early stopping num bloco final do treino em ordem de tempo, média de {len(m['sementes'])} sementes), ridge, "
        "gradient boosting e médias entre eles; referência: passeio aleatório (último preço).",
        "Walk-forward: retreino no fim do mês em que sai um documento novo na CVM; a cada retreino, seleção de variáveis "
        "(cobertura, redundância e LASSO com validação no tempo), normalização, hiperparâmetros e escolha do candidato "
        f"(menor erro médio contra o preço 6 e 12 meses depois, só com resultados já fechados, nos últimos "
        f"{m['escolha']['janela_meses']} meses) são refeitos só com os dados até a data.",
        "Faixa de 80%: preço justo × percentis 10 e 90 de preço ÷ preço justo nos meses fora da amostra já conhecidos "
        "(mínimo de 24 meses). Erro típico: mediana de |V − P| ÷ V nos meses fora da amostra.",
        "Importância dos preditores: SHAP (permutação) sobre o log do valor de mercado estimado, conferida por "
        "importância de permutação.",
        "Painel com os pares do setor (config/setores.yaml): candidatos com o ativo e os pares juntos, alvos em "
        "múltiplo (P/L, P/VP e, fora dos bancos, EV/EBITDA) e efeito fixo por empresa; a seleção de variáveis olha "
        "a variação dentro de cada empresa e a validação separa por data.",
        "Fontes: CVM dados abertos (DFP, ITR, FRE); B3 (COTAHIST); Yahoo Finance (proventos, desdobramentos, índices); "
        "Banco Central (SGS e Boletim Focus); Tesouro Transparente (Tesouro Direto); FRED; Ipeadata.",
    ]


# ------------------------------------------------------------------ lotes

def rodar_lote(tickers: list[str], progresso=print) -> dict:
    tickers = [t.upper() for t in tickers]
    arq = caminhos.arquivo_lote()
    estado = None
    if arq.exists():
        try:
            estado = json.loads(arq.read_text(encoding="utf-8"))
        except Exception:
            estado = None
    if estado and set(estado.get("tickers", [])) == set(tickers) and estado.get("pendentes"):
        progresso(f"Retomando o lote de onde parou: faltam {', '.join(estado['pendentes'])}.")
    else:
        estado = {"tickers": tickers, "pendentes": list(tickers), "concluidos": [], "falhas": {},
                  "inicio": datetime.now().isoformat(timespec="seconds")}
    est = sum(tempo_estimado_min(t) for t in estado["pendentes"])
    progresso(f"Tempo estimado: cerca de {est:.0f} min para {len(estado['pendentes'])} ativo(s).")
    arq.write_text(json.dumps(estado, ensure_ascii=False), encoding="utf-8")
    resultados = {}
    for t in list(estado["pendentes"]):
        try:
            resultados[t] = rodar_ativo(t, progresso=progresso, lote=True)
            estado["concluidos"].append(t)
        except SetorForaDoMapa as e:
            estado["falhas"][t] = f"precisa do seu OK: {e}"
        except Exception as e:
            estado["falhas"][t] = f"{e.__class__.__name__}: {e}"
            traceback.print_exc()
        estado["pendentes"].remove(t)
        arq.write_text(json.dumps(estado, ensure_ascii=False), encoding="utf-8")
    arq.unlink(missing_ok=True)
    return {"resultados": resultados, "falhas": estado["falhas"]}


def resumo_ranking(r: dict, n: int = 10) -> str:
    if not r or r.get("data_precos") is None:
        return "Ranking dos ativos vazio."
    linhas = [f"Preços de {formatos.data(r['data_precos'])} · {r['n_posicao']} ativos com estimativa · "
              f"{r['n_desatualizados']} desatualizados · {r['n_sem']} sem estimativa confiável"]
    for x in r["linhas"][:n]:
        marcas = ", ".join(m["texto"] for m in x["marcas"])
        linhas.append(f"{x['posicao']}º {x['ticker']}  {formatos.pct(x['margem'], 1, True)}  {x['faixa_margem']}"
                      + (f"  [{marcas}]" if marcas else ""))
    for x in r["sem"]:
        linhas.append(f"sem estimativa: {x['ticker']} ({x.get('motivo')})")
    for av in r.get("avisos", []):
        linhas.append("Aviso: " + av)
    return "\n".join(linhas)
