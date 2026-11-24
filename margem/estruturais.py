"""Mudanças estruturais: eventos (config/eventos.yaml e a pesquisa na web da rodada) e testes de quebras
estruturais no preço, nas séries dos principais preditores e na relação entre preditores e preço.

Tudo aqui é diagnóstico, para leitura: nada entra no preço justo nem na escolha do modelo. As datas de
quebra são estimadas com o período todo; o walk-forward continua usando só os dados até cada data, e uma
mudança só aparece no modelo quando chega aos dados.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import yaml

from . import caminhos, formatos, preditores
from . import precos as P

TIPOS = {"tributario": "tributária", "contabil": "contábil", "regulatorio": "regulação do setor",
         "controle": "controle ou privatização", "estatal": "interferência em estatal", "outro": "outra"}
# Na conversão de um múltiplo para preço entra um número da empresa: mudanças nele chegam ao preço justo.
CANAIS_DA_CONVERSAO = {"log_pl": ["log_lucro_12m"], "log_pvp": ["log_patrimonio"], "log_ev_ebitda": ["log_ebitda_12m"]}


# ------------------------------------------------------------------ eventos

def eventos_configurados(setor: str, ticker: str) -> list[dict]:
    arq = caminhos.CONFIG / "eventos.yaml"
    lista = []
    if arq.exists():
        lista = (yaml.safe_load(arq.read_text(encoding="utf-8")) or {}).get("eventos") or []
    out = []
    for e in lista:
        setores, ativos = e.get("setores") or ["todos"], e.get("ativos") or []
        if (ativos and ticker not in ativos) or ("todos" not in setores and setor not in setores):
            continue
        out.append(e)
    return out


def ler_pesquisa(ticker: str) -> dict | None:
    """Pesquisa na web da rodada, gravada pelo Claude na sessão (ver a skill)."""
    p = caminhos.pasta_ultima_rodada(ticker) / "mudancas_estruturais.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _normalizar(e: dict, origem: str) -> dict:
    fontes = e.get("fontes") or ([{"url": e["fonte"], "titulo": e.get("fonte_titulo") or e["fonte"],
                                   "data": e.get("fonte_data")}] if e.get("fonte") else [])
    data = str(e["data"])[:10]
    return {"data": data, "titulo": e.get("titulo", ""), "rotulo": e.get("rotulo") or e.get("titulo", "")[:30],
            "tipo": e.get("tipo", "outro"), "situacao": e.get("situacao", "em vigor"),
            "vigencia": str(e.get("vigencia") or data)[:10], "efeito_esperado": e.get("efeito_esperado", "incerto"),
            "explicacao": e.get("explicacao", ""), "canais": list(e.get("canais") or []), "fontes": fontes,
            "origem": origem}


def variaveis_do_modelo(passo) -> list[str]:
    """Variáveis do candidato escolhido hoje, mais o número da empresa usado na conversão do múltiplo."""
    if passo is None:
        return []
    vs: list[str] = []
    for a, _ in preditores._componentes(passo):
        aj = passo.ajustes.get(a)
        if aj is None:
            continue
        for v in list(aj.variaveis) + CANAIS_DA_CONVERSAO.get(a, []):
            if v not in vs:
                vs.append(v)
    return vs


def capta(e: dict, variaveis_modelo: list[str], hoje) -> str:
    """Se o modelo de hoje tem por onde ver a mudança (os "canais" informados na pesquisa)."""
    canais = e.get("canais") or []

    def nomes(vs):
        return ", ".join(preditores.nome_preditor(v) for v in vs)

    vig = pd.Timestamp(e["vigencia"])
    if vig > pd.Timestamp(hoje):
        return (f"Ainda não: passa a valer em {formatos.data(vig)}; o modelo só vê a mudança quando ela chegar aos "
                "dados" + (f" ({nomes(canais)})." if canais else "."))
    if not canais:
        return "Não: nenhum preditor do modelo mede essa mudança diretamente."
    usados = [c for c in canais if c in variaveis_modelo]
    if usados:
        return (f"Em parte: o modelo de hoje usa {nomes(usados)}, por onde a mudança age; o efeito entra quando "
                "aparece nesses dados.")
    return f"Não: o modelo de hoje não usa os indicadores por onde a mudança age ({nomes(canais)})."


# ------------------------------------------------------------------ testes de quebra

def _custo_mqo():
    import ruptures as rpt

    class CustoMQO(rpt.base.BaseCost):
        """Soma dos quadrados dos resíduos da regressão do segmento (robusta a colunas constantes)."""
        model = "mqo"
        min_size = 2

        def fit(self, signal):
            self.signal = np.asarray(signal, float)
            return self

        def error(self, start, end):
            y, X = self.signal[start:end, 0], self.signal[start:end, 1:]
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            r = y - X @ b
            return float(r @ r)

    return CustoMQO()


def _lwz(ssr: float, n: int, q: int, k: int) -> float:
    """Critério de Liu, Wu e Zidek (1997), indicado por Bai e Perron (2003) quando os resíduos são correlacionados
    (o BIC tende a achar quebras demais)."""
    p = (k + 1) * q + k
    if n - p <= 0:
        return np.inf
    return float(np.log(ssr / (n - p)) + p / n * 0.299 * np.log(n) ** 2.1)


def bai_perron(y, X, min_seg: int, max_quebras: int) -> dict:
    """Quebras na regressão y = X·b por mínimos quadrados em partes (programação dinâmica, Bai e Perron);
    o número de quebras sai do critério LWZ. `quebras`: índice da primeira observação de cada novo trecho."""
    import ruptures as rpt

    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, q = X.shape
    k_max = max(0, min(int(max_quebras), n // int(min_seg) - 1))
    algo = rpt.Dynp(custom_cost=_custo_mqo(), min_size=int(min_seg), jump=1).fit(np.column_stack([y, X]))
    crit, limites = {}, {}
    for k in range(k_max + 1):
        try:
            bk = [n] if k == 0 else list(algo.predict(n_bkps=k))
        except Exception:
            break
        ssr = sum(algo.cost.error(a, b) for a, b in zip([0] + bk[:-1], bk))
        crit[k] = _lwz(max(ssr, 1e-12), n, q, k)
        limites[k] = bk
    k = min(crit, key=crit.get)
    return {"k": k, "quebras": limites[k][:-1], "limites": limites[k], "criterio": crit}


def _coefs(y, X, limites) -> list[np.ndarray]:
    out, ini = [], 0
    for fim in limites:
        out.append(np.linalg.lstsq(X[ini:fim], y[ini:fim], rcond=None)[0])
        ini = fim
    return out


def cusum(y, X) -> dict:
    """CUSUM dos resíduos de MQO (Ploberger e Krämer): p pequeno indica coeficientes instáveis no tempo."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    res = OLS(np.asarray(y, float), np.asarray(X, float)).fit()
    stat, p, _ = breaks_cusumolsresid(res.resid, ddof=int(np.asarray(X).shape[1]))
    return {"estatistica": float(stat), "p_valor": float(p)}


def chow_robusto(y, X, datas, data_evento, min_lado: int, lags: int) -> dict:
    """Teste de Chow na data do evento, com erros robustos a autocorrelação (Newey-West): todos os
    coeficientes podem mudar depois da data; testa se as mudanças são zero em conjunto."""
    from statsmodels.regression.linear_model import OLS

    depois = np.asarray(pd.DatetimeIndex(datas) >= pd.Timestamp(data_evento), dtype=float)
    n1, n2 = int((depois == 0).sum()), int(depois.sum())
    if n1 < min_lado or n2 < min_lado:
        return {"testavel": False, "n_antes": n1, "n_depois": n2}
    X = np.asarray(X, float)
    q = X.shape[1]
    Z = np.column_stack([X, X * depois[:, None]])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = OLS(np.asarray(y, float), Z).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
            w = res.wald_test(np.hstack([np.zeros((q, q)), np.eye(q)]), scalar=True, use_f=True)
        F, p = float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))
        if not (np.isfinite(F) and np.isfinite(p)):
            raise ValueError("teste indefinido")
        return {"testavel": True, "F": F, "p_valor": p, "n_antes": n1, "n_depois": n2}
    except Exception as e:  # matriz singular, por exemplo
        return {"testavel": False, "n_antes": n1, "n_depois": n2, "erro": e.__class__.__name__}


# ------------------------------------------------------------------ análise do ativo

def analisar(ticker: str, setor: str, base: pd.DataFrame, diario: pd.DataFrame, desd: pd.DataFrame, passo,
             imp: pd.DataFrame | None, hoje, cfg: dict) -> dict:
    c = cfg.get("estruturais", {})
    min_rel = int(c.get("meses_min_segmento_relacao", 36))
    min_preco = int(c.get("meses_min_segmento_preco", 24))
    max_q = int(c.get("quebras_max", 3))
    n_pred = int(c.get("preditores_na_relacao", 5))
    min_lado = int(c.get("meses_min_lado_evento", 24))
    lags = int(c.get("defasagens_hac", 12))
    validade = int(c.get("validade_pesquisa_dias", 7))
    hoje = pd.Timestamp(hoje).normalize()
    out = {"relacao": None, "preco": None, "series": [], "eventos": [], "marcas": [], "pesquisa": {}}

    # 1. Relação entre os maiores preditores de hoje e o valor de mercado (fim de cada mês).
    meses = [d for d in base.index if d == (d + pd.offsets.MonthEnd(0)).normalize() and d <= hoje]
    tab = base.loc[meses]
    top = []
    if imp is not None and len(imp):
        top = [v for v in imp["variavel"] if not str(v).startswith("_conv") and v in tab.columns][:n_pred]
    if not top:
        top = [v for v in variaveis_do_modelo(passo) if v in tab.columns][:n_pred]
    tab = tab[tab["log_valor_mercado"].notna()] if "log_valor_mercado" in tab else tab.iloc[0:0]
    rel = None
    if top and len(tab) >= 3 * min_rel:
        Xr = tab[top].astype(float)
        Xr = Xr.fillna(Xr.median())
        Xz = ((Xr - Xr.mean()) / Xr.std().replace(0, 1)).fillna(0.0).to_numpy()
        X = np.column_stack([np.ones(len(Xz)), Xz])
        y = tab["log_valor_mercado"].to_numpy(float)
        bp = bai_perron(y, X, min_rel, max_q)
        coefs = _coefs(y, X, bp["limites"])
        quebras = []
        for i, idx in enumerate(bp["quebras"]):
            delta = np.abs(coefs[i + 1][1:] - coefs[i][1:])
            quebras.append({"data": tab.index[idx],
                            "mudaram": [preditores.nome_preditor(top[j]) for j in np.argsort(-delta, kind="stable")[:2]]})
        rel = {"y": y, "X": X, "datas": tab.index}
        out["relacao"] = {"variaveis": top, "nomes": [preditores.nome_preditor(v) for v in top], "n": len(y),
                          "inicio": tab.index[0], "fim": tab.index[-1], "quebras": quebras, "cusum": cusum(y, X)}
        for q in quebras:
            out["marcas"].append({"data": q["data"].strftime("%Y-%m-%d"), "rotulo": "quebra: preditores × preço",
                                  "tipo": "quebra"})

    # 2. Preço: retorno médio e volatilidade (retornos mensais, só desdobramentos e grupamentos).
    d = diario[diario["data"] <= hoje]
    if len(d):
        aj = P.serie_ajustada(d, desd, hoje)
        r = np.log(aj.resample("ME").last().dropna()).diff().dropna()
        if len(r) >= 3 * min_preco:
            um = np.ones((len(r), 1))
            rv = r.to_numpy(float)
            bpm = bai_perron(rv, um, min_preco, max_q)
            bpv = bai_perron(np.abs(rv - rv.mean()), um, min_preco, max_q)

            def trechos(bp, f):
                lim, ini, out_ = bp["limites"], 0, []
                for fim in lim:
                    out_.append({"inicio": r.index[ini], "fim": r.index[fim - 1], "valor": float(f(rv[ini:fim]))})
                    ini = fim
                return out_

            out["preco"] = {
                "n": len(r), "inicio": r.index[0], "fim": r.index[-1],
                "quebras_media": [r.index[i] for i in bpm["quebras"]],
                "quebras_volatilidade": [r.index[i] for i in bpv["quebras"]],
                "trechos_media": trechos(bpm, lambda x: np.mean(x) * 12),
                "trechos_volatilidade": trechos(bpv, lambda x: np.std(x, ddof=1) * np.sqrt(12)),
                "cusum": cusum(rv, um)}
            for x in out["preco"]["quebras_media"]:
                out["marcas"].append({"data": x.strftime("%Y-%m-%d"), "rotulo": "quebra: retorno médio", "tipo": "quebra"})
            for x in out["preco"]["quebras_volatilidade"]:
                out["marcas"].append({"data": x.strftime("%Y-%m-%d"), "rotulo": "quebra: volatilidade", "tipo": "quebra"})

    # 3. Séries dos maiores preditores: mudanças de patamar (só na tabela do relatório).
    for v in top:
        s = base.loc[meses, v].dropna() if v in base.columns else pd.Series(dtype=float)
        if len(s) < 3 * min_preco or s.std() == 0:
            continue
        bp = bai_perron(s.to_numpy(float), np.ones((len(s), 1)), min_preco, max_q)
        out["series"].append({"variavel": v, "nome": preditores.nome_preditor(v),
                              "quebras": [s.index[i] for i in bp["quebras"]]})

    # 4. Eventos: config/eventos.yaml e a pesquisa na web da rodada, com o teste na data e se o modelo capta.
    pesquisa = ler_pesquisa(ticker)
    itens = {}
    for e in eventos_configurados(setor, ticker):
        n = _normalizar(e, "eventos.yaml")
        itens[(n["data"], n["rotulo"])] = n
    if pesquisa:
        for e in pesquisa.get("itens", []):
            n = _normalizar(e, "pesquisa")
            itens[(n["data"], n["rotulo"])] = n
        dp = pd.Timestamp(str(pesquisa.get("data_pesquisa"))[:10])
        idade = int((hoje - dp).days)
        out["pesquisa"] = {"situacao": "ok" if idade <= validade else "anterior", "data": dp, "idade_dias": idade,
                           "n": len(pesquisa.get("itens", [])), "janela": pesquisa.get("janela")}
    else:
        out["pesquisa"] = {"situacao": "ausente"}
    vars_modelo = variaveis_do_modelo(passo)
    for e in sorted(itens.values(), key=lambda e: (e["data"], e["rotulo"])):
        e["capta"] = capta(e, vars_modelo, hoje)
        if pd.Timestamp(e["data"]) > hoje:
            e["teste"] = {"testavel": False, "futuro": True}
        elif rel is not None:
            e["teste"] = chow_robusto(rel["y"], rel["X"], rel["datas"], e["data"], min_lado, lags)
        else:
            e["teste"] = {"testavel": False}
        out["eventos"].append(e)
        if pd.Timestamp(e["data"]) <= hoje:
            out["marcas"].append({"data": e["data"], "rotulo": e["rotulo"], "tipo": "evento"})
    return out
