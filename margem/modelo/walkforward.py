"""Walk-forward: em cada mês, o preço justo só com o que era conhecido até ali.

- Retreino no fim do mês em que saiu um documento novo na CVM (ITR ou DFP); nos outros meses,
  estima com o último modelo treinado. O treino de um retreino na data r usa só meses s < r:
  o preço do mês estimado nunca entra no treino que o estima.
- Em cada retreino, tudo é refeito com os dados até ali: seleção de variáveis, normalização,
  hiperparâmetros, early stopping e a escolha entre alvos, modelos e ensembles. A escolha usa só
  resultados de 6 e 12 meses já fechados na data (preço futuro já conhecido).
- O último passo é a data da análise (hoje).
Todas as estimativas ficam na base de ações da própria data (sem desdobramentos posteriores).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import precos as P
from . import candidatos as C


def converter(alvo: str, yhat: np.ndarray, linhas: pd.DataFrame) -> np.ndarray:
    """Do espaço do alvo (log) para preço por ação na base da data de cada linha."""
    vm = np.exp(yhat)
    if alvo == "log_pl":
        vm = vm * linhas["lucro_12m"].to_numpy(float)
    elif alvo == "log_pvp":
        vm = vm * linhas["patrimonio"].to_numpy(float)
    elif alvo == "log_ev_ebitda":
        ev = vm * linhas["ebitda_12m"].to_numpy(float)
        vm = ev - linhas["divida_liquida"].to_numpy(float) - linhas.get("minoritarios", pd.Series(0.0, index=linhas.index)).fillna(0).to_numpy(float)
        vm = np.where(vm > 0, vm, np.nan)
    return vm / linhas["acoes"].to_numpy(float)


def nomes_candidatos(alvos: list[str]) -> list[str]:
    nomes = [f"{m}|{a}" for a in alvos for m in list(C.MODELOS) + list(C.ENSEMBLES)]
    return nomes + ["ensemble_geral|todos"]


@dataclass
class Passo:
    data: pd.Timestamp
    retreino: pd.Timestamp
    ajustes: dict          # alvo -> Ajuste
    escolhido: str
    pontuacoes: dict


@dataclass
class Resultado:
    mensal: pd.DataFrame                 # meses fora da amostra
    escolhas: pd.DataFrame               # uma linha por retreino
    hoje: dict                           # último passo (data da análise)
    passos_hoje: Passo | None = None
    variaveis_por_retreino: list = field(default_factory=list)


def _todas_previsoes(ajustes: dict, linhas: pd.DataFrame, alvos: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    logs_base = []
    for a in alvos:
        aj = ajustes.get(a)
        if aj is None:
            continue
        pr = C.prever(aj, linhas)
        for m, yhat in pr.items():
            v = converter(a, yhat, linhas)
            out[f"{m}|{a}"] = v
            if m in C.MODELOS_BASE:
                logs_base.append(np.log(v))
    if logs_base:
        out["ensemble_geral|todos"] = np.exp(np.mean(logs_base, axis=0))
    return out


def _mais_meses(d: pd.Timestamp, h: int) -> pd.Timestamp:
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(h)).normalize()


def escolher(r: pd.Timestamp, previsoes: pd.DataFrame, preco_mensal: pd.Series, desd: pd.DataFrame,
             cfg_escolha: dict, candidatos: list[str]) -> tuple[str, dict]:
    """Candidato com menor erro médio (MAPE) contra o preço 6 e 12 meses depois, só com resultados
    já fechados em r. Sem resultados suficientes, usa o candidato padrão."""
    horizontes = [int(h) for h in cfg_escolha["horizontes_meses"]]
    janela = int(cfg_escolha["janela_meses"])
    minimo = int(cfg_escolha["meses_fechados_min"])
    pont = {}
    validos = {}
    for h in horizontes:
        ss = [s for s in previsoes.index if _mais_meses(s, h) <= r and _mais_meses(s, h) in preco_mensal.index]
        ss = ss[-janela:]
        validos[h] = ss
    if len(validos[max(horizontes)]) < minimo:
        padrao = cfg_escolha["candidato_padrao"]
        return (padrao if padrao in candidatos else candidatos[0]), {}
    for c in candidatos:
        if c not in previsoes.columns:
            continue
        erros = []
        for h in horizontes:
            e = []
            for s in validos[h]:
                v = previsoes.at[s, c]
                fut = preco_mensal[_mais_meses(s, h)]
                if np.isfinite(v) and fut > 0:
                    e.append(abs(fut - v / P.fator_entre(desd, s, _mais_meses(s, h))) / fut)
            if not e:
                break
            erros.append(float(np.mean(e)))
        if len(erros) == len(horizontes):
            pont[c] = float(np.mean(erros))
    if not pont:
        padrao = cfg_escolha["candidato_padrao"]
        return (padrao if padrao in candidatos else candidatos[0]), {}
    melhor = min(pont, key=lambda c: (pont[c], c))
    return melhor, pont


def faixa(razoes: np.ndarray, minimo: int) -> tuple[float, float]:
    razoes = razoes[np.isfinite(razoes)]
    if len(razoes) < minimo:
        return np.nan, np.nan
    return float(np.quantile(razoes, 0.10)), float(np.quantile(razoes, 0.90))


def datas_retreino(base: pd.DataFrame, divulgacoes: pd.Series) -> set[pd.Timestamp]:
    """Fins de mês em que entrou documento novo na CVM."""
    out = set()
    meses = [d for d in base.index if d == (d + pd.offsets.MonthEnd(0)).normalize()]
    anterior = pd.Timestamp("1900-01-01")
    divs = pd.to_datetime(pd.Series(divulgacoes)).sort_values().to_numpy()
    for m in meses:
        if ((divs > np.datetime64(anterior)) & (divs <= np.datetime64(m))).any():
            out.add(m)
        anterior = m
    return out


def rodar(base: pd.DataFrame, alvos: list[str], candidatas_vars: list[str], cfg: dict,
          divulgacoes: pd.Series, desd: pd.DataFrame, ate: pd.Timestamp,
          anteriores: pd.DataFrame | None = None, escolhas_anteriores: pd.DataFrame | None = None,
          progresso=print) -> Resultado:
    """Modelos só do próprio ativo: o histórico dos indicadores da empresa e as variáveis de mercado,
    macro e do setor. Nenhum dado de outra empresa entra no treino."""
    cfg_m = cfg["modelo"]
    cfg_conf = cfg["estimativa_confiavel"]
    ate = pd.Timestamp(ate).normalize()
    base = base[base.index <= ate]
    meses = [d for d in base.index if d == (d + pd.offsets.MonthEnd(0)).normalize() and d <= ate]
    hoje_e_fim_de_mes = bool(meses) and meses[-1] == ate
    retreinos = datas_retreino(base, divulgacoes)
    candidatos = nomes_candidatos(alvos)
    preco_mensal = base.loc[meses, "preco"]

    # Retomada: reaproveita os meses já calculados com a mesma versão do modelo, refazendo a partir
    # do último retreino guardado (para recuperar o modelo em uso).
    prev_rows: list[dict] = []
    escolhas_rows: list[dict] = []
    inicio = None
    if anteriores is not None and len(anteriores) and escolhas_anteriores is not None and len(escolhas_anteriores):
        ult_retreino = pd.Timestamp(escolhas_anteriores["data"].max())
        manter = anteriores[anteriores.index < ult_retreino]
        if len(manter):
            prev_rows = [dict(r, data=i) for i, r in manter.iterrows()]
            escolhas_rows = escolhas_anteriores[escolhas_anteriores["data"] < ult_retreino].to_dict("records")
            inicio = ult_retreino

    previsoes = pd.DataFrame([{**{c: r.get(f"V__{c}", np.nan) for c in candidatos}, "data": r["data"]}
                              for r in prev_rows]).set_index("data") if prev_rows else pd.DataFrame(columns=candidatos)
    linhas_out = list(prev_rows)
    variaveis_log = []

    passo: Passo | None = None
    a_fazer = [m for m in meses if inicio is None or m >= inicio]
    total_retreinos = len([m for m in a_fazer if m in retreinos])
    feitos = 0
    for m in a_fazer:
        if m in retreinos or (passo is None and inicio is not None):
            treino = base[base.index < m]
            ajustes = {a: C.ajustar(treino, a, candidatas_vars, cfg_m) for a in alvos}
            ajustes = {a: aj for a, aj in ajustes.items() if aj is not None}
            if ajustes:
                escolhido, pont = escolher(m, previsoes, preco_mensal, desd, cfg_m["escolha"], candidatos)
                passo = Passo(data=m, retreino=m, ajustes=ajustes, escolhido=escolhido, pontuacoes=pont)
                escolhas_rows.append({"data": m, "escolhido": escolhido,
                                      "pontuacao": pont.get(escolhido, np.nan), "n_resultados": len(pont),
                                      "n_treino": max(aj.n_treino for aj in ajustes.values())})
                for a, aj in ajustes.items():
                    variaveis_log.append({"data": m, "alvo": a, "variaveis": aj.variaveis})
                feitos += 1
                if progresso and (feitos % 5 == 0 or feitos == total_retreinos):
                    progresso(f"    retreinos: {feitos} de {total_retreinos} ({m:%m/%Y})")
        if passo is None:
            continue
        linha = base.loc[[m]]
        vs = _todas_previsoes(passo.ajustes, linha, alvos)
        reg = {"data": m, "preco": float(linha["preco"].iloc[0]), "retreino": passo.retreino,
               "escolhido": passo.escolhido}
        for c in candidatos:
            reg[f"V__{c}"] = float(vs[c][0]) if c in vs else np.nan
        reg["V"] = reg.get(f"V__{passo.escolhido}", np.nan)
        linhas_out.append(reg)
        previsoes.loc[m, [c for c in candidatos]] = [reg[f"V__{c}"] for c in candidatos]

    mensal = pd.DataFrame(linhas_out).set_index("data").sort_index() if linhas_out else pd.DataFrame()
    if len(mensal):
        mensal = _completar_faixa(mensal, int(cfg_conf["meses_faixa_min"]))

    # Último passo: a data da análise.
    hoje: dict = {"data": ate}
    passo_hoje = passo
    if not hoje_e_fim_de_mes and passo is not None:
        ultimo_mes = meses[-1] if meses else None
        divs = pd.to_datetime(pd.Series(divulgacoes))
        novo_doc = bool(((divs > (ultimo_mes if ultimo_mes is not None else pd.Timestamp("1900-01-01"))) & (divs <= ate)).any())
        if novo_doc:
            treino = base[base.index < ate]
            ajustes = {a: C.ajustar(treino, a, candidatas_vars, cfg_m) for a in alvos}
            ajustes = {a: aj for a, aj in ajustes.items() if aj is not None}
            escolhido, pont = escolher(ate, previsoes, preco_mensal, desd, cfg_m["escolha"], candidatos)
            passo_hoje = Passo(data=ate, retreino=ate, ajustes=ajustes, escolhido=escolhido, pontuacoes=pont)
    if passo_hoje is not None:
        linha = base.loc[[ate]]
        vs = _todas_previsoes(passo_hoje.ajustes, linha, alvos)
        V = float(vs.get(passo_hoje.escolhido, [np.nan])[0])
        hoje.update({
            "preco": float(linha["preco"].iloc[0]), "data_preco": linha["data_preco"].iloc[0],
            "V": V, "escolhido": passo_hoje.escolhido, "retreino": passo_hoje.retreino,
            "V_candidatos": {c: float(v[0]) for c, v in vs.items()},
            "pontuacoes": passo_hoje.pontuacoes,
        })
        razoes = (mensal["preco"] / mensal["V"]).to_numpy() if len(mensal) else np.array([])
        q10, q90 = faixa(razoes, int(cfg_conf["meses_faixa_min"]))
        hoje["q10"], hoje["q90"] = q10, q90
        hoje["P10"], hoje["P90"] = V * q10, V * q90
        hoje["variacao_sementes"] = _variacao_sementes(passo_hoje, linha)
    return Resultado(mensal=mensal, escolhas=pd.DataFrame(escolhas_rows), hoje=hoje,
                     passos_hoje=passo_hoje, variaveis_por_retreino=variaveis_log)


def _completar_faixa(mensal: pd.DataFrame, minimo: int) -> pd.DataFrame:
    """Faixa de 80% de cada mês com os meses fora da amostra até ele (inclusive)."""
    mensal = mensal.copy()
    razoes = (mensal["preco"] / mensal["V"]).to_numpy()
    q10s, q90s = [], []
    for i in range(len(mensal)):
        q10, q90 = faixa(razoes[: i + 1], minimo)
        q10s.append(q10)
        q90s.append(q90)
    mensal["q10"], mensal["q90"] = q10s, q90s
    mensal["P10"] = mensal["V"] * mensal["q10"]
    mensal["P90"] = mensal["V"] * mensal["q90"]
    return mensal


def _variacao_sementes(passo: Passo, linha: pd.DataFrame) -> float:
    """Coeficiente de variação do preço justo entre as sementes das redes neurais do candidato escolhido
    (só checagem). Cada rede é medida sozinha, mesmo dentro de uma média com outros modelos."""
    modelo, alvo = passo.escolhido.split("|")
    redes = [m for m in C.membros(modelo) if m in C.REDES]
    if not redes:
        return 0.0
    cvs = []
    for a, aj in passo.ajustes.items():
        if alvo != "todos" and a != alvo:
            continue
        for m in redes:
            if m not in aj.modelos:
                continue
            por_semente = aj.modelos[m].prever_por_semente(aj.matriz(linha, m))[:, 0]
            v = converter(a, por_semente, pd.concat([linha] * len(por_semente)))
            v = v[np.isfinite(v)]
            if len(v) and np.mean(v) > 0:
                cvs.append(float(np.std(v) / np.mean(v)))
    # Maior variação entre sementes, rede a rede e alvo a alvo (sem misturar a diferença entre alvos).
    return max(cvs) if cvs else 0.0


def refazer_passo(base: pd.DataFrame, r: pd.Timestamp, alvos: list[str], candidatas_vars: list[str], cfg: dict,
                  escolhido: str) -> Passo | None:
    """Refaz os ajustes de um retreino passado (determinístico: igual ao original), para explicar
    o modelo daquela data (mudança do ranking dos preditores)."""
    treino = base[base.index < r]
    ajustes = {a: C.ajustar(treino, a, candidatas_vars, cfg["modelo"]) for a in alvos}
    ajustes = {a: aj for a, aj in ajustes.items() if aj is not None}
    if not ajustes:
        return None
    return Passo(data=r, retreino=r, ajustes=ajustes, escolhido=escolhido, pontuacoes={})
