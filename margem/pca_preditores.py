"""Diagnóstico de componentes principais (PCA) dos preditores candidatos.

Em cada retreino, só com os dados de treino até a data: quantos componentes resumem a variação das candidatas,
o que cada componente reúne (cargas), os grupos de preditores redundantes e os isolados. Cruzado com a importância
(SHAP e permutação) dos modelos escolhidos ao longo do tempo e com a frequência com que o LASSO escolheu cada
preditor, diz quais manter, quais reciclar num fator e quais são candidatos a sair.

É diagnóstico: não muda o preço justo nem a escolha do modelo. No walk-forward, a eliminação é feita pelo LASSO e a
reciclagem pelos candidatos sobre componentes principais, a cada retreino e só com os dados até a data. Tirar
preditores da configuração por causa desta tabela seria escolher olhando o período todo.

Regras da recomendação (definidas em 25/11/2026, antes de rodar):
- manter: SHAP ou permutação de 5% ou mais em média nas fotos anuais, SHAP de 5% ou mais hoje, ou escolhido pelo
  LASSO em metade ou mais dos retreinos dos últimos 10 anos;
- reciclar no fator: fora de "manter" e com correlação de postos de 0,90 ou mais com outro preditor (redundante),
  ou com 80% ou mais da variação explicada pelos fatores comuns (comunalidade);
- candidato a sair: fora dos dois casos, escolhido em menos de 10% dos retreinos e com SHAP médio abaixo de 1%;
- observar: os demais (pesam pouco, mas não são redundantes nem esquecidos).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import formatos
from .modelo import candidatos as C

LIMITE_MANTER = 0.05
FREQ_MANTER = 0.5
CORR_REDUNDANTE = 0.90
COMUNALIDADE_FATOR = 0.80
FREQ_SAIR = 0.10
IMP_SAIR = 0.01


def pca_na_data(base: pd.DataFrame, r, candidatas: list[str], cfg_modelo: dict,
                alvo: str = "log_valor_mercado") -> dict | None:
    """PCA das candidatas com os dados de treino do retreino de `r` (linhas antes de r)."""
    treino = base[base.index < pd.Timestamp(r)]
    if alvo in treino.columns:
        treino = treino[treino[alvo].notna()]
    cfg_sel = cfg_modelo["selecao_variaveis"]
    cols, med = C.com_cobertura(treino, candidatas, cfg_sel["cobertura_min"])
    if len(cols) < 3 or len(treino) < int(cfg_modelo["meses_treino_min"]):
        return None
    cfg_pca = cfg_modelo.get("componentes_principais") or {"variancia_explicada": 0.90}
    X = treino[cols].fillna(med).to_numpy(float)
    fat = C.Fatores(cfg_pca["variancia_explicada"], C.limite_variaveis(len(treino), cfg_sel)).fit(X)
    Z = fat.esc.transform(X)
    S = fat.pca.transform(Z)[:, : fat.k]
    Zc, Sc = Z - Z.mean(0), S - S.mean(0)
    den = np.outer(np.sqrt((Zc ** 2).sum(0)), np.sqrt((Sc ** 2).sum(0)))
    cargas = pd.DataFrame(np.where(den > 0, (Zc.T @ Sc) / np.where(den > 0, den, 1.0), 0.0), index=cols,
                          columns=[f"F{i + 1}" for i in range(fat.k)])
    # Componentes não se correlacionam: a soma dos quadrados das cargas é o R² de cada preditor nos k fatores.
    comunalidade = (cargas ** 2).sum(axis=1)
    fator = (cargas ** 2).idxmax(axis=1)
    carga_fator = pd.Series([cargas.at[v, fator[v]] for v in cols], index=cols)
    rk = treino[cols].fillna(med).rank()
    c = rk.corr().abs().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(c, 0.0)
    corr = pd.DataFrame(c, index=cols, columns=cols)
    par = corr.idxmax(axis=1)
    max_corr = corr.max(axis=1)
    return {"data": pd.Timestamp(r), "n_treino": int(len(treino)), "variaveis": cols, "k": int(fat.k),
            "variancia": [float(x) for x in fat.pca.explained_variance_ratio_[:10]],
            "acumulada_k": float(np.sum(fat.pca.explained_variance_ratio_[: fat.k])),
            "cargas": cargas, "comunalidade": comunalidade, "fator": fator, "carga_fator": carga_fator,
            "grupos": _grupos(corr, CORR_REDUNDANTE), "max_corr": max_corr, "par": par}


def _grupos(corr: pd.DataFrame, limite: float) -> list[list[str]]:
    """Componentes conexos do grafo "correlação de postos ≥ limite" (grupos de 2 ou mais)."""
    nomes = list(corr.index)
    vistos: set[str] = set()
    grupos = []
    for v in nomes:
        if v in vistos:
            continue
        pilha, g = [v], []
        vistos.add(v)
        while pilha:
            x = pilha.pop()
            g.append(x)
            for y in nomes:
                if y not in vistos and corr.at[x, y] >= limite:
                    vistos.add(y)
                    pilha.append(y)
        if len(g) >= 2:
            grupos.append(sorted(g))
    return sorted(grupos, key=lambda g: (-len(g), g))


def no_tempo(base: pd.DataFrame, datas, candidatas: list[str], cfg_modelo: dict) -> list[dict]:
    """Resumo da PCA em cada retreino (só com os dados até a data de cada um)."""
    out = []
    for r in sorted(pd.to_datetime(list(datas))):
        d = pca_na_data(base, r, candidatas, cfg_modelo)
        if d is None:
            continue
        top = d["cargas"]["F1"].abs().sort_values(ascending=False).head(3).index.tolist()
        out.append({"data": d["data"], "n": len(d["variaveis"]), "k": d["k"], "acumulada_k": d["acumulada_k"],
                    "f1": d["variancia"][0], "f2": d["variancia"][1] if len(d["variancia"]) > 1 else np.nan,
                    "f3": d["variancia"][2] if len(d["variancia"]) > 2 else np.nan,
                    "grupos": len(d["grupos"]), "em_grupos": sum(len(g) for g in d["grupos"]), "top_f1": top})
    return out


def frequencia_lasso(variaveis_retreino: list[dict], inicio, alvos: list[str]) -> tuple[pd.Series, int]:
    """Fração dos retreinos (desde `inicio`) em que o LASSO escolheu cada preditor, para algum dos alvos."""
    por_data: dict = {}
    for v in variaveis_retreino:
        if v["alvo"] not in alvos:
            continue
        d = pd.Timestamp(v["data"])
        if inicio is not None and d < pd.Timestamp(inicio):
            continue
        por_data.setdefault(d, set()).update(v["variaveis"])
    n = len(por_data)
    cont: dict[str, int] = {}
    for vs in por_data.values():
        for x in vs:
            cont[x] = cont.get(x, 0) + 1
    return (pd.Series({k: c / n for k, c in cont.items()}, dtype=float) if n else pd.Series(dtype=float)), n


def importancia_media(fotos: list[pd.DataFrame]) -> tuple[pd.Series, pd.Series]:
    """Média da participação (SHAP e permutação) em cada foto; ausente numa foto conta como zero."""
    if not fotos:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    shap = pd.concat([f.set_index("variavel")["importancia"] for f in fotos], axis=1).fillna(0.0)
    perm = pd.concat([f.set_index("variavel")["importancia_permutacao"] for f in fotos], axis=1).fillna(0.0)
    return shap.sum(axis=1) / len(fotos), perm.sum(axis=1) / len(fotos)


def recomendar(hoje: dict, imp_hoje: pd.DataFrame | None, shap_medio: pd.Series, perm_media: pd.Series,
               freq: pd.Series, nome, familia) -> pd.DataFrame:
    """Tabela por preditor com o fator, a redundância, a importância no tempo e a recomendação."""
    shap_h = imp_hoje.set_index("variavel")["importancia"] if imp_hoje is not None else pd.Series(dtype=float)
    perm_h = (imp_hoje.set_index("variavel")["importancia_permutacao"] if imp_hoje is not None
              else pd.Series(dtype=float))
    em_grupo = {v: g for g in hoje["grupos"] for v in g}
    linhas = []
    for v in hoje["variaveis"]:
        sm, pm = float(shap_medio.get(v, 0.0)), float(perm_media.get(v, 0.0))
        sh, ph = float(shap_h.get(v, 0.0)), float(perm_h.get(v, 0.0))
        fq = float(freq.get(v, 0.0))
        h2 = float(hoje["comunalidade"][v])
        mc, par = float(hoje["max_corr"][v]), hoje["par"][v]
        if sm >= LIMITE_MANTER or pm >= LIMITE_MANTER or sh >= LIMITE_MANTER or fq >= FREQ_MANTER:
            rec = "manter"
            partes = []
            if sm >= LIMITE_MANTER or pm >= LIMITE_MANTER:
                partes.append(f"pesa {formatos.pct(max(sm, pm), 0)} em média nas fotos anuais")
            if sh >= LIMITE_MANTER:
                partes.append(f"{formatos.pct(sh, 0)} no modelo de hoje")
            if fq >= FREQ_MANTER:
                partes.append(f"escolhido pelo LASSO em {formatos.pct(fq, 0)} dos retreinos")
            motivo = "; ".join(partes)
            if v in em_grupo:
                motivo += f"; representa o grupo redundante com {_lista(em_grupo[v], v, nome)}"
        elif v in em_grupo:
            rec = "reciclar no fator"
            motivo = (f"repete a informação de {nome(par)} (correlação {formatos.num(mc, 2)}); entra pelo fator "
                      f"{hoje['fator'][v]}")
        elif h2 >= COMUNALIDADE_FATOR:
            rec = "reciclar no fator"
            motivo = (f"{formatos.pct(h2, 0)} da variação dele está nos fatores comuns (sobretudo {hoje['fator'][v]}) e "
                      "ele pesa pouco sozinho")
        elif fq < FREQ_SAIR and sm < IMP_SAIR:
            rec = "candidato a sair"
            motivo = (f"escolhido em {formatos.pct(fq, 0)} dos retreinos, SHAP médio de {formatos.pct(sm, 1)} e pouca "
                      f"informação em comum com os demais (comunalidade {formatos.pct(h2, 0)})")
        else:
            rec = "observar"
            motivo = (f"pesa pouco ({formatos.pct(sm, 1)} em média), escolhido em {formatos.pct(fq, 0)} dos retreinos, "
                      f"comunalidade {formatos.pct(h2, 0)}")
        linhas.append({"variavel": v, "nome": nome(v), "familia": familia(v), "fator": hoje["fator"][v],
                       "carga": float(hoje["carga_fator"][v]), "comunalidade": h2, "max_corr": mc, "par": nome(par),
                       "freq_lasso": fq, "shap_medio": sm, "perm_media": pm, "shap_hoje": sh, "perm_hoje": ph,
                       "recomendacao": rec, "motivo": motivo})
    ordem = {"manter": 0, "observar": 1, "reciclar no fator": 2, "candidato a sair": 3}
    tab = pd.DataFrame(linhas)
    if len(tab):
        tab["_o"] = tab["recomendacao"].map(ordem)
        tab = tab.sort_values(["_o", "shap_medio", "freq_lasso", "nome"], ascending=[True, False, False, True])
        tab = tab.drop(columns="_o").reset_index(drop=True)
    return tab


def _lista(grupo: list[str], v: str, nome) -> str:
    outros = [nome(x) for x in grupo if x != v]
    return ", ".join(outros[:4]) + (f" e mais {len(outros) - 4}" if len(outros) > 4 else "")


def fatores_de_hoje(hoje: dict, nome, n_top: int = 4) -> list[dict]:
    """O que cada fator de hoje reúne: os preditores com as maiores cargas (correlação com o fator)."""
    out = []
    for i, f in enumerate(hoje["cargas"].columns):
        c = hoje["cargas"][f]
        top = c.abs().sort_values(ascending=False).head(n_top).index
        out.append({"fator": f, "variancia": hoje["variancia"][i] if i < len(hoje["variancia"]) else np.nan,
                    "cargas": [{"nome": nome(v), "carga": float(c[v])} for v in top]})
    return out


NOTA = ("PCA das candidatas padronizadas, só com os dados de treino de cada data (o mesmo cálculo dos candidatos sobre "
        "componentes principais). Carga = correlação do preditor com o fator; comunalidade = parte da variação do "
        "preditor explicada pelos fatores retidos. Redundantes: correlação de postos de 0,90 ou mais. Importância no "
        "tempo: média das fotos anuais do modelo escolhido em cada data (SHAP e permutação, ausente = zero). "
        "Diagnóstico, só para leitura: não muda o preço justo nem a escolha do modelo, que já elimina (LASSO) e "
        "recicla (componentes principais) a cada retreino. Tirar preditores da configuração por causa desta tabela "
        "seria escolher olhando o período todo.")
