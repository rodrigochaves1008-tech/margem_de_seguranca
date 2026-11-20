"""Backtest (contra o preço 6 e 12 meses depois), erro típico, faixa e critério de estimativa confiável."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import precos as P
from .modelo import candidatos as C


def _mais(d, h):
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(h)).normalize()


def janela_backtest(mensal: pd.DataFrame, ate: pd.Timestamp, anos: int) -> pd.DataFrame:
    ini = (pd.Timestamp(ate) - pd.DateOffset(years=anos)).normalize()
    return mensal[(mensal.index > ini) & mensal["V"].notna()]


def comparar_futuro(serie_v: pd.Series, preco_mensal: pd.Series, desd: pd.DataFrame, h: int,
                    base_hoje: pd.Timestamp) -> dict:
    """MAPE, RMSE (R$, na base de hoje) e R² do preço justo contra o preço h meses depois."""
    fut, est = [], []
    for s, v in serie_v.items():
        alvo = _mais(s, h)
        if alvo not in preco_mensal.index or not np.isfinite(v):
            continue
        f_s_alvo = P.fator_entre(desd, s, alvo)
        f_alvo_hoje = P.fator_entre(desd, alvo, base_hoje)
        fut.append(preco_mensal[alvo] / f_alvo_hoje)
        est.append(v / f_s_alvo / f_alvo_hoje)
    fut, est = np.array(fut), np.array(est)
    if len(fut) < 6:
        return {"n": len(fut), "mape": np.nan, "rmse": np.nan, "r2": np.nan}
    sse = float(np.sum((fut - est) ** 2))
    sst = float(np.sum((fut - fut.mean()) ** 2))
    return {"n": int(len(fut)), "mape": float(np.mean(np.abs(fut - est) / fut)),
            "rmse": float(np.sqrt(sse / len(fut))), "r2": 1 - sse / sst if sst > 0 else np.nan}


def tabela_backtest(mensal: pd.DataFrame, preco_mensal: pd.Series, desd: pd.DataFrame, base_hoje,
                    alvos: list[str], horizontes=(6, 12)) -> pd.DataFrame:
    linhas = []
    series = {"escolha do walk-forward": mensal["V"], "passeio aleatório (último preço)": mensal["preco"]}
    for a in alvos:
        for m in list(C.MODELOS_BASE) + ["ensemble_todos"]:
            col = f"V__{m}|{a}"
            if col in mensal:
                series[f"{C.NOMES_MODELOS[m]} · alvo {C.NOMES_ALVOS[a]}"] = mensal[col]
    if "V__ensemble_geral|todos" in mensal:
        series[C.NOMES_MODELOS["ensemble_geral"]] = mensal["V__ensemble_geral|todos"]
    for nome, s in series.items():
        reg = {"modelo": nome}
        for h in horizontes:
            r = comparar_futuro(s, preco_mensal, desd, h, base_hoje)
            for k, v in r.items():
                reg[f"{k}_{h}m"] = v
        linhas.append(reg)
    return pd.DataFrame(linhas)


def erro_tipico(janela: pd.DataFrame) -> float:
    e = (janela["V"] - janela["preco"]).abs() / janela["V"]
    return float(e.median()) if len(e) else np.nan


def cobertura_faixa(janela: pd.DataFrame) -> tuple[float, int]:
    x = janela.dropna(subset=["P10", "P90"])
    if x.empty:
        return np.nan, 0
    dentro = ((x["preco"] >= x["P10"]) & (x["preco"] <= x["P90"])).mean()
    return float(dentro), int(len(x))


def razoes_contra_passeio(mensal: pd.DataFrame, preco_mensal: pd.Series, desd: pd.DataFrame, ate,
                          janela_meses: int, horizontes=(6, 12)) -> dict:
    """MAPE do modelo ÷ MAPE do passeio aleatório nos últimos meses com resultado fechado."""
    out = {}
    for h in horizontes:
        ss = [s for s in mensal.index if _mais(s, h) <= pd.Timestamp(ate) and _mais(s, h) in preco_mensal.index]
        ss = ss[-janela_meses:]
        x = mensal.loc[ss]
        m = comparar_futuro(x["V"], preco_mensal, desd, h, ate)["mape"]
        rw = comparar_futuro(x["preco"], preco_mensal, desd, h, ate)["mape"]
        out[h] = {"mape_modelo": m, "mape_passeio": rw, "razao": m / rw if rw and np.isfinite(rw) else np.nan,
                  "n": len(ss)}
    return out


def avaliar(mensal, hoje: dict, linha_hoje: pd.Series, base: pd.DataFrame, preco_mensal, desd, cfg: dict,
            obrigatorios: list[str]) -> dict:
    """Critério de estimativa confiável (o mesmo para todos os ativos) com os motivos."""
    c = cfg["estimativa_confiavel"]
    anos = int(cfg["modelo"]["anos_fora_amostra"])
    ate = pd.Timestamp(hoje["data"])
    jan = janela_backtest(mensal, ate, anos) if len(mensal) else mensal
    res = {"meses_fora_amostra": int(len(jan)), "inicio_fora_amostra": jan.index.min() if len(jan) else None}
    res["erro_tipico"] = erro_tipico(jan) if len(jan) else np.nan
    res["cobertura_faixa"], res["meses_com_faixa"] = cobertura_faixa(jan) if len(jan) else (np.nan, 0)
    res["contra_passeio"] = razoes_contra_passeio(mensal, preco_mensal, desd, ate, int(c["janela_comparacao_meses"])) if len(mensal) else {}
    motivos = []
    V, preco = hoje.get("V", np.nan), hoje.get("preco", np.nan)
    if not np.isfinite(V):
        motivos.append("o modelo não produziu preço justo na data")
    if res["meses_fora_amostra"] < int(c["meses_fora_amostra_min"]):
        motivos.append(f"só {res['meses_fora_amostra']} meses de estimativas fora da amostra "
                       f"(mínimo {c['meses_fora_amostra_min']})")
    if not np.isfinite(hoje.get("q10", np.nan)):
        motivos.append(f"menos de {c['meses_faixa_min']} meses para calcular a faixa de 80%")
    r12 = res["contra_passeio"].get(12, {}).get("razao", np.nan)
    r6 = res["contra_passeio"].get(6, {}).get("razao", np.nan)
    if not (np.isfinite(r12) and r12 <= float(c["razao_mape_12m_max"])):
        motivos.append(f"perde para o passeio aleatório no preço 12 meses depois (erro {fmt_razao(r12)} do passeio; "
                       f"limite {fmt_razao(c['razao_mape_12m_max'])})")
    if not (np.isfinite(r6) and r6 <= float(c["razao_mape_6m_max"])):
        motivos.append(f"perde para o passeio aleatório no preço 6 meses depois (erro {fmt_razao(r6)} do passeio; "
                       f"limite {fmt_razao(c['razao_mape_6m_max'])})")
    cob = res["cobertura_faixa"]
    if not (np.isfinite(cob) and float(c["cobertura_faixa_min"]) <= cob <= float(c["cobertura_faixa_max"])):
        motivos.append(f"o preço ficou dentro da faixa de 80% em {fmt_pct(cob)} dos meses "
                       f"(aceito: {fmt_pct(c['cobertura_faixa_min'])} a {fmt_pct(c['cobertura_faixa_max'])})")
    et = res["erro_tipico"]
    if not (np.isfinite(et) and et <= float(c["erro_tipico_max"])):
        motivos.append(f"erro típico de {fmt_pct(et)} (máximo {fmt_pct(c['erro_tipico_max'])})")
    dt_ref = linha_hoje.get("dt_refer")
    if dt_ref is None or pd.isna(dt_ref) or (ate - pd.Timestamp(dt_ref)).days > int(c["dias_ultima_divulgacao_max"]):
        motivos.append("o último balanço usado tem mais de "
                       f"{c['dias_ultima_divulgacao_max']} dias")
    recente = base[(base.index > ate - pd.DateOffset(months=60)) & (base.index <= ate)]
    for o in obrigatorios:
        if o in recente and recente[o].notna().mean() < float(c["cobertura_obrigatorios_min"]):
            motivos.append(f"preditor obrigatório '{o}' com cobertura de {fmt_pct(recente[o].notna().mean())}")
        elif o not in recente:
            motivos.append(f"preditor obrigatório '{o}' sem dados")
    if int(linha_hoje.get("pregoes_sem_negocio", 0)) >= int(c["pregoes_sem_negocio_max"]):
        motivos.append("sem negócios nos últimos pregões")
    if np.isfinite(V) and np.isfinite(preco) and preco > 0:
        rz = V / preco
        if not (float(c["razao_v_p_min"]) <= rz <= float(c["razao_v_p_max"])):
            motivos.append(f"preço justo {rz:.1f} vezes o preço (fora do intervalo de sanidade)")
    vs = hoje.get("variacao_sementes", 0.0)
    if np.isfinite(vs) and vs > float(c["variacao_sementes_max"]):
        motivos.append(f"a rede neural muda {fmt_pct(vs)} entre sementes (instável)")
    res["motivos"] = motivos
    res["confiavel"] = not motivos
    return res


def fmt_pct(v) -> str:
    from .formatos import pct
    return pct(v, 0)


def fmt_razao(v) -> str:
    from .formatos import pct
    return pct(v, 0)
