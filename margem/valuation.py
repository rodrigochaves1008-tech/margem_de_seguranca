"""Checagem com valuation tradicional: modelo de dividendos e P/VP justificado (bancos),
múltiplos contra o histórico, Graham e Bazin. Pares entram na Fase 2."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .formatos import moeda, num, pct


def calcular(base: pd.DataFrame, data_hoje, fundamentos_hist: pd.DataFrame | None, cfg: dict,
             contabilidade: str = "banco", bases_pares: dict | None = None) -> dict:
    v = cfg["valuation"]
    linha = base.loc[pd.Timestamp(data_hoje)]
    acoes = linha.get("acoes", np.nan)
    lucro = linha.get("lucro_12m", np.nan)
    pl = linha.get("patrimonio", np.nan)
    preco = linha.get("preco", np.nan)
    dpa = linha.get("proventos_12m", np.nan)
    lpa = lucro / acoes if np.isfinite(acoes) and acoes > 0 else np.nan
    vpa = pl / acoes if np.isfinite(acoes) and acoes > 0 else np.nan
    juro_real = linha.get("juro_real_longo", np.nan) / 100
    infl = linha.get("focus_ipca_12m", np.nan) / 100
    ke = (1 + juro_real + float(v["premio_risco_acoes"])) * (1 + infl) - 1
    g = (1 + float(v["crescimento_real_longo_prazo"])) * (1 + infl) - 1
    # ROE sustentável: mediana dos últimos 5 anos de ROE conhecidos na data.
    roe_hist = base.loc[(base.index > pd.Timestamp(data_hoje) - pd.DateOffset(years=5)) & (base.index <= pd.Timestamp(data_hoje)), "roe"].dropna()
    roe_s = float(roe_hist.median()) if len(roe_hist) else np.nan
    metodos = []

    def add(nome, valor, premissas):
        metodos.append({"metodo": nome, "valor": float(valor) if valor is not None and np.isfinite(valor) else np.nan,
                        "premissas": premissas})

    if contabilidade == "banco":
        if np.isfinite(ke) and np.isfinite(g) and ke > g:
            add("P/VP justificado (Gordon)", vpa * (roe_s - g) / (ke - g) if np.isfinite(roe_s) else np.nan,
                f"ROE sustentável {pct(roe_s, 1)} (mediana de 5 anos), custo de capital {pct(ke, 1)}, crescimento {pct(g, 1)}")
            add("Modelo de dividendos", dpa * (1 + g) / (ke - g) if np.isfinite(dpa) else np.nan,
                f"proventos de 12 meses {moeda(dpa)} crescendo {pct(g, 1)} ao ano, custo de capital {pct(ke, 1)}")
        else:
            add("P/VP justificado (Gordon)", np.nan, "custo de capital ou inflação esperada indisponível")
            add("Modelo de dividendos", np.nan, "custo de capital ou inflação esperada indisponível")
    else:
        # Fluxo de caixa descontado simples (FCFF de 12 meses em perpetuidade), custo médio ponderado de capital.
        fcl = linha.get("fcl_12m", np.nan)
        dl = linha.get("divida_liquida", np.nan)
        minor = linha.get("minoritarios", 0.0) or 0.0
        vm = linha.get("valor_mercado", np.nan)
        selic = linha.get("selic", np.nan) / 100
        kd = (selic + 0.02) * (1 - 0.34) if np.isfinite(selic) else np.nan
        divida = max(dl, 0.0) if np.isfinite(dl) else np.nan
        wacc = (ke * vm + kd * divida) / (vm + divida) if all(np.isfinite([ke, kd, vm, divida])) and vm + divida > 0 else np.nan
        if np.isfinite(wacc) and np.isfinite(g) and wacc > g and np.isfinite(fcl) and fcl > 0 and np.isfinite(dl):
            ev = fcl * (1 + g) / (wacc - g)
            add("Fluxo de caixa descontado", (ev - dl - minor) / acoes,
                f"fluxo de caixa livre de 12 meses R$ {num(fcl / 1e9, 2)} bi crescendo {pct(g, 1)} ao ano, "
                f"custo médio de capital {pct(wacc, 1)}, dívida líquida R$ {num(dl / 1e9, 2)} bi")
        else:
            add("Fluxo de caixa descontado", np.nan,
                "fluxo de caixa livre de 12 meses negativo ou dados de custo de capital indisponíveis")
    jan10 = base[(base.index > pd.Timestamp(data_hoje) - pd.DateOffset(years=10)) & (base.index <= pd.Timestamp(data_hoje))]
    med_pl = float(np.exp(jan10["log_pl"].median())) if jan10["log_pl"].notna().any() else np.nan
    med_pvp = float(np.exp(jan10["log_pvp"].median())) if jan10["log_pvp"].notna().any() else np.nan
    add("P/L mediano de 10 anos", med_pl * lpa if np.isfinite(lpa) and lpa > 0 else np.nan,
        f"P/L mediano {num(med_pl, 1)} × lucro por ação {moeda(lpa)}")
    add("P/VP mediano de 10 anos", med_pvp * vpa if np.isfinite(vpa) else np.nan,
        f"P/VP mediano {num(med_pvp, 2)} × patrimônio por ação {moeda(vpa)}")
    tem_ev = "log_ev_ebitda" in jan10 and jan10["log_ev_ebitda"].notna().any()
    if contabilidade != "banco" and tem_ev:
        med_ev = float(np.exp(jan10["log_ev_ebitda"].median()))
        ebitda = linha.get("ebitda_12m", np.nan)
        dl = linha.get("divida_liquida", np.nan)
        minor = linha.get("minoritarios", 0.0) or 0.0
        add("EV/EBITDA mediano de 10 anos",
            (med_ev * ebitda - dl - minor) / acoes if np.isfinite(ebitda) and ebitda > 0 and np.isfinite(dl) else np.nan,
            f"EV/EBITDA mediano {num(med_ev, 1)} × EBITDA de 12 meses, menos a dívida líquida")
    pares_res = _multiplos_pares(bases_pares, data_hoje, linha, acoes, lpa, vpa, contabilidade)
    for nome, valor, prem in pares_res:
        add(nome, valor, prem)
    gr = float(v["graham_fator"]) * lpa * vpa if np.isfinite(lpa) and np.isfinite(vpa) else np.nan
    add("Graham", np.sqrt(gr) if np.isfinite(gr) and gr > 0 else np.nan,
        f"√({num(float(v['graham_fator']), 1)} × LPA {moeda(lpa)} × VPA {moeda(vpa)})")
    add("Bazin", dpa / float(v["bazin_taxa"]) if np.isfinite(dpa) else np.nan,
        f"proventos de 12 meses {moeda(dpa)} ÷ {pct(float(v['bazin_taxa']), 0)}")
    return {"metodos": metodos, "preco": preco, "lpa": lpa, "vpa": vpa, "dpa": dpa, "roe_sustentavel": roe_s,
            "ke": ke, "g": g, "pl_atual": preco / lpa if np.isfinite(lpa) and lpa > 0 else np.nan,
            "pvp_atual": preco / vpa if np.isfinite(vpa) and vpa > 0 else np.nan,
            "pl_mediano": med_pl, "pvp_mediano": med_pvp}


def _multiplos_pares(bases_pares, data_hoje, linha, acoes, lpa, vpa, contabilidade) -> list:
    """Mediana dos múltiplos dos pares na mesma data, aplicada aos números do ativo."""
    if not bases_pares:
        return []
    d = pd.Timestamp(data_hoje)
    mult = {"log_pl": {}, "log_pvp": {}, "log_ev_ebitda": {}}
    for t, b in sorted(bases_pares.items()):
        x = b[b.index <= d]
        if x.empty:
            continue
        u = x.iloc[-1]
        for k in mult:
            if k in u and np.isfinite(u[k]):
                mult[k][t] = float(np.exp(u[k]))

    def desc(k, rotulo, casas):
        v = mult[k]
        return f"{rotulo} mediano de {len(v)} pares ({', '.join(v)}): {num(float(np.median(list(v.values()))), casas)}"

    out = []
    if mult["log_pl"] and np.isfinite(lpa) and lpa > 0:
        m = float(np.median(list(mult["log_pl"].values())))
        out.append(("P/L mediano dos pares", m * lpa, desc("log_pl", "P/L", 1)))
    if mult["log_pvp"] and np.isfinite(vpa):
        m = float(np.median(list(mult["log_pvp"].values())))
        out.append(("P/VP mediano dos pares", m * vpa, desc("log_pvp", "P/VP", 2)))
    if contabilidade != "banco" and mult["log_ev_ebitda"]:
        m = float(np.median(list(mult["log_ev_ebitda"].values())))
        ebitda, dl = linha.get("ebitda_12m", np.nan), linha.get("divida_liquida", np.nan)
        minor = linha.get("minoritarios", 0.0) or 0.0
        v = (m * ebitda - dl - minor) / acoes if np.isfinite(ebitda) and ebitda > 0 and np.isfinite(dl) else np.nan
        out.append(("EV/EBITDA mediano dos pares", v, desc("log_ev_ebitda", "EV/EBITDA", 1)))
    return out
