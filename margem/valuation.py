"""Checagem com valuation tradicional: modelo de dividendos e P/VP justificado (bancos),
múltiplos contra o histórico, Graham e Bazin. Pares entram na Fase 2."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .formatos import moeda, num, pct


def calcular(base: pd.DataFrame, data_hoje, fundamentos_hist: pd.DataFrame | None, cfg: dict) -> dict:
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

    if np.isfinite(ke) and np.isfinite(g) and ke > g:
        add("P/VP justificado (Gordon)", vpa * (roe_s - g) / (ke - g) if np.isfinite(roe_s) else np.nan,
            f"ROE sustentável {pct(roe_s, 1)} (mediana de 5 anos), custo de capital {pct(ke, 1)}, crescimento {pct(g, 1)}")
        add("Modelo de dividendos", dpa * (1 + g) / (ke - g) if np.isfinite(dpa) else np.nan,
            f"proventos de 12 meses {moeda(dpa)} crescendo {pct(g, 1)} ao ano, custo de capital {pct(ke, 1)}")
    else:
        add("P/VP justificado (Gordon)", np.nan, "custo de capital ou inflação esperada indisponível")
        add("Modelo de dividendos", np.nan, "custo de capital ou inflação esperada indisponível")
    jan10 = base[(base.index > pd.Timestamp(data_hoje) - pd.DateOffset(years=10)) & (base.index <= pd.Timestamp(data_hoje))]
    med_pl = float(np.exp(jan10["log_pl"].median())) if jan10["log_pl"].notna().any() else np.nan
    med_pvp = float(np.exp(jan10["log_pvp"].median())) if jan10["log_pvp"].notna().any() else np.nan
    add("P/L mediano de 10 anos", med_pl * lpa if np.isfinite(lpa) and lpa > 0 else np.nan,
        f"P/L mediano {num(med_pl, 1)} × lucro por ação {moeda(lpa)}")
    add("P/VP mediano de 10 anos", med_pvp * vpa if np.isfinite(vpa) else np.nan,
        f"P/VP mediano {num(med_pvp, 2)} × patrimônio por ação {moeda(vpa)}")
    gr = float(v["graham_fator"]) * lpa * vpa if np.isfinite(lpa) and np.isfinite(vpa) else np.nan
    add("Graham", np.sqrt(gr) if np.isfinite(gr) and gr > 0 else np.nan,
        f"√({num(float(v['graham_fator']), 1)} × LPA {moeda(lpa)} × VPA {moeda(vpa)})")
    add("Bazin", dpa / float(v["bazin_taxa"]) if np.isfinite(dpa) else np.nan,
        f"proventos de 12 meses {moeda(dpa)} ÷ {pct(float(v['bazin_taxa']), 0)}")
    return {"metodos": metodos, "preco": preco, "lpa": lpa, "vpa": vpa, "dpa": dpa, "roe_sustentavel": roe_s,
            "ke": ke, "g": g, "pl_atual": preco / lpa if np.isfinite(lpa) and lpa > 0 else np.nan,
            "pvp_atual": preco / vpa if np.isfinite(vpa) and vpa > 0 else np.nan,
            "pl_mediano": med_pl, "pvp_mediano": med_pvp}
