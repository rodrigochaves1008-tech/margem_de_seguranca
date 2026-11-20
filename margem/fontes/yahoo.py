"""Yahoo Finance (API de gráficos): fechamentos, desdobramentos/grupamentos e proventos.

Atenção: o `close` do Yahoo já vem ajustado por desdobramentos e grupamentos (não por proventos);
o `adjclose` é ajustado por proventos e NUNCA é usado. Os valores de proventos do Yahoo vêm na
base de ações de hoje (divididos pelos desdobramentos posteriores).
"""
from __future__ import annotations

import time

import pandas as pd

from .. import caminhos, rede

URL = "https://query2.finance.yahoo.com/v8/finance/chart/{codigo}"


def _baixar(codigo: str, max_idade_horas: float) -> dict:
    nome = codigo.replace("^", "_").replace("=", "_")
    destino = caminhos.bruto("yahoo") / f"{nome}.json"
    if destino.exists() and (time.time() - destino.stat().st_mtime) / 3600 < max_idade_horas:
        import json

        return json.loads(destino.read_text(encoding="utf-8"))
    r = rede.get(
        URL.format(codigo=codigo),
        params={"period1": 946684800, "period2": int(time.time()) + 86400, "interval": "1d",
                "events": "div,splits", "includeAdjustedClose": "false"},
        timeout=60,
    )
    destino.write_text(r.text, encoding="utf-8")
    return r.json()


def serie(codigo: str, max_idade_horas: float = 4) -> dict[str, pd.DataFrame]:
    """Devolve {'precos': data, fechamento(ajustado só por desdobramentos), volume;
                'desdobramentos': data, fator (n:1 → n); 'proventos': data, valor (base atual)}.
    Datas no fuso de São Paulo, sem hora."""
    d = _baixar(codigo, max_idade_horas)["chart"]["result"][0]
    fuso = d["meta"].get("exchangeTimezoneName", "America/Sao_Paulo")
    ts = pd.to_datetime(d["timestamp"], unit="s", utc=True).tz_convert(fuso).tz_localize(None).normalize()
    q = d["indicators"]["quote"][0]
    precos = pd.DataFrame({"data": ts, "fechamento": q.get("close"), "volume": q.get("volume")})
    precos = precos.dropna(subset=["fechamento"]).drop_duplicates("data", keep="last")
    ev = d.get("events", {}) or {}
    desd = [
        {"data": pd.Timestamp(v["date"], unit="s", tz="UTC").tz_convert(fuso).tz_localize(None).normalize(),
         "fator": float(v["numerator"]) / float(v["denominator"])}
        for v in (ev.get("splits") or {}).values()
    ]
    prov = [
        {"data": pd.Timestamp(v["date"], unit="s", tz="UTC").tz_convert(fuso).tz_localize(None).normalize(),
         "valor": float(v["amount"])}
        for v in (ev.get("dividends") or {}).values()
    ]
    desd_df = pd.DataFrame(desd, columns=["data", "fator"]).sort_values("data").reset_index(drop=True)
    prov_df = pd.DataFrame(prov, columns=["data", "valor"]).sort_values("data").reset_index(drop=True)
    # Proventos do mesmo dia (dividendo + JCP) somados.
    prov_df = prov_df.groupby("data", as_index=False)["valor"].sum()
    return {"precos": precos.reset_index(drop=True), "desdobramentos": desd_df, "proventos": prov_df}
