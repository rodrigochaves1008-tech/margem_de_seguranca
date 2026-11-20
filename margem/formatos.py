"""Números e datas no formato brasileiro."""
from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
MENOS = "−"  # sinal de menos tipográfico


def _vazio(v) -> bool:
    return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or v is pd.NA


def num(v, casas: int = 2, sinal: bool = False) -> str:
    if _vazio(v):
        return "—"
    s = f"{abs(v):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if v < 0 and float(f"{abs(v):.{casas}f}") != 0:
        return MENOS + s
    if sinal and v > 0:
        return "+" + s
    return s


def moeda(v, casas: int = 2) -> str:
    if _vazio(v):
        return "—"
    s = num(abs(v), casas)
    return (MENOS if v < 0 else "") + "R$ " + s


def pct(v, casas: int = 1, sinal: bool = False) -> str:
    if _vazio(v):
        return "—"
    return num(100 * v, casas, sinal=sinal) + "%"


def data(d) -> str:
    if _vazio(d):
        return "—"
    d = pd.Timestamp(d)
    return d.strftime("%d/%m/%Y")


def mes(d) -> str:
    d = pd.Timestamp(d)
    return f"{MESES[d.month - 1]}/{d.year}"


def data_hora(d: datetime) -> str:
    return pd.Timestamp(d).strftime("%d/%m/%Y %H:%M")


def ordinal(n: int) -> str:
    return f"{n}º"


def iso(d) -> str:
    if isinstance(d, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(d).strftime("%Y-%m-%d")
    return str(d)
