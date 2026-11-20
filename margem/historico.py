"""Histórico das análises (JSON Lines, só acrescenta) e registro de saídas do ranking dos ativos."""
from __future__ import annotations

import json
from datetime import datetime

from . import caminhos


def registrar(linha: dict) -> None:
    """Acrescenta uma linha ao histórico, sem apagar nem reescrever as anteriores."""
    p = caminhos.arquivo_historico()
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(linha, ensure_ascii=False, default=str) + "\n")


def ler() -> list[dict]:
    p = caminhos.arquivo_historico()
    if not p.exists():
        return []
    out = []
    for linha in p.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha:
            try:
                out.append(json.loads(linha))
            except json.JSONDecodeError:
                continue
    return out


def mais_recentes() -> dict[str, dict]:
    """A análise mais recente de cada ativo (é a que entra no ranking dos ativos)."""
    ultimo: dict[str, dict] = {}
    for h in ler():
        t = h.get("ticker")
        if t and (t not in ultimo or str(h.get("data_analise")) >= str(ultimo[t].get("data_analise"))):
            ultimo[t] = h
    return ultimo


def do_ativo(ticker: str) -> list[dict]:
    return sorted([h for h in ler() if h.get("ticker") == ticker.upper()], key=lambda h: str(h.get("data_analise")))


def registrar_fora(ticker: str, acao: str, motivo: str | None = None) -> None:
    """acao: 'saida' (tirar do ranking dos ativos) ou 'volta' (rodou de novo)."""
    p = caminhos.arquivo_fora_do_ranking()
    reg = {"ticker": ticker.upper(), "acao": acao, "data": datetime.now().isoformat(timespec="seconds")}
    if motivo:
        reg["motivo"] = motivo
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(reg, ensure_ascii=False) + "\n")


def fora_do_ranking() -> set[str]:
    p = caminhos.arquivo_fora_do_ranking()
    estado: dict[str, str] = {}
    if p.exists():
        for linha in p.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                r = json.loads(linha)
                estado[r["ticker"]] = r["acao"]
    return {t for t, a in estado.items() if a == "saida"}
