"""Onde ficam código, configuração, saídas e cache.

- Código, configuração e saídas (telas, relatórios, histórico): na pasta do repositório
  (no PC: Documentos\\margem_de_seguranca).
- Ambiente Python e cache de dados: fora do OneDrive, em %LOCALAPPDATA%\\margem_de_seguranca
  (no Linux, ~/.cache/margem_de_seguranca).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = Path(os.environ.get("MS_CONFIG", RAIZ / "config"))
PACOTE = Path(__file__).resolve().parent


def saida() -> Path:
    p = Path(os.environ.get("MS_SAIDA", RAIZ / "saida"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def base_local() -> Path:
    if os.environ.get("MS_BASE_LOCAL"):
        return Path(os.environ["MS_BASE_LOCAL"])
    if sys.platform == "win32":
        raiz = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        raiz = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return raiz / "margem_de_seguranca"


def _pasta(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def bruto(*partes: str) -> Path:
    """Pasta do cache de arquivos baixados, como vieram da fonte."""
    return _pasta(base_local() / "bruto" / Path(*partes))


def tratado(*partes: str) -> Path:
    """Pasta do cache de dados já lidos e convertidos (parquet)."""
    return _pasta(base_local() / "tratado" / Path(*partes))


def arquivo_env() -> Path:
    return base_local() / ".env"


# Saídas, junto dos relatórios (fora do cache).
def pasta_ativos() -> Path:
    return _pasta(saida() / "ativos")


def pasta_lib() -> Path:
    return _pasta(saida() / "lib")


def pasta_ultima_rodada(ticker: str) -> Path:
    return _pasta(saida() / "ultima_rodada" / ticker.upper())


def pasta_relatorios(ticker: str) -> Path:
    return _pasta(saida() / "relatorios" / ticker.upper())


def arquivo_historico() -> Path:
    return saida() / "historico_analises.jsonl"


def arquivo_fora_do_ranking() -> Path:
    return saida() / "fora_do_ranking_dos_ativos.jsonl"


def arquivo_lote() -> Path:
    return saida() / "lote_atual.json"


def pagina_ranking_dos_ativos() -> Path:
    return saida() / "ranking_dos_ativos.html"


def ler_env() -> dict[str, str]:
    """Lê o arquivo .env do cache local (chaves de API). Nunca grave chaves no repositório."""
    valores: dict[str, str] = {}
    p = arquivo_env()
    if p.exists():
        for linha in p.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if linha and not linha.startswith("#") and "=" in linha:
                k, v = linha.split("=", 1)
                valores[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("FRED_API_KEY",):
        if os.environ.get(k):
            valores[k] = os.environ[k]
    return valores
