"""Leitura dos arquivos de configuração (config/*.yaml)."""
from __future__ import annotations

import copy
import functools
import hashlib
import json

import yaml

from .caminhos import CONFIG


@functools.cache
def _carregar(nome: str) -> dict:
    with open(CONFIG / f"{nome}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def carregar(nome: str) -> dict:
    return copy.deepcopy(_carregar(nome))


def cfg() -> dict:
    return carregar("configuracao")


def setores() -> dict:
    return carregar("setores")


def catalogo_series() -> dict:
    return carregar("series")


def eventos() -> dict:
    return carregar("eventos")


def recarregar() -> None:
    _carregar.cache_clear()


def info_ativo(ticker: str) -> dict:
    """Dados do ativo no setores.yaml (nome, setor, CNPJ). Erro claro se não estiver no mapa."""
    t = ticker.upper()
    ativos = setores().get("ativos", {})
    if t not in ativos:
        raise KeyError(
            f"{t} não está em config/setores.yaml (seção 'ativos'). "
            "Inclua nome, setor e CNPJ; se o setor não estiver no mapa, os preditores "
            "precisam ser propostos e aprovados antes."
        )
    info = dict(ativos[t])
    info["ticker"] = t
    return info


def info_setor(setor: str) -> dict:
    s = setores().get("setores", {})
    if setor not in s:
        raise KeyError(f"Setor '{setor}' fora do mapa em config/setores.yaml")
    return s[setor]


def hash_config_setor(setor: str) -> str:
    """Hash curto da parte da configuração que altera as estimativas desse setor."""
    c = cfg()
    relevante = {
        "setor": info_setor(setor),
        "modelo": c.get("modelo", {}),
        "series": {
            k: v
            for k, v in catalogo_series().items()
            if k in _preditores_do_setor(setor)
        },
    }
    bruto = json.dumps(relevante, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(bruto.encode("utf-8")).hexdigest()[:8]


def _preditores_do_setor(setor: str) -> set[str]:
    p = info_setor(setor).get("preditores", {})
    return {x for lista in p.values() for x in (lista or [])}


def versao_modelo(setor: str) -> str:
    from . import VERSAO_MODELO_BASE

    return f"{VERSAO_MODELO_BASE}+{setor}-{hash_config_setor(setor)}"
