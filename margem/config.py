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


def setor_do_cvm(setor_cvm: str) -> str | None:
    """Setor deste mapa para o setor de atividade da CVM (primeira regra que casar)."""
    for trecho, setor in setores().get("mapa_cvm", []) or []:
        if str(trecho).lower() in (setor_cvm or "").lower():
            return setor
    return None


def info_ativo(ticker: str) -> dict:
    """Empresa, CNPJ e setor do ativo: cadastro da CVM, com os ajustes de config/setores.yaml.
    KeyError com mensagem clara se o ticker não existir ou se o setor estiver fora do mapa."""
    from .fontes import cadastro

    t = ticker.upper()
    ajuste = dict(setores().get("ativos", {}).get(t, {}) or {})
    try:
        res = cadastro.resolver(t)
    except Exception:
        res = None
    if res is None and not ajuste.get("cnpj"):
        raise KeyError(f"{t}: ticker não encontrado no cadastro da CVM (FCA). Confira o código.")
    info = {**(res or {}), **ajuste}
    if not info.get("setor"):
        s = setor_do_cvm(info.get("setor_cvm", ""))
        if s is None:
            raise KeyError(
                f"{t}: o setor da CVM '{info.get('setor_cvm')}' está fora do mapa (config/setores.yaml, mapa_cvm). "
                "Os preditores desse setor precisam ser propostos e aprovados antes.")
        info["setor"] = s
    info["ticker"] = t
    info["estatal"] = "ESTATAL" in str(info.get("controle") or "").upper()
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
