"""Acesso à internet com novas tentativas, intervalo mínimo por site e cache em disco."""
from __future__ import annotations

import email.utils
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Sites que recusam clientes sem cara de navegador. Os demais ficam com o nome padrão da
# biblioteca requests: o FRED, por exemplo, bloqueia tanto quem se apresenta como navegador sem
# ser um quanto nomes de cliente desconhecidos.
UA_NAVEGADOR = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
_HOSTS_NAVEGADOR = {"query1.finance.yahoo.com", "query2.finance.yahoo.com", "bvmf.bmfbovespa.com.br"}

# Intervalo mínimo entre pedidos ao mesmo site, para respeitar os limites de acesso.
_INTERVALO = {
    "query1.finance.yahoo.com": 0.6,
    "query2.finance.yahoo.com": 0.6,
    "api.bcb.gov.br": 0.25,
    "olinda.bcb.gov.br": 0.4,
    "fred.stlouisfed.org": 0.6,
    "api.stlouisfed.org": 0.6,
    "dados.cvm.gov.br": 0.3,
    "bvmf.bmfbovespa.com.br": 1.0,
    "www.ipeadata.gov.br": 0.5,
}
_ultimo: dict[str, float] = {}
_trava = threading.Lock()
_sessao: requests.Session | None = None


def sessao() -> requests.Session:
    global _sessao
    if _sessao is None:
        s = requests.Session()
        tentativas = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            respect_retry_after_header=True,
        )
        s.mount("https://", HTTPAdapter(max_retries=tentativas))
        s.mount("http://", HTTPAdapter(max_retries=tentativas))
        s.headers.update({"Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8"})
        _sessao = s
    return _sessao


def _esperar(url: str) -> None:
    host = urlparse(url).netloc
    intervalo = _INTERVALO.get(host, 0.2)
    with _trava:
        agora = time.monotonic()
        espera = _ultimo.get(host, 0.0) + intervalo - agora
        if espera > 0:
            time.sleep(espera)
        _ultimo[host] = time.monotonic()


def _cabecalhos(url: str, extra: dict | None = None) -> dict:
    h = {"User-Agent": UA_NAVEGADOR} if urlparse(url).netloc in _HOSTS_NAVEGADOR else {}
    if extra:
        h.update(extra)
    return h


def get(url: str, *, params=None, timeout: float = 60, **kw) -> requests.Response:
    _esperar(url)
    r = sessao().get(url, params=params, timeout=timeout, headers=_cabecalhos(url, kw.pop("headers", None)), **kw)
    r.raise_for_status()
    return r


def baixar_arquivo(
    url: str,
    destino: Path,
    *,
    max_idade_horas: float | None = None,
    timeout: float = 900,
    rotulo: str | None = None,
) -> Path:
    """Baixa `url` para `destino`, se ainda não estiver no cache ou se estiver velho.

    max_idade_horas=None: nunca baixa de novo (arquivos de anos fechados).
    Com idade definida, pergunta ao servidor se mudou (If-Modified-Since).
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    cabecalhos = {}
    if destino.exists() and destino.stat().st_size > 0:
        if max_idade_horas is None:
            return destino
        idade_h = (time.time() - destino.stat().st_mtime) / 3600
        if idade_h < max_idade_horas:
            return destino
        cabecalhos["If-Modified-Since"] = email.utils.formatdate(destino.stat().st_mtime, usegmt=True)
    parcial = destino.with_name(destino.name + ".part")
    _esperar(url)
    if rotulo:
        print(f"  baixando {rotulo}...", flush=True)
    with sessao().get(url, stream=True, timeout=timeout, headers=_cabecalhos(url, cabecalhos)) as r:
        if r.status_code == 304:
            destino.touch()
            return destino
        r.raise_for_status()
        with open(parcial, "wb") as f:
            for bloco in r.iter_content(chunk_size=1 << 20):
                f.write(bloco)
    parcial.replace(destino)
    return destino
