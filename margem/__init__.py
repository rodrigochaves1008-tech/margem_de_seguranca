"""Margem de Segurança: preço justo por modelo, margem de segurança e ranking dos ativos da B3.

Estimativa de modelo, não é recomendação de investimento.
"""
import os as _os

# Reprodutibilidade: uma thread nas bibliotecas numéricas. Sem isso, somas em paralelo mudam
# a ordem das operações e o teste de vazamento deixa de fechar ao centavo.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_var, "1")

VERSAO_CODIGO = "0.2.0"
# Mude só quando a mudança alterar as estimativas (ver CLAUDE.md, "Ranking dos ativos analisados").
VERSAO_MODELO_BASE = "1.1.0"

AVISO = "Estimativa de modelo, não é recomendação de investimento."
