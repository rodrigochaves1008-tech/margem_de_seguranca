"""Fontes de dados gratuitas, com cache em disco.

Toda tabela devolvida por uma fonte traz a coluna `divulgacao`: a data em que o dado passou
a ser público. A montagem da base mensal só usa linhas com divulgacao <= data de corte.
"""
