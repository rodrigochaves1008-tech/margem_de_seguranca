# Margem de Segurança

Estima o preço justo de ações da B3 com um modelo (rede neural e referências, validados em
walk-forward), calcula a margem de segurança contra o preço atual, mostra os preditores que mais
pesam e mantém o ranking dos ativos analisados.

**Estimativa de modelo, não é recomendação de investimento.**

A especificação completa e o plano aprovado estão em [CLAUDE.md](CLAUDE.md).

## Instalação no Windows

1. Instale o Python 3.13 (python.org ou `winget install Python.Python.3.13`).
2. Clone o repositório em `Documentos\margem_de_seguranca`.
3. Na pasta do projeto, no PowerShell:

```
powershell -ExecutionPolicy Bypass -File .\instalar.ps1
```

O instalador cria o ambiente Python e o cache em `%LOCALAPPDATA%\margem_de_seguranca` (fora do
OneDrive) e instala a skill do Claude Code no seu usuário, para os pedidos funcionarem em
qualquer sessão. Opcional: ponha a chave gratuita do FRED em `%LOCALAPPDATA%\margem_de_seguranca\.env`.

## Uso

No Claude Code: "rodar BBAS3 no Margem de Segurança", "ranking dos ativos no Margem de Segurança",
"abrir BBAS3 no Margem de Segurança", "refazer desatualizados", "tirar BBAS3 do ranking dos ativos".

Direto no terminal (com o Python do projeto):

```
python -m margem rodar BBAS3
python -m margem rodar BBAS3 WEGE3 CMIG4
python -m margem abrir BBAS3
python -m margem ranking-dos-ativos
python -m margem refazer desatualizados
python -m margem tirar BBAS3
python -m margem teste-vazamento BBAS3
```

## Setores e pares

O setor de cada ativo vem do cadastro da CVM, ligado aos setores do modelo por `config/setores.yaml`
(`mapa_cvm`). Hoje estão ativos: bancos, indústria e energia elétrica. Cada setor tem a sua lista de
preditores e de pares; os pares entram num painel (candidato a mais no walk-forward) e na comparação
de múltiplos. Ativo de setor fora do mapa: a rodada para e pede a aprovação dos preditores.

## Onde ficam as coisas

- `saida/ativos/<TICKER>.html`: tela do ativo, com os gráficos de 10 anos.
- `saida/ranking_dos_ativos.html`: página do ranking dos ativos.
- `saida/relatorios/<TICKER>/<data_hora>/relatorio.html`: relatório, com a cópia dos dados usados.
- `saida/historico_analises.jsonl`: histórico das análises (só acrescenta linhas).
- `saida/ultima_rodada/<TICKER>/`: o que cada tela usa da última rodada.
- `config/`: faixas da margem, critério de estimativa confiável, mapa setor → preditores, séries.
