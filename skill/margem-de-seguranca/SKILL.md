---
name: margem-de-seguranca
description: Margem de Segurança — preço justo de ações da B3 por modelo, margem de segurança e ranking dos ativos. Use quando o usuário pedir para rodar, abrir, atualizar, refazer ou tirar ativos no "Margem de Segurança", ou pedir o "ranking dos ativos" (ex.: "rodar BBAS3 no Margem de Segurança", "roda WEGE3 e CMIG4 no margem", "abrir BBAS3 no Margem de Segurança", "ranking dos ativos no Margem de Segurança", "refazer desatualizados", "refazer todos", "tirar BBAS3 do ranking dos ativos").
---

# Margem de Segurança

Ferramenta local (Python) que estima o preço justo de ações da B3, calcula a margem de segurança
e mantém o ranking dos ativos analisados. Estimativa de modelo, não é recomendação de investimento.

## Onde está

- Código e saídas: `{{PASTA_PROJETO}}`
- Python do projeto: `{{PYTHON}}`
- Comandos: rode sempre a partir da pasta do projeto, com o Python do projeto:

```
cd "{{PASTA_PROJETO}}"
& "{{PYTHON}}" -m margem <comando>
```

## Pedidos e o que fazer

Entenda variações dos textos abaixo (com ou sem "no Margem de Segurança", "roda", "rode",
"atualiza", vários tickers separados por vírgula ou "e", tickers em minúsculas).

1. **"rodar <TICKER> no Margem de Segurança"** (um ticker): faça antes a pesquisa das mudanças
   estruturais (seção abaixo) e depois rode `rodar <TICKER>`. Ao terminar,
   diga a posição dele no ranking dos ativos (ex.: "3º de 12 ativos com estimativa") ou, se não
   houver estimativa confiável, o motivo. A tela do ativo abre sozinha; dê também o link do relatório.
2. **Vários tickers**: antes, diga o tempo estimado (o comando imprime) e comece sem esperar OK.
   Faça a pesquisa das mudanças estruturais de cada ativo e rode `rodar T1 T2 T3`. O lote roda um por um; se um ativo falhar ou precisar de OK (setor fora
   do mapa), o lote segue com os outros. No fim, mostre o resumo impresso (data dos preços e os 10
   primeiros, um por linha: posição, ticker, margem em %, faixa da margem e marcas), o total, o link
   da página do ranking dos ativos e os ativos que falharam. Se o lote parou no meio, use `retomar`.
3. **"abrir <TICKER> no Margem de Segurança"**: rode `abrir <TICKER>` (atualização rápida, sem rodar
   o modelo). Se o comando disser que o ativo nunca foi rodado, diga isso e pergunte se o usuário
   quer rodá-lo.
4. **"ranking dos ativos no Margem de Segurança"**: rode `ranking-dos-ativos` e mostre o resumo
   impresso com o link da página.
5. **"refazer desatualizados no Margem de Segurança"** / **"refazer todos"**: primeiro rode
   `refazer desatualizados --so-listar` (ou `refazer todos --so-listar`) e mostre a lista; depois
   rode `refazer desatualizados` (ou `refazer todos`) como lote, seguindo o item 2.
6. **"tirar <TICKER> do ranking dos ativos"**: rode `tirar <TICKER>` e confirme. Nada é apagado;
   rodar o ativo de novo o traz de volta.

## Mudanças estruturais: pesquisa na web a cada rodada

Antes de rodar cada ativo (também em lotes), pesquise na web as mudanças dos últimos 24 meses e as já
aprovadas que afetam a empresa e o setor: reforma tributária do consumo (CBS e IBS), IR sobre
dividendos e JCP, regras contábeis (como as de provisão dos bancos), regulação do setor, privatização
ou troca de controle, interferência em estatais. Use fontes com data (leis no Planalto, Banco Central,
CVM, agências reguladoras, imprensa econômica). Grave o resultado em
`saida/ultima_rodada/<TICKER>/mudancas_estruturais.json` e só então rode o ativo: o relatório lê esse
arquivo (seção 6). Num lote, a parte do setor pode ser pesquisada uma vez e repetida nos ativos dele.

```
{"ticker": "BBAS3", "data_pesquisa": "AAAA-MM-DD", "janela": "últimos 24 meses e mudanças já aprovadas",
 "itens": [{"data": "AAAA-MM-DD", "titulo": "...", "rotulo": "texto curto para o gráfico",
            "tipo": "tributario | contabil | regulatorio | controle | estatal | outro",
            "situacao": "em vigor | aprovada | proposta ...", "vigencia": "AAAA-MM-DD",
            "efeito_esperado": "positivo | negativo | incerto", "explicacao": "...",
            "canais": ["log_lucro_12m", "payout"],
            "fontes": [{"url": "...", "titulo": "...", "data": "AAAA-MM-DD ou null"}]}]}
```

- `canais`: preditores por onde a mudança age (nomes de `config/series.yaml` ou indicadores da
  empresa, como `log_lucro_12m`, `payout`, `custo_credito`, `margem_ebitda`). A ferramenta usa a lista
  para dizer se o modelo capta a mudança.
- Efeito esperado: só a direção (positivo, negativo ou incerto) e a explicação. Sem números e sem
  projeções para o futuro (decisão do usuário).
- Eventos estáveis, que valem para todas as rodadas, ficam em `config/eventos.yaml`.
- Sem internet, rode mesmo assim: o relatório diz que a pesquisa não foi feita nesta rodada.

## Regras

- Diga sempre "ranking dos ativos", nunca só "ranking" (para não confundir com o ranking dos
  preditores). Use rótulos neutros ("maior margem"), nunca "melhores" ou "oportunidades".
- Setor fora do mapa (o comando avisa "Precisa do seu OK"): proponha os preditores do setor (veja
  `config/setores.yaml` e `config/series.yaml`), mostre ao usuário e só use depois do OK dele.
- Se o modelo não for confiável para o ativo, diga isso; não force um número.
- Não faça projeções para o futuro (decisão do usuário): nem de preço justo nem de cenários.
- Termine as respostas com: "Estimativa de modelo, não é recomendação de investimento."

## Uso dos créditos (regra do usuário)
Ao chegar em 50% da janela de 5 horas, pause e volte sozinho quando ela renovar; ao chegar em 50% do
limite semanal, pause e só volte quando o semanal renovar. Se a sessão tiver como ler o uso (na nuvem:
`list_events` da própria sessão com `kinds: ["rate_limit_event"]`) e agendar a volta (`send_later`),
siga o procedimento da seção "Uso dos créditos" do CLAUDE.md do projeto. Se não tiver, avise o usuário,
antes de lotes longos, que a pausa automática não está disponível nesta sessão.
