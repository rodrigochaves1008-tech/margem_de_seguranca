# Margem de Segurança: especificação e plano aprovado

Este arquivo fica em `Documentos\margem_de_seguranca\CLAUDE.md`. O Claude Code o lê sozinho
quando uma sessão é aberta nesta pasta.

## Estado do projeto
Atualize esta seção ao fim de cada fase.

- 20/11/2026: plano apresentado e aprovado (seção "Plano aprovado", no fim). Ainda não há código.
- O item 1 de "Como entregar" (apresentar o plano e esperar o OK) já foi cumprido. Não
  apresente o plano de novo: comece a Fase 1 (BBAS3).
- Código no repositório privado `rodrigochaves1008-tech/margem_de_seguranca` (GitHub). No PC,
  `Documentos\margem_de_seguranca` é um clone dele.
- O código pode ser escrito numa sessão na nuvem ou no PC, sempre pelo repositório. Os testes
  com dados reais precisam de acesso às fontes: no PC há; na nuvem, só com os domínios das
  fontes liberados nas configurações do ambiente.

## Decisões do usuário
1. Execução: no PC do usuário (Windows); código versionado no repositório do GitHub.
2. Critério de estimativa confiável: regra estrita de não perder para o passeio aleatório
   (seção 5 do plano). O limite final é revisto depois do resultado de BBAS3 na Fase 1:
   mostrar os números ao usuário antes de fixar.
3. Preditores extras: no BB, soja, milho e inadimplência do crédito rural; na WEG, euro e
   produção industrial dos EUA.
4. Substitutos: EMBI+ Brasil (Ipeadata) no lugar do CDS de 5 anos; indicador antecedente da
   OCDE para a China se o PMI oficial (NBS) não tiver acesso estável. Custos da margem como
   sinal: 0,25% ida e volta.

Os itens 2 a 4 são as propostas do plano, adotadas sem objeção do usuário. Ele pode mudá-las.

## Pendências com o usuário
- Chave gratuita da API do FRED, para as versões da época (ALFRED). O usuário cria a chave em
  fred.stlouisfed.org e grava em `%LOCALAPPDATA%\margem_de_seguranca\.env` como
  `FRED_API_KEY=...`. Nunca pedir a chave no chat.
- Python 3.11 ou mais novo no Windows: verificar com `py -0p`. Se faltar, pedir ao usuário
  para instalar antes da Fase 1.

## Especificação do usuário (texto original, com os títulos um nível abaixo)

### Papel
Atue como analista de valuation sênior com experiência em modelos quantitativos e machine
learning aplicados a ações brasileiras. Seja rigoroso com dados e validação, e honesto sobre
a incerteza.

### Objetivo
Eu informo o ticker de uma empresa listada na B3. A ferramenta estima o preço justo da ação
com uma rede neural, calcula a margem de segurança em relação ao preço atual e mostra o
ranking dos preditores, os fatores que mais pesam nesse preço. Também guarda os ativos já
analisados e os ordena por margem de segurança. Ao clicar num deles, mostra os gráficos de
10 anos do preço e do preço justo estimado pelo modelo.

### Como vou usar
- Vou escrever na sessão: "rodar <TICKER> no Margem de Segurança", com um ticker ou vários.
- Com um ticker, você roda o processo completo, abre a tela do ativo e gera o relatório.
- Com vários, você roda um por um, gera a tela e o relatório de cada um e, no fim, abre só a
  página do ranking dos ativos. Diga antes o tempo estimado e comece sem esperar meu OK.
  Se um ativo falhar ou precisar do meu OK (setor fora do mapa), siga com os outros e me
  diga no fim. Se o lote parar no meio, retome de onde parou.
- Na página do ranking dos ativos, vou clicar no ativo que me interessar para abrir a tela
  dele, com os gráficos de 10 anos.
- "abrir <TICKER> no Margem de Segurança": faz a atualização rápida e abre a tela do ativo,
  sem rodar o modelo. Se o ativo nunca foi rodado, diga isso e pergunte se quero rodá-lo.
- "ranking dos ativos no Margem de Segurança": faz a atualização rápida e abre a página do
  ranking dos ativos.
- "refazer desatualizados no Margem de Segurança": roda de novo, como lote, os ativos com
  análise desatualizada; "refazer todos" roda todos. Antes, mostre a lista.
- "tirar <TICKER> do ranking dos ativos": registra a saída num arquivo à parte e refaz a
  página do ranking dos ativos, sem apagar o histórico, os relatórios nem a tela, que passa
  a dizer "fora do ranking dos ativos". A atualização rápida deixa de incluí-lo. Se eu
  rodar o ativo de novo, ele volta.
- Crie uma skill do Claude Code no nível do usuário, para esses pedidos funcionarem em
  qualquer sessão nova, de qualquer pasta. Entenda também variações desses textos.
- Não quero planilha como interface.

### Definições
- Preço justo (V): preço que o modelo estima para hoje, dados os fundamentos da empresa, o
  cenário macro e os fatores do setor. Mostrar o valor central e a faixa de 80%.
- Faixa de 80% (P10 a P90): V multiplicado pelos percentis 10 e 90 de P ÷ V nos meses fora
  da amostra já conhecidos na data. Com menos de 24 meses, sem faixa. Nunca usar a
  dispersão entre sementes, estreita demais.
- Margem de segurança = (V − P) ÷ V, com P = preço atual.
  Positiva: preço abaixo do justo (desconto). Negativa: preço acima do justo (prêmio).
- Potencial até o preço justo = V ÷ P − 1.
  Exemplo: V = R$ 20 e P = R$ 15 → margem de 25% e potencial de 33%.
- Faixas da margem, configuráveis: acima de 30% desconto forte; de 15% a 30% desconto
  moderado; de −15% a 15% perto do preço justo; de −15% a −30% prêmio moderado; abaixo de
  −30% prêmio forte.
- Erro típico: mediana de |V − P| ÷ V nos meses fora da amostra, com P do mesmo mês.
  Dizer sempre se a margem, em módulo, supera o erro típico. Se não superar, tratar a
  diferença como dentro da margem de erro.

### Dados (fontes gratuitas, com cache)
Preço: histórico diário desde 2010, quando houver, com os fatores de ajuste por proventos,
desdobramentos e grupamentos; ações em circulação.

Fundamentos da empresa, trimestrais, desde 2010 quando houver, o máximo disponível:
- Rentabilidade: ROE, ROA, ROIC, margens bruta, EBITDA, EBIT e líquida.
- Crescimento: receita, EBITDA e lucro em 12 meses e em 3 anos.
- Endividamento e liquidez: dívida líquida/EBITDA, dívida líquida/PL, cobertura de juros,
  liquidez corrente.
- Caixa e eficiência: fluxo de caixa livre, capex/receita, giro do ativo, ciclo de caixa.
- Por ação: LPA, VPA, fluxo de caixa livre por ação, dividendos por ação, payout.
- Estrutura: controle estatal ou privado, free float, recompras e emissões.
- Fontes: CVM dados abertos (DFP e ITR), B3, relações com investidores da empresa.

Macro e mercado:
- Brasil: Selic, CDI, DI futuro de 1 ano (juro curto) e de 5 a 10 anos (juro longo),
  NTN-B longa (juro real), inclinação da curva, IPCA, IGP-M, expectativas do Boletim Focus,
  dólar, Ibovespa, risco-país (CDS de 5 anos), IBC-Br, desemprego, massa salarial,
  crédito, fluxo estrangeiro na B3.
- Exterior: Fed funds, Treasuries de 2 e 10 anos, DXY, VIX, S&P 500, Brent, minério de
  ferro, índice de commodities, PMI da China.
- Fontes: BCB (SGS, Focus, IF.data), IBGE (SIDRA), Ipeadata, Tesouro, FRED, Yahoo Finance.

Fatores do setor: cada setor tem preditores próprios. Mantenha um mapa setor → preditores
num arquivo de configuração que eu possa editar. Exemplos:
- Bancos: inadimplência acima de 90 dias, custo do crédito (PDD), margem financeira (NIM),
  índice de Basileia, índice de eficiência, crescimento da carteira, spread bancário.
- Indústria: utilização da capacidade instalada (ociosidade), produção industrial, PMI
  industrial, preço dos insumos, câmbio.
- Energia elétrica: reservatórios, GSF, PLD, consumo de energia, índices dos reajustes,
  revisões tarifárias.
- Varejo: vendas no varejo (PMC), confiança do consumidor, massa salarial, juros ao
  consumidor.
- Saúde e farmacêutico: reajuste da CMED, câmbio dos insumos importados.
- Educação: FIES e Prouni, matrículas, evasão, ticket médio.
- Commodities (mineração, siderurgia, celulose, petróleo): preço da commodity, câmbio,
  demanda da China.
- Construção: INCC, lançamentos, vendas, financiamento imobiliário (SBPE).
Se o setor não estiver no mapa, proponha os preditores e me mostre antes de usar.

Regras de qualidade:
- Point-in-time: cada número entra na data em que foi divulgado (na CVM, para os balanços;
  no calendário de cada fonte, para as séries macro), nunca na data de referência. Nenhuma
  informação do futuro.
- Balanço reapresentado: em cada data, usar a versão já entregue. Séries revisadas depois
  (IBC-Br, produção industrial, PMC, desemprego): usar a versão da época quando a fonte
  guardar (ALFRED, para as do FRED); se não der, dizer em Limitações e na legenda do
  gráfico do preço justo.
- Não usar como preditor do preço indicadores que já contêm o preço (P/L, P/VP, EV/EBITDA,
  dividend yield). Eles podem servir de alvo alternativo ou de comparação.
- Base mensal: fundamentos valem até a divulgação seguinte; macro, o último dado já
  divulgado no fim de cada mês.
- Mostrar a cobertura: de onde veio cada série e o que faltou.

### Modelo
- Saída final em R$. O modelo pode prever o log do preço ou um múltiplo (P/L, P/VP,
  EV/EBITDA) e converter para preço, o que validar melhor fora da amostra.
- O preço do alvo é o fechamento ajustado só por desdobramentos e grupamentos, com os
  dados por ação na mesma base. Nunca o preço ajustado por proventos (padrão de algumas
  bibliotecas): o valor dele numa data passada depende de proventos pagos depois.
- Modelo principal: rede neural com regularização, early stopping e média de várias
  sementes.
- Comparar com referências: último preço (passeio aleatório), regressão linear
  regularizada e gradient boosting. Dizer qual ganhou, pelo critério do backtest; usar
  ensemble se for melhor.
- Controle de overfitting: remover variáveis redundantes, selecionar variáveis e validar
  sempre no tempo. Se o histórico do ativo for curto, treinar em painel com empresas do
  mesmo setor e avisar no relatório.
- Checagem com valuation tradicional: fluxo de caixa descontado (modelo de dividendos para
  bancos), múltiplos contra o histórico e os pares e, como referência simples, Graham e
  Bazin. Se o modelo divergir muito, explicar por quê.

### Ranking dos preditores
- Importância por SHAP, conferida por permutação, em % do total.
- Top 10 do ativo com a direção do efeito (empurra o preço para cima ou para baixo) e a
  família (empresa, macro Brasil, exterior, setor).
- Mostrar se o ranking dos preditores mudou nos últimos 3 anos.
- Explicar os 5 maiores em linguagem simples.

### Backtest
- Walk-forward: em cada mês, estimar o preço justo só com o que era conhecido até ali e
  comparar com o que aconteceu depois. Pode retreinar a cada trimestre, depois de cada
  divulgação, e estimar todo mês com o último modelo treinado. O preço do mês estimado
  nunca entra no treino que o estima. A análise de hoje é o último passo do walk-forward.
- Em cada retreino, tudo é refeito só com dados até aquela data: seleção de variáveis,
  normalização, hiperparâmetros, early stopping e a escolha entre alvos, modelos e
  ensemble (só com resultados de 6 e 12 meses já fechados na data). Nada é escolhido
  olhando o período todo.
- Fora da amostra, cobrir os últimos 10 anos; se os dados não permitirem, pelo menos 5.
- Guardar a estimativa de cada mês (V, P10 e P90). Numa nova rodada, calcular só os meses
  novos, a não ser que a versão do modelo mude.
- Critério para escolher alvo, modelo e ensemble, e para comparar com o passeio aleatório:
  o preço 6 e 12 meses depois ficou mais perto do preço justo ou do preço da data? Medir
  MAPE, RMSE e R² contra esse preço futuro. A comparação com o preço do mesmo mês serve
  só para o erro típico e a faixa de 80%: nela ganha quem copia o preço.
- Mostrar em quantos % dos meses o preço ficou dentro da faixa de 80%.
- Margem como sinal: quando a margem entrou nas faixas de desconto moderado e forte, qual
  foi o retorno em 6 e 12 meses contra Ibovespa e CDI, taxa de acerto e número de sinais,
  já com custos. Sem sinais sobrepostos: um novo sinal só depois do prazo do anterior.
- Essas estimativas formam o gráfico "Preço justo ao longo do tempo" (ver "Gráficos do
  ativo").

### Mudanças estruturais
- Testar quebras estruturais nas séries e na relação entre preditores e preço.
- Marcar eventos: reforma tributária do consumo (CBS e IBS, transição a partir de 2026),
  mudanças no IR sobre dividendos e JCP, mudanças contábeis (como as novas regras de
  provisão dos bancos), regulação do setor, privatização ou troca de controle,
  interferência em estatais.
- A cada rodada, pesquisar na web as mudanças dos últimos 24 meses e as já aprovadas, com
  fonte e data. Dizer se o efeito esperado é positivo, negativo ou incerto, e se o modelo
  capta isso.

### Projeção
Preço justo em 12 meses em três cenários: base (Boletim Focus), otimista e pessimista,
variando juros, câmbio e atividade.

### Ranking dos ativos analisados
Lista dos ativos que já passaram pela análise, ordenada por margem de segurança. No código,
na tela e nos textos, chame sempre de "ranking dos ativos", nunca só "ranking", para não
confundir com o "Ranking dos preditores".

Histórico:
- Cada rodada acrescenta, ao terminar, uma linha a um arquivo de texto (JSON Lines), sem
  apagar nem reescrever as anteriores. A linha guarda: ticker e setor; data da análise;
  preço usado; preço justo (V, P10 e P90); erro típico no backtest; principal preditor e
  direção; se houve estimativa ou o motivo de não haver; última divulgação da CVM usada;
  versão do modelo; caminho do relatório.
- O ranking dos ativos usa a análise mais recente de cada ativo.

Comparação justa:
- Mesma data de preço para todos: o último fechamento disponível, contra o último preço
  justo de cada ativo, sem rodar o modelo de novo.
- Ajustar o preço justo e a faixa de 80% pelos proventos, desdobramentos e grupamentos
  ocorridos depois da análise, com os mesmos fatores do histórico de preços.
- Ordem: da maior para a menor margem em % (maior desconto no topo, maior prêmio no fim).
  Nunca pela margem em R$, que não se compara entre ativos.
- Análise desatualizada: tem 30 dias ou mais (configurável), há na CVM ITR ou DFP mais
  novo que o último usado nela ou foi feita com versão anterior do modelo. Continua no
  ranking dos ativos, marcada com o motivo. Mude a versão do modelo só quando a mudança
  alterar as estimativas.
- Se a análise mais recente não tiver estimativa confiável, o ativo vai para uma lista à
  parte, sem posição, com data e motivo. O critério de estimativa confiável é o mesmo para
  todos os ativos e fica na configuração.

Atualização rápida: busca os últimos preços e os proventos, desdobramentos e grupamentos,
confere se saiu ITR ou DFP novo, recalcula margens e marcas e refaz a tela de cada ativo.
Não é uma rodada: não treina modelos, não baixa fundamentos nem macro, não pesquisa
mudanças estruturais e não grava no histórico. Toda rodada faz a atualização rápida dos
demais ativos antes de gerar o relatório.

Página do ranking dos ativos, em HTML, junto dos relatórios, refeita a cada rodada e a cada
atualização:
- No topo: data dos preços; quantos ativos têm posição, quantos estão desatualizados e
  quantos estão sem estimativa confiável. Se metade ou mais dos 10 primeiros for do mesmo
  setor ou tiver o mesmo principal preditor, dizer isso: o desconto pode vir de um fator
  comum (por exemplo, juro longo).
- Uma linha por ativo: posição, ativo e setor, preço atual, preço justo e faixa de 80%,
  margem em % e faixa da margem, erro típico no backtest, marcas (dentro da margem de erro,
  desatualizada, preço justo ajustado), principal preditor com a direção, data da análise e
  link para o relatório.
- Filtro por setor e, no fim, a lista dos ativos sem estimativa confiável.
- Clicar em qualquer ativo da página, inclusive os da lista à parte, abre a tela do ativo,
  com os gráficos de 10 anos.

No relatório, a foto do ranking dos ativos não muda depois; os preços usados nela entram na
cópia dos dados.

Na sessão: depois de rodar um ativo, diga a posição dele (ex.: "3º de 12 ativos com
estimativa"). Depois de um lote ou da atualização rápida, mostre em texto curto a data dos
preços e os 10 primeiros ativos, um por linha (posição, ticker, margem em %, faixa da
margem e marcas), com o total e o link da página.

Em todos esses lugares, use rótulos neutros, como "maior margem"; nunca "melhores" ou
"oportunidades".

### Gráficos do ativo
Dois gráficos dos últimos 10 anos, um sobre o outro, com o mesmo eixo de tempo (zoom e
cursor sincronizados) e os valores ao passar o mouse. Se houver menos de 10 anos de dados,
mostrar o que houver e dizer desde quando.

1. Preço: fechamento diário ajustado por desdobramentos e grupamentos, com a opção de ver
   o retorno com proventos reinvestidos. Marcar os eventos da seção "Mudanças estruturais"
   e as quebras estruturais detectadas.
2. Preço justo ao longo do tempo: o preço justo mensal e a faixa de 80% sobre o preço do
   gráfico 1, sem correção pela inflação nem proventos reinvestidos. Ao passar o mouse:
   data, preço, preço justo, faixa de 80% e margem de segurança.
   - Cada ponto é a estimativa walk-forward do backtest: o que o modelo teria estimado
     naquela data só com os dados divulgados até ali. Nunca mostrar valores ajustados com
     dados posteriores (dentro da amostra): eles colam no preço e dão uma falsa ideia de
     precisão.
   - A legenda diz que são estimativas simuladas com a versão do modelo da análise mais
     recente (indicar qual), e não análises feitas na época. As análises do histórico
     aparecem como pontos.
   - O preço justo de cada data, a faixa e os pontos das análises do histórico recebem os
     desdobramentos e grupamentos posteriores àquela data, como o preço.
   - O último ponto cheio é o preço justo da análise mais recente, o do relatório. Depois
     dele, a linha e a faixa seguem tracejadas até o último fechamento, com o mesmo ajuste
     do ranking dos ativos; o fim da linha é o preço justo que a tela mostra.
   - Opcional, desligada no início: quando houver a projeção, os três cenários de 12 meses
     à direita.
   - Se o ativo não tiver estimativa confiável, mostrar o motivo no lugar deste gráfico.

Na tela, os gráficos vão até o último fechamento; no relatório, ficam como na data do
relatório.

### Saída
Tela do ativo, em HTML, uma por ativo, refeita a cada rodada e a cada atualização rápida.
No topo, poucas informações:
1. Ativo e setor; preço atual e data do preço; data da análise e marcas (como
   desatualizada).
2. Preço justo e faixa de 80%.
3. Margem de segurança em % e em R$, faixa da margem, erro típico no backtest e se a
   margem, em módulo, o supera.
4. Top 5 preditores com a direção.
5. Posição no ranking dos ativos (ex.: 3º de 12 ativos com estimativa) e links para o
   relatório e para a página do ranking dos ativos.
Abaixo, os dois gráficos de 10 anos (ver "Gráficos do ativo").

Relatório em HTML, que abre no navegador e pode ser salvo em PDF:
1. Resumo em até 5 frases.
2. Preço justo e margem de segurança: sinal e tamanho da diferença explicados, com os dois
   gráficos de 10 anos.
3. Posição entre os ativos analisados: a foto do ranking dos ativos na data do relatório,
   com este ativo destacado. Clicar num ativo da foto abre a tela dele.
4. Ranking dos preditores.
5. Cenário macro e projeção de 12 meses.
6. Mudanças estruturais do setor.
7. Backtest.
8. Checagem com valuation tradicional.
9. Limitações e qualidade dos dados.
10. Metodologia e fontes.

Página do ranking dos ativos, em HTML: ver a seção "Ranking dos ativos analisados".

Tudo em português, números e datas no formato brasileiro, com o aviso "estimativa de
modelo, não é recomendação de investimento".

### Requisitos técnicos
- Roda no meu PC pessoal (Windows), em Python, com ambiente virtual.
- Código e relatórios em Documentos\margem_de_seguranca; ambiente Python e cache de dados
  fora do OneDrive (por exemplo em %LOCALAPPDATA%\margem_de_seguranca).
- Junto dos relatórios, fora do cache, ficam as telas, o histórico e o que cada tela usa da
  última rodada (estimativas walk-forward, preditores, eventos). Sem o cache, a
  atualização rápida ainda refaz as telas.
- Só fontes gratuitas, respeitando os limites de acesso; cache para não baixar tudo de novo.
- Até 10 minutos por ativo, exceto na primeira carga de dados e quando o walk-forward
  inteiro precisar ser refeito (ativo novo ou nova versão do modelo), que devem mostrar o
  progresso. Para caber no prazo, reduza sementes ou a busca de hiperparâmetros; nunca
  faça escolhas com o período todo. Atualização rápida do ranking dos ativos, com as
  telas: até 2 minutos para 30 ativos.
- Gráficos interativos que funcionem sem internet, com a biblioteca de gráficos salva uma
  vez junto das páginas, e não copiada em cada arquivo.
- Reproduzível: sementes fixas e cópia dos dados usados em cada relatório.
- Se os dados forem insuficientes ou o modelo não for confiável para o ativo, diga isso em
  vez de forçar um número.

### Como entregar
1. Antes de programar, apresente o plano: fontes, preditores por setor, arquitetura,
   critério de estimativa confiável e desenho da tela do ativo, com os gráficos, e da
   página do ranking dos ativos. Espere meu OK.
2. Fase 1: um ativo (BBAS3), dados essenciais, modelo com walk-forward, tela com os dois
   gráficos e relatório. Já gravar cada análise no histórico, no formato final.
3. Fase 2: outros setores, ranking completo dos preditores, ranking dos ativos analisados e
   backtest com a margem como sinal.
4. Fase 3: mudanças estruturais e projeção por cenários.
5. Aceite: num único pedido, rodar um banco (BBAS3), uma indústria (WEGE3) e uma elétrica
   (CMIG4); mostrar a tela e o relatório de cada um e a página do ranking dos ativos com os
   três, com o mesmo preço justo de cada relatório. Na página, clicar em cada ativo abre a
   tela com os dois gráficos de 10 anos, e o último ponto cheio do preço justo é o do
   relatório. Depois, pedir o ranking dos ativos: nenhum modelo é treinado e nenhum preço
   justo muda. Com o prazo de 30 dias mudado para 0, os três aparecem desatualizados.
   Rodar BBAS3 de novo não pode duplicá-lo no ranking dos ativos.
6. Ainda no aceite, teste de vazamento em cada ativo: para 3 meses sorteados, trocar por
   valores aleatórios tudo o que foi divulgado depois da data (preços, proventos,
   balanços, macro) e refazer do zero o processo inteiro, inclusive seleção de variáveis,
   normalização e escolha do modelo. Na mesma base de ajuste, a estimativa daquela data
   tem de ser igual, ao centavo, à guardada no backtest, que alimenta o gráfico.

## Plano aprovado (20/11/2026)

### 1. Fontes

| Dado | Fonte gratuita | Entrada *point-in-time* | Observação |
|---|---|---|---|
| Preço diário bruto | B3 COTAHIST (arquivos anuais) | Fechamento oficial do dia | Primeira carga de cerca de 1 GB; guardar no cache só o necessário, em parquet |
| Desdobramentos, grupamentos, bonificações, proventos | Eventos corporativos da B3 | Data "com" de cada evento | Dois fatores próprios: um só com desdobramentos e grupamentos (alvo e dados por ação) e outro com proventos (retorno com proventos reinvestidos) |
| Conferência do preço | Yahoo Finance, coluna `Close` (só desdobramentos); nunca `Adj Close` | — | Só para conferir ou na falta da B3. O Yahoo erra eventos de algumas ações da B3 |
| Ações em circulação, tesouraria, free float, recompras, emissões, controle estatal ou privado | CVM: composição do capital (ITR/DFP), FRE, FCA, cadastro | Data de entrega | — |
| Balanços | CVM dados abertos: DFP e ITR (desde 2010 e 2011; confirmar na primeira carga) | `DT_RECEB` (data de entrega) | Reapresentação: ver abaixo |
| Dados próprios de bancos (Basileia, carteira por segmento, PDD) | BCB IF.data, além do ITR/DFP | Data-base + 90 dias (conservador) | O IF.data não registra a data exata de publicação |
| Selic, CDI, IPCA, IGP-M, dólar, IBC-Br, crédito, inadimplência, spread | BCB SGS | Calendário de divulgação de cada série | O SGS só guarda a série revisada |
| Expectativas | Boletim Focus (API Olinda do BCB) | Data de cada expectativa | Já vem *point-in-time* |
| Juro curto (1 ano), juro longo (5 a 10 anos), inclinação | Curva DI x pré da B3 (taxas referenciais); se falhar, prefixados do Tesouro Direto | Diário | — |
| Juro real | NTN-B longa (Tesouro Transparente) | Diário | — |
| Desemprego, massa salarial, PMC, produção industrial | IBGE SIDRA | Calendário do IBGE | Sem versões da época: dizer em Limitações e na legenda do gráfico do preço justo |
| Risco-país | EMBI+ Brasil (Ipeadata) | Diário | No lugar do CDS de 5 anos, que não tem fonte gratuita com histórico. Se o EMBI+ estiver descontinuado, avisar e propor alternativa |
| Fluxo estrangeiro na B3 | Participação dos investidores (B3) | Data de publicação | Tamanho do histórico a verificar |
| Fed funds, Treasuries de 2 e 10 anos, VIX, Brent, minério de ferro, índice de commodities | FRED, com versões da época no ALFRED | ALFRED | Precisa da chave gratuita do FRED (ver Pendências) |
| DXY, S&P 500, Ibovespa | Yahoo (`DX-Y.NYB`, `^GSPC`, `^BVSP`) | Diário | — |
| PMI da China | NBS (oficial); se não houver acesso estável, indicador antecedente da OCDE para a China | Mensal | O Caixin PMI é pago |

- **Balanço reapresentado:** os arquivos abertos da CVM trazem só a versão mais recente de
  cada documento (confirmar). Para documentos com versão maior que 1, buscar a original no
  sistema da CVM (RAD/ENET). Se não der, a versão reapresentada só entra a partir da data
  dela e, antes disso, vale o trimestre anterior. Registrar em Limitações.
- **Séries macro sem calendário histórico:** usar uma defasagem conservadora por série,
  documentada na configuração.
- **Cobertura:** cada rodada gera a tabela série → fonte → período → faltas.
- **Setor do ativo:** classificação setorial da B3, ligada aos setores do `setores.yaml` por
  um mapa editável.

### 2. Preditores por setor (`config\setores.yaml`, editável)
- **Bancos (BBAS3):** inadimplência acima de 90 dias (PF, PJ e rural), PDD sobre a carteira,
  NIM, Basileia, índice de eficiência, crescimento da carteira, spread (ICC), Selic,
  inclinação da curva. Extras aprovados: soja, milho, inadimplência do crédito rural.
  Nos bancos, saem EBITDA, liquidez corrente, ciclo de caixa e dívida líquida; ficam ROE,
  ROA, payout, LPA e VPA.
- **Indústria (WEGE3):** utilização da capacidade instalada (FGV e CNI), produção
  industrial, produção de bens de capital, confiança da indústria (FGV; o PMI industrial do
  Brasil é pago), cobre e aço, IPA industrial, câmbio. Extras aprovados: euro e produção
  industrial dos EUA.
- **Energia elétrica (CMIG4):** reservatórios do Sudeste/Centro-Oeste e GSF (ONS), PLD
  (CCEE), carga e consumo (ONS e EPE), IGP-M e IPCA dos reajustes, reajustes e revisões
  tarifárias (ANEEL, também como eventos).
- **Varejo, saúde e farmacêutico, educação, commodities e construção:** entram no YAML com os
  exemplos da especificação, inativos até a Fase 2.
- **Setor fora do mapa:** a rodada desse ativo para e mostra a proposta de preditores ao
  usuário; o resto do lote segue.

### 3. Arquitetura

```
Documentos\margem_de_seguranca\
  CLAUDE.md
  margem\            pacote Python: fontes, pit, indicadores, modelo, valuation,
                     estruturais, cenarios, ranking_dos_ativos, saida (templates Jinja2)
  config\            configuracao.yaml (faixas da margem, prazo de 30 dias, critério de
                     estimativa confiável, sementes, custos), setores.yaml, eventos.yaml
  saida\
    lib\echarts.min.js            biblioteca de gráficos, uma cópia para todas as páginas
    ranking_dos_ativos.html
    ativos\BBAS3.html + ativos\BBAS3.dados.js     tela do ativo; dados em .js para abrir sem servidor
    ultima_rodada\BBAS3\          walk-forward, preditores, eventos, quebras, mudanças estruturais
    relatorios\BBAS3\2026-11-20_1530\relatorio.html + dados\   cópia dos dados usados e dos preços da foto
    historico_analises.jsonl      só acrescenta linhas
    fora_do_ranking_dos_ativos.jsonl
    lote_atual.json               para retomar um lote que parou
%LOCALAPPDATA%\margem_de_seguranca\
  .venv\  cache\ (bruto por fonte, em parquet, com a data do download)  .env
```

Linha de comando (`ms`):
- `ms rodar BBAS3 [WEGE3 ...]`: rodada completa, de um ativo ou em lote, com retomada.
- `ms abrir BBAS3`: atualização rápida e abre a tela.
- `ms ranking-dos-ativos`: atualização rápida e abre a página.
- `ms refazer --desatualizados` ou `--todos`: mostra a lista e roda como lote.
- `ms tirar BBAS3`: registra em `fora_do_ranking_dos_ativos.jsonl` e refaz a página.
- `ms teste-vazamento BBAS3 --meses 3`.

Skill no nível do usuário: `%USERPROFILE%\.claude\skills\margem-de-seguranca\SKILL.md`.
- Entende os pedidos da especificação e as variações deles, e chama os comandos com o
  Python do `.venv`.
- Informa o tempo estimado dos lotes e abre as páginas com `start ""`.
- A pesquisa na web das mudanças estruturais é feita pelo próprio Claude na sessão. Ele
  grava o resultado, com fonte e data, em
  `saida\ultima_rodada\<TICKER>\mudancas_estruturais.json`, e o relatório lê esse arquivo.

Regras de implementação:
- Toda função de dados recebe `data_corte` e devolve só o que foi divulgado até ela.
- Nenhuma validação sorteada: todas em ordem de tempo. O `early_stopping` embutido do
  scikit-learn (MLP e gradient boosting) sorteia a validação e não serve. Usar early
  stopping próprio, num bloco final do treino em ordem de tempo.
- Gráficos com ECharts: uma instância com dois painéis, com eixo de tempo, zoom e cursor
  sincronizados, sem internet.
- Determinismo, para o teste de vazamento fechar ao centavo: sementes fixas,
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS` e `OPENBLAS_NUM_THREADS` iguais a 1, float64 e
  ordenação estável.
- Bibliotecas previstas: pandas, numpy, pyarrow, scikit-learn, shap, statsmodels, ruptures,
  requests, jinja2, pyyaml.
- Datas em ISO nos arquivos; formato brasileiro nas páginas.

### 4. Modelo e walk-forward
- **Base mensal (fim de mês):** fundamentos valem até a divulgação seguinte; macro é o último
  dado já divulgado.
- **Alvos em disputa:** log do preço, log P/L, log P/VP e log EV/EBITDA (fora dos bancos).
  Cada um é convertido em R$ pelo LPA, VPA ou EBITDA, dívida líquida e ações do mês. Os
  múltiplos nunca entram como preditores.
- **Modelos:**
  - rede MLP pequena (1 a 2 camadas, L2, early stopping num bloco final do treino separado
    por 12 meses, média de 5 a 10 sementes);
  - passeio aleatório, Ridge/ElasticNet, gradient boosting e ensemble.
- **Retreino:** no mês seguinte a cada nova divulgação da CVM; estimativa todo mês com o
  último modelo treinado. O preço do mês estimado nunca entra no treino.
- **Em cada retreino, só com dados até a data:**
  - remoção das variáveis redundantes (|correlação| > 0,9) e seleção das variáveis;
  - normalização, hiperparâmetros e early stopping;
  - escolha de alvo, modelo e ensemble pelos erros de 6 e 12 meses já fechados na data.
  Nos primeiros retreinos, quando ainda não há meses fora da amostra para comparar, a
  escolha usa um walk-forward interno, dentro dos dados de treino.
- **Histórico curto (menos de cerca de 8 anos):** painel com os pares do setor, alvo em
  múltiplo e aviso no relatório.
- **Faixa de 80% e erro típico:** como na especificação; sem faixa com menos de 24 meses.
- **Estimativas guardadas:** V, P10 e P90 de cada mês. Uma nova rodada calcula só os meses
  novos, salvo nova versão do modelo.
- **Versão do modelo:** versão do código mais um hash da configuração do setor. Editar o
  `setores.yaml` refaz só o walk-forward daquele setor.
- **Ranking dos preditores:**
  - SHAP do modelo final, conferido por permutação, em % do total;
  - top 10 com direção e família;
  - mudança em 3 anos, comparando com os retreinos de 12, 24 e 36 meses atrás;
  - os 5 maiores explicados em linguagem simples.
- **Backtest:**
  - MAPE, RMSE e R² contra o preço de 6 e 12 meses depois, para cada modelo e para o
    passeio aleatório;
  - % de meses dentro da faixa de 80%;
  - margem como sinal (desconto moderado e forte), sem sobreposição, contra Ibovespa e CDI,
    com custos de 0,25% ida e volta.
- **Quebras estruturais:** CUSUM e Bai-Perron no preço e no resíduo da relação entre
  preditores e preço; Chow nas datas dos eventos (`config\eventos.yaml` mais a pesquisa
  na web).
- **Projeção de 12 meses:** o cenário base usa o Focus. Otimista e pessimista deslocam
  juros, câmbio e atividade pelos percentis 10 e 90 das surpresas históricas em 12 meses.
- **Valuation tradicional:**
  - modelo de dividendos para bancos, fluxo de caixa descontado para os demais;
  - múltiplos contra o histórico de 10 anos e contra os pares;
  - Graham: √(22,5 × LPA × VPA);
  - Bazin: dividendos de 12 meses ÷ 6% (configurável).
  Se o modelo divergir muito, explicar por quê.

### 5. Critério de estimativa confiável (configuração, o mesmo para todos os ativos)
1. Pelo menos 60 meses de estimativas fora da amostra e a faixa de 80% calculável.
2. Não perder para o passeio aleatório, nos últimos 60 meses:
   - MAPE contra o preço 12 meses depois ≤ 1,00 × o MAPE do passeio aleatório;
   - em 6 meses, ≤ 1,10 ×.
   O limite é revisto depois de BBAS3.
3. Preço dentro da faixa de 80% em 70% a 90% dos meses.
4. Erro típico ≤ 35%.
5. Dados na data:
   - última divulgação da CVM usada com no máximo 180 dias;
   - preditores obrigatórios do setor com pelo menos 80% de cobertura;
   - negócios nos últimos 5 pregões.
6. Sanidade: V > 0; 0,2 ≤ V ÷ P ≤ 5; variação entre sementes ≤ 10% (usada só como
   checagem, nunca para a faixa).

Sem estimativa confiável: mostrar o motivo, sem forçar um número.

### 6. Tela do ativo (valores ilustrativos)

```
BBAS3 · Banco do Brasil · Bancos        Preço R$ 21,40 (19/11/2026) · Análise 20/11/2026 [desatualizada: ITR novo]
Preço justo R$ 25,10   (faixa de 80%: R$ 19,80 a R$ 31,00)
Margem +14,7% (R$ 3,70) · perto do preço justo · erro típico 18% → dentro da margem de erro
Top 5 preditores: Selic ↓ · inadimplência ↓ · ROE ↑ · juro longo ↓ · carteira ↑
3º de 12 ativos com estimativa · [Relatório] · [Ranking dos ativos]
┌ Preço diário (só desdobramentos) [ ] retorno com proventos reinvestidos ── eventos ▲ e quebras │ ┐
└ Preço justo mensal + faixa de 80% sobre o preço · pontos = análises do histórico ·
  linha tracejada depois da última análise · [ ] cenários de 12 meses                          ┘
  (zoom e cursor juntos; ao passar o mouse: data, preço, preço justo, faixa e margem)
```

### 7. Página do ranking dos ativos

```
Preços de 19/11/2026 · 12 com posição · 3 desatualizados · 2 sem estimativa confiável
⚠ 6 dos 10 primeiros têm "juro longo" como principal preditor: o desconto pode vir de um fator comum.
[Filtro: setor ▾]
Pos | Ativo · setor | Preço | Preço justo (faixa 80%) | Margem | Faixa da margem | Erro típico | Marcas | Principal preditor | Análise | Relatório
Ordem: da maior para a menor margem em %. Clicar numa linha abre a tela do ativo.
— Sem estimativa confiável: ativo · data · motivo (também clicáveis)
```

- Mesma data de preço para todos: o último fechamento.
- Ajuste depois da análise:
  - desdobramento ou grupamento: divide pelo fator;
  - dividendos e JCP: multiplica pelo mesmo fator do retorno com proventos reinvestidos.
- Desatualizada: 30 dias ou mais (configurável), ITR ou DFP mais novo na CVM, ou versão
  anterior do modelo.
- Rótulos neutros ("maior margem"), nunca "melhores" ou "oportunidades".

### 8. Linha do histórico (`saida\historico_analises.jsonl`, valores ilustrativos)

```json
{"ticker": "BBAS3", "setor": "Bancos", "data_analise": "2026-11-20T15:30:00-03:00",
 "data_preco": "2026-11-19", "preco": 21.40, "preco_justo": 25.10, "p10": 19.80, "p90": 31.00,
 "erro_tipico": 0.18, "principal_preditor": "selic", "direcao": "baixa",
 "estimativa": true, "motivo_sem_estimativa": null,
 "ultima_divulgacao_cvm": {"documento": "ITR", "data_referencia": "2026-09-30",
                           "data_entrega": "2026-11-12", "versao": 1},
 "versao_modelo": "1.0.0+setor-ab12cd",
 "relatorio": "relatorios/BBAS3/2026-11-20_1530/relatorio.html"}
```
(No arquivo, uma linha por análise; aqui quebrada só para leitura.)

### 9. Fases e tempo
- **Fase 1 (BBAS3):** dados essenciais, walk-forward, tela com os dois gráficos, relatório e
  histórico no formato final.
- **Fases 2 e 3 e aceite:** como na especificação.
- **Tempo:**
  - primeira carga de dados: cerca de 20 a 40 minutos, conforme a conexão;
  - primeiro walk-forward de cada ativo: cerca de 10 a 25 minutos, com o progresso na tela;
  - rodadas seguintes: menos de 10 minutos;
  - atualização rápida: até 2 minutos para 30 ativos.

### 10. Aviso honesto
- Um ativo tem cerca de 190 meses de dados, e os fundamentos só mudam por trimestre. Na
  prática, isso dá umas 60 observações independentes.
- É provável que, em vários ativos, o passeio aleatório ou a regressão linear vençam a rede
  neural. Quando acontecer, o relatório diz isso.
- Com 12 meses sem sobreposição, a margem como sinal tem no máximo uns 10 sinais em 10
  anos, pouco para tirar conclusão.
