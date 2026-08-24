# Atividades — AutoML com PyCaret, do dado ao deploy

> Antes de começar, execute o roteiro completo do [README](README.md) pelo menos
> uma vez e guarde os resultados de referência (ranking do `compare_models`, AUC
> de teste, F1, limiar escolhido, matriz de confusão e importâncias). Todo
> exercício abaixo é comparado com **essa linha de base**. Um experimento sem
> baseline não prova nada.

Regra permanente para todos os exercícios: **mexa em uma variável por vez** e
registre o resultado em uma tabela. Mudar três coisas e ver o AUC subir não diz
qual das três funcionou.

E uma regra específica desta aula: **sempre reporte o desvio entre dobras junto
com a média.** Diferença de 0,001 com desvio de 0,0016 não é diferença — é
ruído. Relatório que apresenta só a média está escondendo metade da informação.

Linha de base desta implementação (`session_id=42`):

| Métrica | Valor |
|---|---|
| Features após o `setup()` | 90 |
| Modelo vencedor do `compare_models` | LightGBM (AUC 0,9276) |
| Modelo publicado | LightGBM ajustado |
| AUC de validação cruzada (treino, 5 dobras) | 0,9281 ± 0,0016 |
| AUC de validação (hold-out) | 0,9293 |
| **AUC-ROC (teste)** | **0,9275** |
| **F1 (teste, limiar 0,41)** | **0,721** |
| **Acurácia (teste)** | **86,71%** |
| Acurácia do modelo trivial (Dummy) | 76,38% |
| Tempo total `compare_models` + `tune_model` | 188 s |

---

## Nível 1 — Compreensão

**1.1 O `setup()` linha a linha.** Sem rodar o código, diga o que cada um destes
argumentos substitui do `Preprocessador` da aula de MLP, e o que aconteceria se
você o omitisse:

| Argumento | O que substitui | Se eu omitir? |
|---|---|---|
| `numeric_features` / `categorical_features` | ? | ? |
| `numeric_imputation="median"` | ? | ? |
| `max_encoding_ohe=50` | ? | ? |
| `normalize=True` | ? | ? |
| `test_data=df_val` | ? | ? |
| `session_id=42` | ? | ? |

**1.2 Com estado x sem estado.** Classifique cada transformação abaixo e
justifique em uma linha:

| Transformação | Com ou sem estado? | Pode ocorrer antes de separar? |
|---|---|---|
| `log1p(capital-gain)` | ? | ? |
| `(x − média) / desvio` | ? | ? |
| `idade → faixa_etária` com cortes fixos (18, 30, 45, 65) | ? | ? |
| `idade → decil da idade` | ? | ? |
| Substituir `occupation` pela média de `y` naquela ocupação | ? | ? |

A última é *target encoding*. Explique por que ela é a mais perigosa das cinco.

**1.3 Por que 90 e não 95.** A aula de MLP produziu 95 features; esta produz 90.
Explique as duas diferenças de convenção (veja §5, ETAPA 2 do README) e responda:
(i) a coluna `sex` como uma única coluna 0/1 perde informação em relação a duas
colunas one-hot? Demonstre. (ii) Qual das duas convenções para a categoria
`Desconhecido` você adotaria num sistema de produção que recebe formulários
incompletos, e por quê?

**1.4 Leitura do ranking.** Olhando a tabela do `compare_models` no README,
responda sem rodar nada:
(a) Por que o `Naive Bayes` tem revocação 0,972 e acurácia 40%?
(b) Por que o `Dummy` tem AUC exatamente 0,500?
(c) O `Ridge Classifier` aparece com AUC 0,894, mas não produz probabilidade
calibrada. Como o PyCaret calculou esse AUC, e para que esse modelo serve e não
serve?
(d) LightGBM e XGBoost empatam em 0,9276. Qual critério você usaria para
desempatar, e por que **não** o quarto decimal do AUC?

**1.5 O limiar padrão.** `predict_model` corta em 0,5. Explique, para um gestor
que não é da área, o que muda concretamente na vida das pessoas classificadas
quando o limiar vai de 0,50 para 0,41 — use os números da matriz de confusão do
README (limiar 0,50: FP=736, FN=1325; limiar 0,41: FP=1108, FN=1056).

---

## Nível 2 — Experimentação controlada

Preencha a tabela para cada item, sempre reexecutando `train.py` e `evaluate.py`:

| Experimento | AUC val. cruzada (± desvio) | AUC teste | F1 teste | Limiar | Tempo | Observação |
|---|---|---|---|---|---|---|
| Linha de base | 0,9281 ± 0,0016 | 0,9275 | 0,721 | 0,41 | 188 s | |
| ... | | | | | | |

**2.1 Sem padronização.** Rode com `normalize=False` no `config.py`. Compare o
efeito **por família de modelo**, olhando a tabela inteira do `compare_models`:
quais modelos praticamente não mudaram e quais desabaram? Explique por que
árvores são indiferentes à escala e KNN e MLP não são.

**2.2 A armadilha do `max_encoding_ohe`.** Volte para o padrão do PyCaret
(`max_encoding_ohe=25`) e rode `python src/setup_experimento.py`.
(i) Quantas features agora? (ii) O que aconteceu com `native-country`?
(iii) O AUC mudou? (iv) Explique por que essa mudança de esquema, feita sem
aviso pela biblioteca, é mais perigosa do que uma queda de AUC.

**2.3 Sem o `log1p`.** Esvazie `COLUNAS_LOG` em `config.py`. Compare o AUC de
cada família e a importância por permutação de `capital-gain`. **Atenção:** o
LightGBM praticamente não deve mudar. Explique por quê, e diga para qual dos 13
modelos da tabela essa transformação mais importa.

**2.4 Desbalanceamento: SMOTE x limiar.** Rode
`python src/train.py --balancear` (liga `fix_imbalance=True`).
Compare com a linha de base: (i) o AUC mudou? (ii) a revocação no limiar 0,5
mudou? (iii) **abra `outputs/calibracao.png` nos dois casos** e descreva o que
aconteceu com a curva de confiabilidade. (iv) Conclua: para atingir mais
revocação, é melhor reequilibrar a classe ou mover o limiar? Justifique
considerando um sistema em que a probabilidade prevista é usada para definir
preço, e não só para dizer sim/não.

**2.5 Ordenar por outra métrica.** Rode `python src/comparar.py --metrica Accuracy`,
`--metrica Recall` e `--metrica F1`. Para cada uma, anote em que posição fica o
`Dummy` e qual modelo vence. Escreva um parágrafo sobre por que a métrica de
ordenação é uma decisão de projeto e não de gosto.

**2.6 A busca vale a pena?** Rode `train.py` com `n_iter_tune` = 5, 20 e 50 em
`config.py`. Registre o ganho de AUC e o tempo. Monte o gráfico ganho × tempo.
Considerando o desvio entre dobras (0,0016), a partir de que ponto o ganho deixa
de ser distinguível de ruído?

**2.7 Combinar modelos.** Rode `python src/train.py --blend` (voto suave entre os
3 melhores). O ganho compensa? Compare também: tempo de inferência do modelo
combinado x do LightGBM sozinho, e tamanho do `.pkl`. Em que cenário de produção
você aceitaria triplicar o custo de inferência por esse ganho?

**2.8 Calibração explícita.** Rode `python src/train.py --calibrar` (regressão
isotônica). O AUC mudou? E a curva de calibração? Explique por que calibrar
quase não mexe no AUC — o que a operação preserva?

**2.9 O limiar no teste.** Rode `python src/evaluate.py --limiar 0.3`, `0.5` e
`0.8`. Monte a tabela precisão × revocação × acurácia × F1 × AUC. **O AUC muda?**
Explique em uma frase por que não, e escolha o limiar que você usaria se o custo
de um falso negativo fosse 10 vezes o de um falso positivo.

**2.10 Estabilidade da semente.** Rode o pipeline com `session_id` = 0, 42 e
2024. Quanto o AUC de teste varia? E o modelo vencedor, muda? Se mudar, o que
isso diz sobre confiar no primeiro lugar de um `compare_models`?

---

## Nível 3 — Implementação

**3.1 Vazamento na prática.** Modifique `train.py` para chamar
`criar_experimento` com o dataframe **completo** (treino + validação + teste)
como `data`, deixando o PyCaret separar sozinho. Meça o AUC do hold-out e depois
o AUC no `df_teste` original. (i) O hold-out melhorou? (ii) E o teste?
(iii) Explique por que essa é a forma mais perigosa de erro metodológico: o que
você veria se só tivesse o hold-out para julgar.

**3.2 Mover o `log1p` para dentro do pipeline.** Use o argumento
`custom_pipeline` do `setup()` com um `FunctionTransformer` que aplique `log1p`
nas duas colunas, posicionado no início. Depois:
(i) remova o `aplicar_log1p` de `data.py` e o bloco correspondente de
`deploy/api.py`; (ii) reexporte o modelo; (iii) **tente subir a API a partir de
um diretório onde `src/` não seja importável**. O que acontece, e por quê?
(iv) Conclua: qual é o custo real de colocar código próprio dentro de um pipeline
serializado com pickle? Este exercício vale mais pela falha do que pelo sucesso —
documente a mensagem de erro exata.

**3.3 Métricas por subgrupo.** Escreva um script que calcule AUC, precisão,
revocação e **taxa de falsos negativos** separadamente para `sex = Male` e
`sex = Female`, e para as categorias de `race`, usando o modelo já treinado. A
qualidade do modelo é a mesma para todos os grupos? Qual grupo é mais
prejudicado, e de que forma?

**3.4 Remover o atributo sensível resolve?** Retire `sex` e `race` de
`COLUNAS_CATEGORICAS`, retreine e responda com números:
(i) quanto o AUC de teste mudou? (ii) a diferença na taxa de falsos negativos
entre homens e mulheres do exercício 3.3 desapareceu? Use o resultado para
discutir o conceito de **proxy**: quais colunas restantes ainda carregam
informação sobre sexo? *(Dica: olhe as categorias de `relationship` e a
importância por permutação — `sex` já aparecia com apenas 0,0027.)*

**3.5 Interpretabilidade com SHAP.** Use `exp.interpret_model(modelo)` para gerar
o gráfico SHAP de resumo. Compare as três leituras de importância que você agora
tem: permutação nas colunas originais, importância nativa do LightGBM, e SHAP.
Onde elas concordam? Onde discordam, e por quê? Escolha uma pessoa específica do
conjunto de teste que o modelo errou e explique a previsão dela com SHAP.

**3.6 Enxugar o modelo.** A importância por permutação mostra `native-country`
com queda de AUC de 0,0000 — e ela custa 42 colunas. Retreine sem essa coluna e
meça: (i) AUC de teste; (ii) número de features; (iii) tamanho do `.pkl`;
(iv) latência de inferência. O modelo menor é pior? Se não for, por que a
biblioteca não fez isso sozinha?

**3.7 Robustez em produção.** Envie para a API cadastros patológicos: campos
faltando, `age = 999` (rejeitado pelo Pydantic — por quê?), uma
`native-country` que não existe no treino, todos os campos nulos, e
`education_num` com underscore em vez de hífen. O serviço responde ou quebra?
**O caso do underscore é o mais importante:** ele não gera erro nenhum e a
previsão muda. Meça quanto. Para cada caso, decida qual **deveria** ser o
comportamento correto e implemente a mudança que faltar.

**3.8 Reproduzir o MLP dentro do PyCaret.** Rode
`python src/train.py --modelo mlp` e ajuste `hidden_layer_sizes` para `(64, 32)`
via `tune_model` com `custom_grid`, aproximando a arquitetura da aula anterior.
Compare com o resultado artesanal (AUC 0,9053, F1 0,682). Se o número ficar
diferente, liste as causas possíveis — e note que a aula de MLP usava
`pos_weight`, `BatchNorm`, early stopping por AUC de validação e limiar
otimizado, nenhum dos quais o `MLPClassifier` faz.

---

## Nível 4 — Desafio final (Grupo de 2 Alunos — Entrega dia 26/08/2026)

Escolha **um** dos caminhos:

**Caminho A — Novo conjunto de dados.**
Troque o Adult por outro problema tabular (por exemplo *Bank Marketing*,
*Telco Churn* ou *Credit Card Default*). Você terá de reescrever `COLUNAS`,
`COLUNAS_NUMERICAS` e `COLUNAS_CATEGORICAS`, tratar os ausentes daquele domínio
e revisar os argumentos do `setup()`. Documente o que precisou mudar no pipeline
e **o que funcionou sem alteração nenhuma** — a segunda lista é a evidência de
que o projeto foi bem estruturado. Compare também: o modelo vencedor foi o
mesmo? Se não, o que no dado explica a diferença?

**Caminho B — AutoML contra o artesanal, com rigor.**
Faça a comparação das duas aulas virar um experimento defensável: mesma
partição, 5 sementes diferentes, intervalo de confiança para a diferença de AUC.
Responda com números: (i) a diferença entre LightGBM e o MLP artesanal é
estatisticamente significativa? (ii) quanto do ganho vem da família de modelo e
quanto vem do ajuste de hiperparâmetros (compare LightGBM de fábrica x ajustado)?
(iii) proponha e teste **uma** modificação no MLP que reduza a diferença — por
exemplo *embeddings* de categorias no lugar do one-hot.

**Caminho C — Serviço completo.**
Estenda a API com: (i) registro em arquivo de toda predição com
`confiabilidade = "baixa"`; (ii) endpoint `/metricas` com contagem de
requisições, latência p50/p95 e distribuição das probabilidades previstas;
(iii) detecção simples de *drift* comparando a distribuição das features
recebidas com as do treino (guarde as estatísticas do treino em
`metadados.json`); (iv) *cache* do pipeline e comparação de throughput entre
`/prever` e `/prever_lote` sob carga; (v) imagem Docker funcionando, com as
versões fixadas a partir de `metadados.json`.

**Entregáveis (todos os caminhos):**

1. Código no repositório, organizado e comentado.
2. Relatório de 3 a 5 páginas: problema, pipeline, **tabela de experimentos com
   média e desvio**, ranking dos modelos, matriz de confusão, análise de erros e
   conclusão.
3. Uma seção de **uma página** sobre implicações éticas: quem seria afetado por
   um erro do seu modelo, em qual direção, e o que você faria a respeito.
4. Uma seção de **meia página** respondendo: *neste projeto, o AutoML foi a
   escolha certa?* Com números, não com opinião.
5. Demonstração de 5 minutos com o serviço rodando ao vivo.

---

## Rubrica de avaliação (100 pontos)

| Critério | Pts | Excelente | Suficiente | Insuficiente |
|---|---|---|---|---|
| **Pipeline de dados** | 15 | Colunas tipadas com justificativa de domínio; `setup()` recebe só o treino; distinção com/sem estado explicada | Funciona, mas alguma escolha não é justificada | `setup()` no dataset completo, ou tipagem deixada por conta do dtype |
| **Seleção de modelo** | 15 | Ranking completo apresentado e lido; `Dummy` na tabela; métrica de ordenação justificada; empates tratados pelo desvio | Ranking apresentado sem análise | Só o primeiro lugar, sem linha de base |
| **Ajuste de hiperparâmetros** | 15 | Busca documentada (espaço, nº de combinações, melhor e pior); ganho comparado ao desvio entre dobras; `choose_better` explicado | Ajuste feito, resultado reportado sem contexto | Ajuste no conjunto de teste, ou ganho apresentado sem desvio |
| **Validação e limiar** | 15 | Limiar escolhido na validação e justificado pelo custo do erro; calibração analisada | Limiar padrão 0,5 usado, mas a escolha é mencionada | Limiar ou hiperparâmetros ajustados no teste |
| **Avaliação no teste** | 15 | Teste usado uma única vez; comparado ao trivial; precisão/revocação interpretadas; importâncias discutidas com a ressalva de causalidade | Métricas corretas com análise rasa | Só acurácia, sem linha de base |
| **Deploy** | 15 | Serviço funcionando; artefatos e **versões** versionados; paridade verificada com teste de integração; latência medida | API funciona, sem teste de integração | Não roda, ou artefato sem as versões registradas |
| **Comunicação** | 10 | Relatório claro, tabela de experimentos com desvio, discussão ética concreta, conclusões honestas | Relatório completo mas confuso | Incompleto ou sem evidências |

**Observação sobre honestidade experimental:** relatar um experimento que
*piorou* o resultado, com a análise do motivo, vale mais do que apresentar
apenas o melhor número. Reportar métricas obtidas ajustando hiperparâmetros ou
limiar no conjunto de teste zera o critério "Avaliação no teste".

**Observação sobre a acurácia:** um relatório que apresenta 86% de acurácia sem
mencionar que o modelo trivial faz 76,4% está incompleto, mesmo que todo o resto
esteja correto.

**Observação específica desta aula:** um relatório que apresenta o resultado do
`compare_models` sem explicar **o que cada argumento do `setup()` fez com os
dados** demonstra exatamente o risco que esta aula existe para combater — usar a
ferramenta sem entender o que ela automatizou. Isso reduz pela metade a nota de
"Pipeline de dados".
