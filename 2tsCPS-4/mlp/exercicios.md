# Atividades — MLP do dado ao deploy

> Antes de começar, execute o roteiro completo do [README](README.md) pelo menos
> uma vez e guarde os resultados de referência (AUC de teste, F1, limiar
> escolhido, curvas e matriz de confusão). Todo exercício abaixo é comparado com
> **essa linha de base**. Um experimento sem baseline não prova nada.

Regra permanente para todos os exercícios: **mexa em uma variável por vez** e
registre o resultado em uma tabela. Mudar três coisas e ver o AUC subir não diz
qual das três funcionou.

Linha de base desta implementação (semente 42):

| Métrica | Valor |
|---|---|
| Features de entrada | 95 |
| Parâmetros | 8.353 |
| AUC-ROC (teste) | 0,9053 |
| F1 (teste, limiar 0,65) | 0,682 |
| Acurácia (teste) | 83,31% |
| Acurácia do modelo trivial | 76,38% |

---

## Nível 1 — Compreensão

**1.1 Contagem de parâmetros.** Sem rodar o código, calcule quantos parâmetros
treináveis tem cada configuração, considerando 95 features de entrada e
`BatchNorm1d` em todas as camadas ocultas:

| # | Camadas ocultas | Parâmetros |
|---|---|---|
| a | `(64, 32)` | ? |
| b | `(128,)` | ? |
| c | `(256, 128, 64)` | ? |

Confira com `python src/model.py` depois de alterar `camadas_ocultas` em
`config.py`. Lembre: cada `Linear` sem viés tem `entrada × saída` pesos, cada
`BatchNorm1d` tem `2 × largura` parâmetros, e a camada de saída tem viés.

**1.2 O colapso linear.** Em `src/xor.py`, troque `nn.Tanh()` por
`nn.Identity()` e rode de novo. Quantos acertos a rede "com camada oculta"
consegue? Demonstre algebricamente, em duas linhas, por que o resultado é
exatamente o do perceptron de uma camada.

**1.3 One-hot x codificação ordinal.** A coluna `education` foi descartada em
favor de `education-num`. Explique por que `education-num` pode entrar direto
como número, enquanto `occupation` precisa virar 14 colunas one-hot. O que a
rede aprenderia de errado se `occupation` fosse codificada como 1, 2, 3, …, 14?

**1.4 Leitura de métricas.** Um modelo tem acurácia de 76,4% neste dataset.
Antes de olhar qualquer outra métrica, o que você já sabe sobre ele? Agora
suponha precisão 0,62 e revocação 0,76: escreva em português, para um gestor
que não é da área, o que cada um desses dois números significa neste problema.

**1.5 Por que sem sigmoide?** Explique, com base em `model.py` e `train.py`,
por que o `forward` devolve logits. O que acontece numericamente se você aplicar
sigmoide no modelo *e* usar `BCELoss`?

---

## Nível 2 — Experimentação controlada

Preencha a tabela para cada item, sempre reexecutando `train.py` e `evaluate.py`:

| Experimento | AUC val. | AUC teste | F1 teste | Limiar | Parâmetros | Observação |
|---|---|---|---|---|---|---|
| Linha de base | 0,9098 | 0,9053 | 0,682 | 0,65 | 8.353 | |
| ... | | | | | | |

**2.1 Sem padronização.** Em `Preprocessador.transform`, devolva os numéricos
crus (sem `(x - media) / desvio`). Treine 15 épocas. O que acontece com a
velocidade de convergência e com o AUC? Por quê?

**2.2 Sem o `log1p`.** Esvazie `COLUNAS_LOG` em `config.py`. Compare o AUC e
observe especialmente a importância por permutação de `capital-gain`. O que a
transformação estava fazendo?

**2.3 Capacidade do modelo.** Teste `camadas_ocultas = (16,)`, `(64, 32)` e
`(256, 128, 64)`. Relacione número de parâmetros, tempo por época e AUC. Mais
parâmetros sempre ajudam? Em que ponto a curva de validação começa a se afastar
da de treino?

**2.4 Desbalanceamento.** Rode `python src/train.py --sem-peso`. Compare
acurácia, precisão, revocação e AUC com a linha de base. **A acurácia sobe ou
desce? E a revocação?** Explique por que a acurácia é uma métrica traiçoeira
aqui e por que o AUC quase não muda.

**2.5 Taxa de aprendizado.** Rode com `--lr 1e-1`, `1e-3` e `1e-5` por 15
épocas. Descreva o comportamento de cada uma. Uma delas provavelmente não vai
aprender quase nada — explique o motivo.

**2.6 Dropout.** Compare `dropout = 0.0`, `0.3` e `0.7`. Em qual deles a
distância entre a perda de treino e a de validação é maior? Isso confirma o
papel regularizador do dropout?

**2.7 Ablação do BatchNorm.** Coloque `usar_batchnorm = False`. Treine com
`--lr 1e-3` e depois com `--lr 1e-2`. O BatchNorm permitiu taxas maiores?

**2.8 O limiar.** Rode `python src/evaluate.py --limiar 0.3`, `0.5` e `0.8`.
Monte a tabela precisão × revocação × acurácia. **O AUC muda?** Explique em uma
frase por que não, e escolha o limiar que você usaria se o custo de um falso
negativo fosse 10 vezes o de um falso positivo.

---

## Nível 3 — Implementação

**3.1 Baseline honesto.** Implemente uma regressão logística (um `nn.Linear`
de 95 → 1, sem camada oculta), treine nas mesmas condições e compare com o MLP:
AUC, F1, número de parâmetros e tempo. **Este é o experimento que justifica a
existência das camadas ocultas neste problema** — e o resultado pode surpreender
você. Discuta: o ganho do MLP compensa a complexidade adicional?

**3.2 Vazamento na prática.** Modifique `obter_dataloaders` para ajustar o
`Preprocessador` no dataframe **completo** (treino + validação) antes de
separar. Meça o AUC de validação e o de teste. A validação melhorou? E o teste?
Explique por que essa é a forma mais perigosa de erro metodológico: o que você
veria se só tivesse a validação para julgar.

**3.3 Métricas por subgrupo.** Escreva um script que calcule AUC, precisão,
revocação e **taxa de falsos negativos** separadamente para `sex = Male` e
`sex = Female`, e para as categorias de `race`. A qualidade do modelo é a mesma
para todos os grupos? Qual grupo é mais prejudicado, e de que forma?

**3.4 Remover o atributo sensível resolve?** Retire `sex` e `race` de
`COLUNAS_CATEGORICAS`, retreine e responda com números:
(i) quanto o AUC de teste mudou? (ii) a diferença na taxa de falsos negativos
entre homens e mulheres do exercício 3.3 desapareceu? Use o resultado para
discutir o conceito de **proxy**: quais colunas restantes ainda carregam
informação sobre sexo? *(Dica: olhe as categorias de `relationship`.)*

**3.5 Calibração.** Divida as probabilidades previstas em 10 faixas (0–0,1,
0,1–0,2, …) e, para cada faixa, compare a probabilidade média prevista com a
proporção real de positivos. Plote o resultado (diagrama de confiabilidade). O
modelo é **calibrado**? Investigue o efeito do `pos_weight` sobre isso: um
modelo treinado com peso ainda produz probabilidades interpretáveis como
frequências?

**3.6 Robustez em produção.** Envie para a API cadastros patológicos: campos
faltando, `age = 999` (rejeitado pelo Pydantic — por quê?), uma
`native-country` que não existe no treino, todos os campos nulos. O serviço
responde ou quebra? Para cada caso, decida qual **deveria** ser o comportamento
correto e implemente a mudança que faltar.

---

## Nível 4 — Desafio final (Grupo de 2 Alunos - Entrega dia 26/08/2026)

Escolha **um** dos caminhos:

**Caminho A — Novo conjunto de dados.**
Troque o Adult por outro problema tabular (por exemplo o *Bank Marketing* ou o
*Telco Churn*, ambos na UCI/Kaggle). Você terá de reescrever `COLUNAS`,
`COLUNAS_NUMERICAS` e `COLUNAS_CATEGORICAS`, tratar os ausentes daquele domínio
e reajustar a arquitetura. Documente o que precisou mudar no pipeline e o que
funcionou sem alteração nenhuma — a segunda lista é a evidência de que o
projeto foi bem estruturado.

**Caminho B — MLP contra gradient boosting.**
Implemente um baseline com `sklearn.ensemble.HistGradientBoostingClassifier` no
mesmo pipeline de dados e compare com o MLP em AUC, F1, tempo de treino e tempo
de inferência. Investigue *por que* árvores costumam vencer em dados tabulares
(leia Grinsztajn et al., 2022, citado no README) e proponha uma modificação no
MLP que reduza a diferença — por exemplo *embeddings* de categorias no lugar do
one-hot.

**Caminho C — Serviço completo.**
Estenda a API com: (i) endpoint de lote (`POST /prever_lote`), (ii) registro em
arquivo de toda predição com `confiabilidade = "baixa"`, (iii) endpoint
`/metricas` com contagem de requisições, latência média e distribuição das
probabilidades previstas, (iv) detecção simples de *drift* comparando a média
das features recebidas com as do treino, e (v) imagem Docker funcionando.

**Entregáveis (todos os caminhos):**

1. Código no repositório, organizado e comentado.
2. Relatório de 3 a 5 páginas: problema, pipeline, tabela de experimentos, curvas, matriz de confusão, análise de erros e conclusão.
3. Uma seção de **uma página** sobre implicações éticas: quem seria afetado por um erro do seu modelo, em qual direção, e o que você faria a respeito.
4. Demonstração de 5 minutos com o serviço rodando ao vivo.

---

## Rubrica de avaliação (100 pontos)

| Critério | Pts | Excelente | Suficiente | Insuficiente |
|---|---|---|---|---|
| **Pipeline de dados** | 15 | Ausentes, categóricas e escala tratados com justificativa; pré-processador ajustado só no treino | Funciona, mas alguma escolha não é justificada | Vazamento entre conjuntos ou estatísticas calculadas antes da separação |
| **Arquitetura** | 15 | Escolhas fundamentadas; nº de parâmetros explicado; baseline linear comparado | Rede funcional, justificativa superficial | Copiada sem entendimento |
| **Treinamento** | 15 | Laço correto, checkpoint do melhor modelo, early stopping, semente fixa, desbalanceamento tratado | Treina corretamente, sem recursos de controle | Laço com erro (ex.: sem `zero_grad`) ou não reprodutível |
| **Validação e diagnóstico** | 15 | Curvas analisadas; limiar escolhido na validação; overfitting identificado e tratado | Curvas apresentadas sem análise | Limiar ou hiperparâmetros ajustados no teste |
| **Avaliação no teste** | 15 | Teste usado uma única vez; métricas comparadas à linha de base trivial; precisão/revocação interpretadas | Métricas corretas com análise rasa | Só acurácia, sem linha de base |
| **Deploy** | 15 | Serviço funcionando, os três artefatos versionados, paridade de pré-processamento verificada, health check | API funciona, sem teste de integração | Não roda, ou pré-processador não acompanha o modelo |
| **Comunicação** | 10 | Relatório claro, tabela de experimentos, discussão ética concreta, conclusões honestas | Relatório completo mas confuso | Incompleto ou sem evidências |

**Observação sobre honestidade experimental:** relatar um experimento que
*piorou* o resultado, com a análise do motivo, vale mais do que apresentar
apenas o melhor número. Reportar métricas obtidas ajustando hiperparâmetros ou
limiar no conjunto de teste zera o critério "Avaliação no teste".

**Observação sobre a acurácia:** um relatório que apresenta 80% de acurácia sem
mencionar que o modelo trivial faz 76,4% está incompleto, mesmo que todo o
resto esteja correto.
