# Atividades — CNN do dado ao deploy

> Antes de começar, execute o roteiro completo do [README](README.md) pelo menos
> uma vez e guarde os resultados de referência (acurácia de teste, curvas e
> matriz de confusão). Todo exercício abaixo é comparado com **essa linha de
> base**. Um experimento sem baseline não prova nada.

Regra permanente para todos os exercícios: **mexa em uma variável por vez** e
registre o resultado em uma tabela. Mudar três coisas e ver a acurácia subir não
diz qual das três funcionou.

---

## Nível 1 — Compreensão

**1.1 Fórmula do tamanho.** Sem rodar o código, calcule o formato do tensor após
cada operação, para uma entrada `(1, 1, 28, 28)`:

| # | Operação | Saída (C, H, W) |
|---|---|---|
| a | `Conv2d(1, 8, kernel_size=5, padding=0, stride=1)` | ? |
| b | `MaxPool2d(2, 2)` sobre o resultado de (a) | ? |
| c | `Conv2d(8, 16, kernel_size=3, padding=1, stride=2)` sobre (b) | ? |

Depois confirme com `torch.zeros(1,1,28,28)` e as camadas reais. Erros aqui são a
causa nº 1 de `shape mismatch`.

**1.2 Contagem de parâmetros.** Quantos parâmetros treináveis tem uma
`Conv2d(16, 32, kernel_size=3)` **com** e **sem** `bias`? Compare com uma
`Linear` que ligasse `16×14×14` a `32×14×14`. Explique em duas frases o que essa
diferença significa em termos de memória e de risco de overfitting.

**1.3 Leitura de curvas.** Descreva o diagnóstico de cada cenário e a correção
que você aplicaria:
- (a) treino 99 %, validação 82 %, perda de validação subindo há 4 épocas;
- (b) treino 61 %, validação 60 %, ambas as perdas estáveis e altas;
- (c) perda de treino serrilhada, oscilando muito entre lotes.

**1.4 Por que sem softmax?** Explique, com base no código de `model.py` e
`train.py`, por que o `forward` devolve logits. O que acontece numericamente se
você aplicar softmax duas vezes?

---

## Nível 2 — Experimentação controlada

Preencha a tabela para cada item, sempre reexecutando `train.py` e `evaluate.py`:

| Experimento | Val. acc | Teste acc | Parâmetros | Tempo/época | Observação |
|---|---|---|---|---|---|
| Linha de base | | | 98.554 | | |
| ... | | | | | |

**2.1 Sem normalização.** Remova o `transforms.Normalize` das duas
transformações em `data.py`. Treine 5 épocas. O que acontece com a velocidade de
convergência? Por quê?

**2.2 Sem aumento de dados.** Deixe a transformação de treino igual à de
avaliação. Compare as curvas: a distância entre treino e validação aumentou?
Isso é overfitting?

**2.3 Capacidade do modelo.** Em `config.py`, teste `canais_conv = (8, 16, 32)` e
depois `(32, 64, 128)`. Relacione o número de parâmetros, o tempo por época e a
acurácia. Mais parâmetros sempre ajudam?

**2.4 Taxa de aprendizado.** Rode com `--lr 1e-1`, `1e-3` e `1e-5` por 5 épocas.
Descreva o comportamento de cada uma. Uma delas provavelmente não vai aprender
nada — explique o motivo.

**2.5 Dropout.** Compare `dropout = 0.0`, `0.3` e `0.7`. Em qual deles a
diferença entre acurácia de treino e de validação é maior? Isso confirma o papel
regularizador do dropout?

**2.6 Ablação do BatchNorm.** Remova `nn.BatchNorm2d` de `bloco_conv`. Treine com
`--lr 1e-3` e depois com `--lr 1e-2`. O BatchNorm permitiu taxas maiores?

---

## Nível 3 — Implementação

**3.1 Baseline honesto.** Implemente em `model.py` uma classe `MLPSimples`
(achatar 784 → densa 128 → ReLU → densa 10), treine nas mesmas condições e
compare com a CNN: acurácia, número de parâmetros e tempo. **Este é o
experimento que justifica a existência da convolução.**

**3.2 Matriz de confusão interpretada.** A partir de `outputs/matriz_confusao.png`,
identifique os três pares de classes mais confundidos. Para cada par, proponha
uma hipótese sobre a causa e uma intervenção concreta (mais dados? resolução
maior? outra transformação? juntar as classes?).

**3.3 Métrica de topo-2.** Implemente em `evaluate.py` a **acurácia top-2** (o
rótulo verdadeiro está entre as duas maiores probabilidades). Compare com a
top-1. Em que aplicação real a métrica top-2 seria a mais adequada?

**3.4 Visualização dos filtros.** Escreva um script que salve como imagem os 16
filtros 3×3 aprendidos na primeira convolução
(`modelo.extrator[0][0].weight`). Descreva o que eles parecem detectar. Depois
salve os **mapas de ativação** do primeiro bloco para uma imagem de teste.

**3.5 Robustez.** Fotografe (ou desenhe) três peças de roupa, classifique com
`predict.py` e `--inverter`, e explique os erros. O que a distância entre o seu
domínio e o domínio do Fashion-MNIST diz sobre generalização?

---

## Nível 4 — Desafio final (Grupo de 2 Alunos - Entrega dia 24/08/2026)

Escolha **um** dos caminhos:

**Caminho A — Novo conjunto de dados.**
Troque o Fashion-MNIST pelo **CIFAR-10** (`datasets.CIFAR10`), que é colorido e
32×32. Você terá de ajustar `canais = 3`, `tamanho_imagem = 32`, as estatísticas
de normalização e provavelmente a profundidade da rede. Documente a queda de
acurácia e explique por que este problema é mais difícil.

**Caminho B — Transferência de aprendizado.**
Use uma `resnet18` pré-treinada do `torchvision.models`, congele o extrator,
troque a última camada e compare com a CNN treinada do zero, controlando o tempo
de treino. Quando vale a pena treinar do zero?

**Caminho C — Serviço completo.**
Estenda a API com: (i) endpoint de lote (`POST /prever_lote`), (ii) registro em
arquivo de toda predição com confiança abaixo de 0,5, (iii) endpoint `/metricas`
com contagem de requisições e latência média, e (iv) imagem Docker funcionando.

**Entregáveis (todos os caminhos):**

1. Código no repositório, organizado e comentado.
2. Relatório de 3 a 5 páginas: problema, pipeline, tabela de experimentos, curvas, matriz de confusão, análise de erros e conclusão.
3. Demonstração de 5 minutos com o serviço rodando ao vivo.

---

## Rubrica de avaliação (100 pontos)

| Critério | Pts | Excelente | Suficiente | Insuficiente |
|---|---|---|---|---|
| **Pipeline de dados** | 15 | Particionamento correto, sem vazamento, augmentation justificado e aplicado só no treino | Funciona, mas alguma escolha não é justificada | Vazamento entre conjuntos ou augmentation na validação |
| **Arquitetura** | 15 | Escolhas fundamentadas; formatos e nº de parâmetros explicados | Rede funcional, justificativa superficial | Copiada sem entendimento; erros de formato |
| **Treinamento** | 15 | Laço correto, checkpoint do melhor modelo, early stopping, semente fixa | Treina corretamente, sem recursos de controle | Laço com erro (ex.: sem `zero_grad`) ou não reprodutível |
| **Validação e diagnóstico** | 15 | Curvas analisadas, overfitting/underfitting identificado e tratado | Curvas apresentadas sem análise | Ausentes ou lidas incorretamente |
| **Avaliação no teste** | 15 | Teste usado uma única vez; matriz de confusão e métricas por classe interpretadas | Métricas corretas com análise rasa | Teste usado para ajustar hiperparâmetros |
| **Deploy** | 15 | Serviço funcionando, paridade de pré-processamento verificada, health check | API funciona, sem teste de integração | Não roda ou pré-processamento divergente |
| **Comunicação** | 10 | Relatório claro, tabela de experimentos, conclusões honestas | Relatório completo mas confuso | Incompleto ou sem evidências |

**Observação sobre honestidade experimental:** relatar um experimento que
*piorou* o resultado, com a análise do motivo, vale mais do que apresentar
apenas o melhor número. Reportar acurácia obtida ajustando hiperparâmetros no
conjunto de teste zera o critério "Avaliação no teste".
