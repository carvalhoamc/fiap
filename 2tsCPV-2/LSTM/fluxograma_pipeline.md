# Fluxograma do Pipeline — `analise_sentimentos_SAC.ipynb`

Fluxograma detalhado de cada etapa do pipeline de análise de sentimentos com PyTorch + Bi-LSTM.

---

## 1. Visão Geral do Pipeline (alto nível)

```mermaid
flowchart TD
    A([Início]) --> B[1. Imports e Configuração]
    B --> C[2. Carregar Dataset CSV]
    C --> D[3. Análise Exploratória - EDA]
    D --> E[4. Pré-processamento de Texto]
    E --> F[5. Vocabulário e Sequências]
    F --> G[6. Dataset e DataLoader PyTorch]
    G --> H[7. Definir Modelo Bi-LSTM]
    H --> I[8. Treinamento]
    I --> J[9. Avaliação no Teste]
    J --> K[10. Inferência - Triagem SAC]
    K --> L[11. Conclusão e Benefícios]
    L --> M([Fim: Modelo pronto para produção])

    style A fill:#2c3e50,color:#fff
    style M fill:#27ae60,color:#fff
    style I fill:#e74c3c,color:#fff
    style H fill:#9b59b6,color:#fff
```

---

## 2. Detalhamento — Configuração e Carga de Dados

```mermaid
flowchart TD
    A([Início]) --> B[Importar torch, pandas,<br/>sklearn, matplotlib, seaborn]
    B --> C{GPU disponível?<br/>torch.cuda.is_available}
    C -->|Sim| D[device = cuda]
    C -->|Não| E[device = cpu<br/>build atual: torch +cpu]
    D --> F[Ler reclamacoes_clientes.csv<br/>encoding utf-8-sig]
    E --> F
    F --> G[DataFrame com 1000 linhas<br/>colunas: texto, sentimento, label]
    G --> H[sentimento derivado do label<br/>+ coluna num_palavras]
    H --> I[value_counts por sentimento<br/>400 Neg / 300 Neu / 300 Pos]
    I --> J([Dados carregados])

    style A fill:#2c3e50,color:#fff
    style J fill:#27ae60,color:#fff
    style C fill:#f39c12,color:#000
```

---

## 3. Detalhamento — EDA (Análise Exploratória)

```mermaid
flowchart LR
    A[DataFrame] --> B[Gráfico 1:<br/>Barras de distribuição<br/>das 3 classes]
    A --> C[Gráfico 2:<br/>Boxplot de palavras<br/>por sentimento]
    B --> E[describe num_palavras<br/>média, min, max, quartis]
    C --> E
    E --> F([Insights do dataset])

    style A fill:#3498db,color:#fff
    style F fill:#27ae60,color:#fff
```

---

## 4. Detalhamento — Pré-processamento de Texto

```mermaid
flowchart TD
    A[Texto bruto<br/>'Produto chegou DANIFICADO!!!'] --> B[lower<br/>minúsculas]
    B --> C[regex: remover pontuação<br/>e números, manter acentos]
    C --> D[regex: normalizar<br/>espaços múltiplos]
    D --> E[texto_limpo<br/>'produto chegou danificado']
    E --> F[Aplicar a todas as<br/>1000 linhas do DataFrame]
    F --> G([coluna texto_limpo criada])

    style A fill:#e67e22,color:#fff
    style G fill:#27ae60,color:#fff
```

---

## 5. Detalhamento — Vocabulário e Tokenização

```mermaid
flowchart TD
    A[Todos os textos limpos] --> B[Counter:<br/>contar frequência<br/>de cada palavra]
    B --> C{"freq >= min_freq?<br/>min_freq = 1"}
    C -->|Sim| D["Adicionar ao vocabulário<br/>palavra -> índice inteiro"]
    C -->|Não| E[Descartar palavra rara]
    D --> F[vocab = PAD:0, UNK:1, ...]
    E --> F
    F --> G[MAX_LEN = 20 tokens<br/>valor fixo]

    G --> H[Para cada texto:<br/>texto_para_sequencia]
    H --> I[Split em tokens<br/>truncar em MAX_LEN]
    I --> J{Palavra no vocab?}
    J -->|Sim| K[usar índice da palavra]
    J -->|Não| L[usar índice UNK = 1]
    K --> M[Padding com 0 PAD<br/>até MAX_LEN]
    L --> M
    M --> N([coluna sequencia:<br/>lista de inteiros])

    style A fill:#e67e22,color:#fff
    style N fill:#27ae60,color:#fff
    style C fill:#f39c12,color:#000
    style J fill:#f39c12,color:#000
```

---

## 6. Detalhamento — Dataset e DataLoader

```mermaid
flowchart TD
    A[X = sequências<br/>y = labels] --> B["train_test_split<br/>80% treino / 20% teste<br/>stratify = y"]
    B --> C[X_train, y_train<br/>800 exemplos]
    B --> D[X_test, y_test<br/>200 exemplos]
    C --> E[ReclamacoesDataset treino<br/>tensores torch.long]
    D --> F[ReclamacoesDataset teste]
    E --> G[train_loader<br/>batch_size=16, shuffle=True]
    F --> H[test_loader<br/>batch_size=16]
    G --> I([Batches prontos<br/>shape: 16 x MAX_LEN])
    H --> I

    style A fill:#3498db,color:#fff
    style I fill:#27ae60,color:#fff
```

---

## 7. Detalhamento — Arquitetura do Modelo Bi-LSTM

```mermaid
flowchart TD
    A[Entrada: batch x MAX_LEN<br/>índices de palavras] --> B[Embedding Layer<br/>vocab_size x 64<br/>padding_idx=0]
    B --> C[Dropout 0.4]
    C --> D[batch x MAX_LEN x 64<br/>vetores densos]
    D --> E[Bi-LSTM<br/>2 camadas, hidden=128<br/>bidirectional=True]
    E --> F[concat das 2 direções<br/>hidden -2 e hidden -1<br/>batch x 256]
    F --> G[Dropout 0.4]
    G --> H["Linear<br/>256 -> 3"]
    H --> I[Logits<br/>batch x 3]
    I --> J([Neg / Neu / Pos])

    style A fill:#9b59b6,color:#fff
    style E fill:#8e44ad,color:#fff
    style J fill:#27ae60,color:#fff
```

---

## 8. Detalhamento — Loop de Treinamento

```mermaid
flowchart TD
    A([Início do treino]) --> B[CrossEntropyLoss<br/>Adam lr=0.001, weight_decay=1e-4<br/>ReduceLROnPlateau]
    B --> C{Para cada época<br/>1 a 40}
    C --> D[model.train]
    D --> E{Para cada batch<br/>no train_loader}
    E --> F[Mover batch para device<br/>cpu / gpu]
    F --> G[optimizer.zero_grad]
    G --> H[forward: outputs = model seqs]
    H --> I[loss = criterion outputs, labels]
    I --> J[loss.backward<br/>backpropagation]
    J --> K[clip_grad_norm 1.0<br/>evita explosão de gradiente]
    K --> L[optimizer.step<br/>atualiza pesos]
    L --> M[Acumular loss e acertos]
    M --> E
    E -->|fim dos batches| N[Calcular loss e acc da época]
    N --> O[scheduler.step loss<br/>ajusta learning rate]
    O --> P[Salvar no histórico]
    P --> C
    C -->|fim das épocas| Q([Modelo treinado])

    style A fill:#e74c3c,color:#fff
    style Q fill:#27ae60,color:#fff
    style C fill:#f39c12,color:#000
    style E fill:#f39c12,color:#000
    style J fill:#c0392b,color:#fff
```

---

## 9. Detalhamento — Avaliação

```mermaid
flowchart TD
    A([Modelo treinado]) --> B[model.eval<br/>desativa dropout]
    B --> C[torch.no_grad<br/>sem cálculo de gradiente]
    C --> D{Para cada batch<br/>no test_loader}
    D --> E[forward: outputs]
    E --> F["argmax -> predição"]
    F --> G[Acumular preds e labels reais]
    G --> D
    D -->|fim| H[classification_report<br/>precision, recall, f1]
    H --> I[confusion_matrix]
    I --> J[Heatmap seaborn<br/>real vs predito]
    J --> K([Métricas de qualidade])

    style A fill:#1abc9c,color:#fff
    style K fill:#27ae60,color:#fff
    style D fill:#f39c12,color:#000
```

---

## 10. Detalhamento — Inferência (Triagem SAC)

```mermaid
flowchart TD
    A[Nova reclamação<br/>texto cru] --> B[preprocessar<br/>limpeza]
    B --> C[texto_para_sequencia<br/>tokenizar + padding]
    C --> D[tensor para device]
    D --> E[model forward<br/>sem gradiente]
    E --> F["softmax<br/>-> probabilidades"]
    F --> G["argmax<br/>-> classe predita"]
    G --> H{Qual sentimento?}
    H -->|Negativo| I[🔴 PRIORIDADE ALTA<br/>fila urgente]
    H -->|Neutro| J[🟡 Fila normal]
    H -->|Positivo| K[🟢 Registrar elogio]
    I --> L([Roteamento automático])
    J --> L
    K --> L

    style A fill:#e67e22,color:#fff
    style L fill:#27ae60,color:#fff
    style H fill:#f39c12,color:#000
    style I fill:#e74c3c,color:#fff
```

---

## 11. Conclusão e Benefícios Corporativos

```mermaid
flowchart TD
    A([Pipeline completo]) --> B[Triagem automática 24/7<br/>sem intervenção humana]
    A --> C[Priorização inteligente<br/>casos críticos primeiro]
    A --> D[Redução de SLA<br/>resposta mais rápida]
    A --> E[Análise de tendências<br/>satisfação ao longo do tempo]
    B --> F([Próximos passos:<br/>BERTimbau, API REST,<br/>monitorar data drift])
    C --> F
    D --> F
    E --> F

    style A fill:#3498db,color:#fff
    style F fill:#27ae60,color:#fff
```

---

## 12. Fluxo de Dados End-to-End (resumo visual)

```mermaid
flowchart LR
    A[Texto PT-BR] --> B[Limpeza]
    B --> C[Tokens]
    C --> D[Índices + Padding]
    D --> E[Embedding 64d]
    E --> F[Bi-LSTM 128 x2 = 256]
    F --> G[Linear 3]
    G --> H[Softmax]
    H --> I[Sentimento + Confiança]

    style A fill:#e67e22,color:#fff
    style E fill:#9b59b6,color:#fff
    style F fill:#8e44ad,color:#fff
    style I fill:#27ae60,color:#fff
```
