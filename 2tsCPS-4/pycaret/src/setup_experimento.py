"""
setup_experimento.py — ETAPA 2: o pré-processamento inteiro, em uma chamada
-----------------------------------------------------------------------------

Este é o arquivo mais importante da aula, e o mais curto.

Na aula de MLP, o pré-processamento eram 77 linhas de código de uma classe
`Preprocessador` escrita à mão — imputar mediana, criar a categoria
"Desconhecido", montar o one-hot na ordem certa, padronizar com média e desvio
do treino, serializar tudo em JSON — mais 32 linhas reimplementando a MESMA
regra em numpy no servidor. Todo esse trabalho continua acontecendo; o que muda
é que agora quem escreve é o PyCaret, a partir da configuração que VOCÊ declara
aqui, e a segunda implementação deixa de existir.

A função abaixo tem 28 linhas de código. É esse o tamanho da troca.

O QUE O setup() DEVOLVE
-----------------------
Um objeto de experimento que carrega:

  * o `Pipeline` do scikit-learn com todos os transformadores AJUSTADOS
    apenas no conjunto de treino;
  * a partição interna (treino x hold-out) e a estratégia de validação cruzada;
  * a semente, que fixa TODA a aleatoriedade a partir daí.

A partir desse objeto, `compare_models`, `tune_model`, `predict_model` e
`save_model` reaproveitam o mesmo pipeline. É isso que impede o erro clássico
de treinar com uma transformação e prever com outra.

O QUE O PYCARET **NÃO** FAZ POR VOCÊ
-----------------------------------
  * Não decide o que é numérico e o que é categórico com critério de domínio
    (ele chuta pelo dtype, e chuta errado quando a categoria vem como número).
  * Não sabe que `fnlwgt` é peso amostral e precisa ser descartado.
  * Não sabe que o teste é a partição oficial do UCI.
  * Não escolhe o limiar de decisão pelo custo do erro no seu negócio.
  * Não percebe que `sex` e `race` levantam uma questão ética.

Todas essas continuam sendo decisões suas. O AutoML automatizou a digitação,
não o julgamento.

Uso:
    python src/setup_experimento.py
    python src/setup_experimento.py --rapido
"""

import argparse

import pandas as pd
from pycaret.classification import ClassificationExperiment

from config import (ALVO_BINARIO, CFG, COLUNAS_CATEGORICAS, COLUNAS_NUMERICAS)
from data import obter_particoes


def criar_experimento(df_treino: pd.DataFrame, df_val: pd.DataFrame,
                      cfg=CFG, balancear: bool | None = None,
                      verbose: bool = False) -> ClassificationExperiment:
    """Monta o experimento do PyCaret com o pré-processamento desta aula.

    Cada argumento abaixo substitui um pedaço do `Preprocessador` da aula de
    MLP — o comentário ao lado diz qual.

    Repare no par (data, test_data): entregamos TREINO e VALIDAÇÃO. O conjunto
    de TESTE não entra aqui de propósito. Ele não é visto por nenhuma etapa de
    ajuste; só aparece em evaluate.py, uma única vez.
    """
    exp = ClassificationExperiment()
    exp.setup(
        data=df_treino,
        test_data=df_val,          # o "hold-out" do PyCaret é a NOSSA validação
        target=ALVO_BINARIO,

        # --- tipagem das colunas: a decisão de modelagem, não do dtype -----
        numeric_features=list(COLUNAS_NUMERICAS),
        categorical_features=list(COLUNAS_CATEGORICAS),

        # --- imputação (equivale ao `medianas` + `CATEGORIA_AUSENTE`) ------
        # Mediana e não média: robusta a valores extremos. Os ausentes
        # categóricos já viraram "Desconhecido" em data.py.
        imputation_type="simple",
        numeric_imputation="median",
        categorical_imputation="mode",

        # --- categóricas -> números (equivale ao bloco one-hot) ------------
        # ARMADILHA: o padrão é max_encoding_ohe=25. Acima disso o PyCaret
        # troca one-hot por target encoding SEM AVISAR — e native-country tem
        # 42 categorias. Target encoding usa a média do alvo por categoria, o
        # que é legítimo mas é OUTRO modelo de dados, e uma fonte conhecida de
        # vazamento quando mal implementado. Aqui forçamos one-hot em tudo para
        # que a comparação com a aula de MLP seja honesta.
        max_encoding_ohe=cfg.max_encoding_ohe,

        # --- escala (equivale ao `(x - media) / desvio`) -------------------
        # Ajustada SÓ no treino, pelo pipeline. Árvores não precisam disso;
        # regressão logística, KNN e MLP precisam muito.
        normalize=cfg.normalize,
        normalize_method=cfg.normalize_method,

        # --- desbalanceamento ----------------------------------------------
        # SMOTE sintetiza positivos até equilibrar. É o análogo do `pos_weight`
        # da aula de MLP, com uma diferença importante: pos_weight repondera a
        # perda, SMOTE inventa linhas. Ver o exercício 2.4.
        fix_imbalance=cfg.fix_imbalance if balancear is None else balancear,

        # --- validação ------------------------------------------------------
        # 5 dobras estratificadas dentro do TREINO. É o que substitui a curva
        # de validação por época da aula de MLP: em vez de acompanhar uma
        # partição fixa ao longo do tempo, medimos 5 vezes em partições
        # diferentes e olhamos média e desvio.
        fold=cfg.fold,
        fold_strategy=cfg.fold_strategy,

        # --- reprodutibilidade e ambiente -----------------------------------
        session_id=cfg.semente,   # o `definir_semente(42)` da aula anterior
        n_jobs=cfg.n_jobs,
        use_gpu=cfg.use_gpu,
        index=False,
        html=False,               # sem widgets de notebook: rodamos no terminal
        verbose=verbose,
        memory=False,
    )
    return exp


def resumo_do_pipeline(exp: ClassificationExperiment) -> pd.DataFrame:
    """Mostra o que o setup() efetivamente montou.

    Não confie na configuração que você escreveu: confira o que o PyCaret
    entendeu. `exp.pipeline` é um Pipeline do scikit-learn comum — você pode
    (e deve) abri-lo e ler passo a passo.
    """
    return exp.pull()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspeção do setup() do PyCaret")
    p.add_argument("--rapido", action="store_true")
    args = p.parse_args()

    df_treino, df_val, df_teste = obter_particoes(rapido=args.rapido)
    print(f"Treino {df_treino.shape} | Validação {df_val.shape} | "
          f"Teste {df_teste.shape} (guardado)\n")

    exp = criar_experimento(df_treino, df_val)

    print("=" * 72)
    print("O QUE O setup() ENTENDEU")
    print("=" * 72)
    print(resumo_do_pipeline(exp).to_string())

    print("\n" + "=" * 72)
    print("O PIPELINE MONTADO (um sklearn.Pipeline comum)")
    print("=" * 72)
    for nome, passo in exp.pipeline.steps:
        print(f"  {nome:<28} {type(passo).__name__}")

    # A transformação aplicada de fato, para conferir a contagem de features.
    X_treino = exp.get_config("X_train_transformed")
    X_val = exp.get_config("X_test_transformed")
    print(f"\nMatriz de treino    : {X_treino.shape}")
    print(f"Matriz de validação : {X_val.shape}")
    print(f"Primeiras 8 features: {list(X_treino.columns[:8])}")

    print("\nConferência da padronização (deve dar média ~0 e desvio ~1 NO TREINO):")
    for coluna in COLUNAS_NUMERICAS:
        print(f"  {coluna:<16} treino: média {X_treino[coluna].mean():+.3f} "
              f"desvio {X_treino[coluna].std():.3f}   |   "
              f"validação: média {X_val[coluna].mean():+.3f} "
              f"desvio {X_val[coluna].std():.3f}")
    print("  (na validação NÃO dá exatamente 0 e 1 — e isso está certo: as "
          "estatísticas\n   vieram do treino. Se desse exatamente, haveria vazamento.)")
