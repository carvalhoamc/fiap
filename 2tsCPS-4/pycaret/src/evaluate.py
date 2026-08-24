"""
evaluate.py — ETAPA 6: teste final
-----------------------------------

Aqui o conjunto de TESTE é usado pela primeira e única vez. Ele responde à
pergunta que interessa: "quanto o modelo acerta em dados que ninguém — nem a
validação cruzada, nem o tune_model, nem eu escolhendo o limiar — jamais viu?"

Acurácia sozinha engana, e neste dataset engana muito: um modelo que responde
"<=50K" para todo mundo acerta 76,4%. Por isso reportamos:

    Matriz de confusão : VN, FP, FN, VP — a estrutura do erro.
    Precisão   : dos que EU disse que ganham mais de 50k, quantos ganham mesmo?
    Revocação  : dos que GANHAM mais de 50k, quantos eu encontrei?
    F1         : média harmônica das duas.
    AUC-ROC    : qualidade do ordenamento, independente do limiar.
    Precisão média (AUC-PR): idem, mas sensível à raridade da classe positiva.

E, no fim, duas leituras de importância:

  * por PERMUTAÇÃO nas colunas ORIGINAIS (12 campos do cadastro) — funciona
    para qualquer modelo e responde "que campo do formulário importa";
  * a importância NATIVA do modelo (nas 90 colunas transformadas) — só existe
    para modelos que a expõem, e responde "que coluna interna o modelo usou".

As duas discordam com frequência, e entender por quê é metade do exercício.

Uso:
    python src/evaluate.py
    python src/evaluate.py --limiar 0.5     # compara com o limiar padrão
    python src/evaluate.py --rapido
"""

import argparse

import numpy as np
import pandas as pd
from pycaret.classification import load_model, predict_model
from sklearn.inspection import permutation_importance

from config import ALVO_BINARIO, CFG, CLASSES, OUT_DIR
from data import obter_particoes
from train import ARQ_MODELO
from utils import (carregar_json, imprimir_metricas, metricas_binarias,
                   plotar_importancia, plotar_matriz_confusao, plotar_roc_pr,
                   salvar_json, silenciar_bibliotecas)


def carregar_modelo():
    """Recarrega o pipeline completo salvo por train.py.

    Repare no que vem dentro do .pkl: NÃO é só o estimador, é o `Pipeline`
    inteiro — imputação, one-hot e padronização já ajustados no treino, mais o
    modelo. Na aula de MLP eram três arquivos que precisavam viajar juntos
    (pesos, preprocessador.json, metadados.json) e um pré-processador
    reimplementado à mão no servidor. Aqui é um objeto só.

    O preço disso aparece na ETAPA 7: para abrir este arquivo é preciso ter o
    PyCaret e o scikit-learn instalados, nas mesmas versões. Ver export_model.py.
    """
    caminho = OUT_DIR / ARQ_MODELO
    if not (OUT_DIR / f"{ARQ_MODELO}.pkl").exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {caminho}.pkl. "
            "Rode antes: python src/train.py")
    return load_model(str(caminho), verbose=False)


def probabilidades(pipeline, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (y_verdadeiro, P(classe 1)) para qualquer dataframe cru.

    `predict_model` aplica o pipeline inteiro: o dataframe entra com as 12
    colunas do cadastro e sai com as previsões. Nenhuma transformação precisa
    ser repetida à mão — é justamente essa a vantagem de empacotar
    pré-processamento e modelo juntos.
    """
    previsoes = predict_model(pipeline, data=df, raw_score=True, round=6, verbose=False)
    coluna = ("prediction_score_1" if "prediction_score_1" in previsoes.columns
              else "prediction_score")
    return (df[ALVO_BINARIO].to_numpy().astype(int),
            previsoes[coluna].to_numpy(dtype=float))


def importancia_por_permutacao(pipeline, df: pd.DataFrame, repeticoes: int = 5,
                               semente: int = 42):
    """Quanto o AUC cai ao embaralhar UMA coluna do cadastro por vez.

    A ideia independe do tipo de modelo: se destruir a relação entre a coluna e
    o alvo não muda nada, o modelo não estava usando aquela coluna.

    Calculada na VALIDAÇÃO, não no teste: o teste é usado uma vez só, para o
    número final.

    Cuidado ao interpretar: colunas correlacionadas dividem a importância entre
    si (embaralhar uma delas ainda deixa a informação disponível na outra), e
    IMPORTÂNCIA NÃO É CAUSALIDADE. `marital-status` importar muito não quer
    dizer que casar aumente a renda.
    """
    X = df.drop(columns=[ALVO_BINARIO])
    y = df[ALVO_BINARIO].to_numpy().astype(int)
    resultado = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=repeticoes,
        random_state=semente, n_jobs=1)
    return list(X.columns), resultado.importances_mean, resultado.importances_std


def importancia_nativa(pipeline, n: int = 15):
    """Importância que o próprio modelo expõe, nas colunas TRANSFORMADAS.

    Em modelos de árvore é o ganho acumulado nos splits; em modelos lineares,
    o módulo do coeficiente (que só é comparável porque padronizamos as
    colunas). Devolve None quando o modelo não expõe nada — KNN, por exemplo.

    Por que ela discorda da permutação: esta mede o quanto o modelo USOU a
    coluna DURANTE O TREINO; a permutação mede o quanto o desempenho PIORA sem
    ela. Uma coluna redundante pode ter sido muito usada e, ainda assim, não
    fazer falta.
    """
    estimador = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    nomes = getattr(estimador, "feature_names_in_", None)
    if nomes is None:
        return None

    if hasattr(estimador, "feature_importances_"):
        valores = np.asarray(estimador.feature_importances_, dtype=float)
    elif hasattr(estimador, "coef_"):
        valores = np.abs(np.asarray(estimador.coef_, dtype=float)).ravel()
    else:
        return None

    ordem = np.argsort(-valores)[:n]
    return [(str(nomes[i]), float(valores[i])) for i in ordem]


def avaliar(rapido: bool = False, limiar_forcado: float | None = None) -> dict:
    silenciar_bibliotecas()
    pipeline = carregar_modelo()

    historico = {}
    caminho_hist = OUT_DIR / CFG.arq_historico
    if caminho_hist.exists():
        historico = carregar_json(caminho_hist)
    limiar = (limiar_forcado if limiar_forcado is not None
              else historico.get("limiar", CFG.limiar_padrao))

    nome = historico.get("modelo", type(pipeline.steps[-1][1]).__name__)
    print(f"Modelo publicado: {nome}")
    print(f"AUC de validação registrado no treino: "
          f"{historico.get('auc_validacao', float('nan')):.4f}")
    print(f"Limiar de decisão: {limiar:.2f}"
          f"{' (forçado pela linha de comando)' if limiar_forcado is not None else ' (escolhido na validação)'}\n")

    _, df_val, df_teste = obter_particoes(CFG, rapido=rapido)
    y, prob = probabilidades(pipeline, df_teste)

    m = metricas_binarias(y, prob, limiar)
    prevalencia = float(y.mean())

    print("=" * 72)
    print(f"TESTE — {len(y)} pessoas (usado UMA única vez)")
    print("=" * 72)
    print(f"AUC-ROC                {m['auc_roc']:.4f}   (0,5 = chute)")
    print(f"Precisão média (AUC-PR){m['auc_pr']:8.4f}   (linha de base = {prevalencia:.4f})")
    imprimir_metricas(f"No limiar {limiar:.2f}:", m)

    # --- referências honestas de comparação -------------------------------
    print("\nLinhas de base:")
    print(f"  chutar sempre '<=50K'  -> acurácia {1 - prevalencia:.2%}, "
          f"revocação 0,000 (não encontra ninguém)")
    print("  chutar ao acaso        -> AUC 0,500")

    # O mesmo modelo no limiar padrão, para deixar visível que o limiar move
    # precisão e revocação em direções opostas sem mudar o AUC.
    if limiar_forcado is None and abs(limiar - CFG.limiar_padrao) > 1e-9:
        imprimir_metricas(
            f"Para comparação, no limiar padrão {CFG.limiar_padrao:.2f} "
            f"(o que o predict_model usa sozinho):",
            metricas_binarias(y, prob, CFG.limiar_padrao))
        print("\n  Note que o AUC é idêntico nos dois casos: o limiar não muda o\n"
              "  modelo, apenas onde você corta a mesma lista ordenada.")

    plotar_matriz_confusao(m["matriz"], CLASSES, OUT_DIR / CFG.arq_confusao, limiar)
    plotar_roc_pr(y, prob, OUT_DIR / CFG.arq_curvas)

    # --- importância por permutação, na VALIDAÇÃO -------------------------
    print(f"\nImportância por permutação nas colunas do cadastro "
          f"(validação, {len(df_val)} linhas):")
    nomes, quedas, desvios = importancia_por_permutacao(pipeline, df_val)
    ordem = np.argsort(-quedas)
    for i in ordem:
        print(f"  {nomes[i]:<18} queda de AUC {quedas[i]:+.4f}  "
              f"(+/- {desvios[i]:.4f})")
    plotar_importancia(nomes, list(quedas), OUT_DIR / CFG.arq_importancia,
                       n=len(nomes))

    # --- importância nativa do modelo, nas colunas transformadas ----------
    nativa = importancia_nativa(pipeline)
    if nativa:
        print("\nImportância nativa do modelo (top 15 das colunas transformadas):")
        for nome_feature, valor in nativa:
            print(f"  {nome_feature:<38} {valor:.4f}")
        print("  (compare com a lista acima: as duas medem coisas diferentes)")

    resultado = {
        "modelo": nome,
        "limiar": limiar,
        "metricas_teste": m,
        "metricas_teste_limiar_padrao": metricas_binarias(y, prob, CFG.limiar_padrao),
        "prevalencia_teste": prevalencia,
        "importancia_permutacao": {nomes[i]: float(quedas[i]) for i in ordem},
        "importancia_nativa": dict(nativa) if nativa else None,
        "classes": CLASSES,
    }
    salvar_json(resultado, OUT_DIR / CFG.arq_metricas)
    print(f"\n[ok] Métricas salvas em {OUT_DIR / CFG.arq_metricas}")
    return resultado


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Avaliação no conjunto de teste")
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--limiar", type=float, default=None,
                   help="força um limiar específico (padrão: o escolhido na validação)")
    args = p.parse_args()
    avaliar(args.rapido, args.limiar)
