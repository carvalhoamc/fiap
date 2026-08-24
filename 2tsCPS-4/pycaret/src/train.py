"""
train.py — ETAPAS 4 e 5: ajuste fino, validação e escolha do limiar
---------------------------------------------------------------------

Na aula de MLP, a ETAPA 4 era o laço de treino: forward, perda, zero_grad,
backward, step — e a ETAPA 5 era ler a curva de perda para diagnosticar
underfitting, overfitting e taxa de aprendizado.

Aqui esse laço não existe: `create_model` e `tune_model` fazem o `.fit()` por
você. O que NÃO desaparece — e é o motivo de este arquivo existir — é a etapa 5:

    1. Escolher o modelo com validação cruzada (nunca com o teste).
    2. Ajustar hiperparâmetros com validação cruzada (nunca com o teste).
    3. Verificar se o ajuste realmente melhorou (`choose_better=True`).
    4. Escolher o LIMIAR de decisão na VALIDAÇÃO.
    5. Salvar o modelo e o limiar juntos.

O passo 4 é o que o AutoML mais deixa de lado, e o mais caro de errar em
produção: `predict_model` corta em 0,5 sem perguntar nada a você, e 0,5 não tem
nada de especial num problema com 24% de positivos.

`choose_better=True` MERECE UM PARÁGRAFO
----------------------------------------
Busca de hiperparâmetros costuma PIORAR o resultado quando o modelo padrão já
está bom — a busca aleatória testa 20 combinações e escolhe a melhor DELAS, que
pode ser pior que a de fábrica. Com `choose_better=True`, o PyCaret compara o
modelo ajustado com o original e devolve o melhor dos dois. Sem isso, é comum
"ajustar" um modelo e publicar uma versão pior sem perceber.

Uso:
    python src/train.py                     # compara, ajusta e escolhe o limiar
    python src/train.py --rapido            # demonstração em sala
    python src/train.py --modelo lightgbm   # pula o compare_models
    python src/train.py --blend             # combina os 3 melhores (voto suave)
    python src/train.py --calibrar          # calibra as probabilidades
    python src/train.py --balancear         # liga o SMOTE (exercício 2.4)
"""

import argparse
import time

import numpy as np

from config import ALVO_BINARIO, CFG, OUT_DIR
from comparar import modelos_incluidos
from data import obter_particoes
from setup_experimento import criar_experimento
from utils import (imprimir_metricas, melhor_limiar, metricas_binarias,
                   plotar_calibracao, salvar_json, silenciar_bibliotecas)

# Nome do arquivo do modelo treinado (save_model acrescenta .pkl).
ARQ_MODELO = "modelo_treinado"


def _formatar_parametros(parametros: dict, largura: int = 96) -> str:
    """Deixa o dicionário de hiperparâmetros legível numa linha de terminal.

    O PyCaret prefixa tudo com `actual_estimator__` porque o que ele ajusta é
    um passo de um Pipeline, não o estimador solto. O prefixo é ruído para
    quem lê; o comportamento é idêntico.
    """
    partes = [f"{chave.split('__')[-1]}={valor}"
              for chave, valor in sorted(parametros.items())]
    texto = ", ".join(partes)
    return texto if len(texto) <= largura else texto[:largura - 3] + "..."


def probabilidades_validacao(exp, modelo) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (y_verdadeiro, P(classe 1)) no conjunto de VALIDAÇÃO.

    `exp.predict_model(modelo)` sem o argumento `data` usa o hold-out do setup
    — que, pela forma como montamos o experimento, é exatamente a nossa
    validação. Com `raw_score=True` vêm as duas probabilidades; queremos a da
    classe positiva.

    Nunca chame isto com o conjunto de teste. Toda decisão tomada a partir
    destes números (limiar, calibração, escolha entre dois modelos) transforma
    o conjunto usado em validação, por definição.
    """
    # round=6: por padrão o predict_model arredonda a probabilidade para 4
    # casas. Irrelevante para um limiar em 0,41; relevante se a sua política
    # cortar em 0,9999. Em deploy/api.py usamos predict_proba, sem arredondar.
    previsoes = exp.predict_model(modelo, raw_score=True, round=6, verbose=False)
    y = previsoes[ALVO_BINARIO].to_numpy().astype(int)
    coluna = ("prediction_score_1" if "prediction_score_1" in previsoes.columns
              else "prediction_score")
    return y, previsoes[coluna].to_numpy(dtype=float)


def treinar(args) -> dict:
    silenciar_bibliotecas()
    t0 = time.time()

    # --- dados -------------------------------------------------------------
    df_treino, df_val, df_teste = obter_particoes(CFG, rapido=args.rapido)
    print(f"Treino {df_treino.shape} | Validação {df_val.shape} | "
          f"Teste {df_teste.shape} (guardado até evaluate.py)\n")

    exp = criar_experimento(df_treino, df_val, balancear=args.balancear)
    if args.balancear:
        print("[setup] fix_imbalance=True — SMOTE ativo no conjunto de treino.\n")

    # --- ETAPA 4a: escolher a família de modelo ---------------------------
    ranking = None
    if args.modelo:
        print(f"Criando '{args.modelo}' direto (compare_models pulado)...")
        # Para o lightgbm passamos a INSTÂNCIA em vez da sigla, só para poder
        # incluir verbose=-1 e calar o log em C++ (ver comparar.modelos_incluidos).
        pedido = args.modelo
        if pedido == "lightgbm":
            from lightgbm import LGBMClassifier
            pedido = LGBMClassifier(random_state=CFG.semente, n_jobs=CFG.n_jobs,
                                    verbose=-1)
        base = exp.create_model(pedido, verbose=False)
        topo = [base]
        cv_base = exp.pull()
    else:
        print(f"compare_models: {len(CFG.modelos_comparados)} famílias, "
              f"{CFG.fold} dobras, ordenado por {CFG.metrica_ordenacao}...\n")
        topo = exp.compare_models(include=modelos_incluidos(CFG),
                                  sort=CFG.metrica_ordenacao,
                                  n_select=CFG.n_melhores, verbose=False)
        topo = topo if isinstance(topo, list) else [topo]
        ranking = exp.pull()
        print("=" * 110)
        print(f"1) RANKING COMPLETO — {len(ranking)} modelos x {CFG.fold} dobras")
        print("=" * 110)
        print(ranking.to_string())
        print("=" * 110)
        base = topo[0]
        cv_base = None
    print(f"\nModelo escolhido: {type(base).__name__}")
    print(f"Os {len(topo)} melhores guardados para o --blend: "
          f"{[type(m).__name__ for m in topo]}\n")

    # Hiperparâmetros DE FÁBRICA, antes de qualquer busca. Guardar isto é o que
    # torna a próxima seção legível: sem o "antes", o "depois" não diz nada.
    parametros_antes = dict(base.get_params())

    # --- ETAPA 4b: ajuste fino de hiperparâmetros -------------------------
    # Busca aleatória de `n_iter` combinações, avaliada pelas MESMAS 5 dobras
    # do compare_models. choose_better=True devolve o original se a busca piorou.
    print("=" * 110)
    print(f"2) AJUSTE FINO (tune_model) — {CFG.n_iter_tune} combinações "
          f"sorteadas, otimizando {CFG.metrica_tune}")
    print("=" * 110)
    print(f"São {CFG.n_iter_tune} x {CFG.fold} = {CFG.n_iter_tune * CFG.fold} "
          f"treinos. Isto costuma ser a parte mais cara do pipeline.\n")

    modelo, buscador = exp.tune_model(base, n_iter=CFG.n_iter_tune,
                                      optimize=CFG.metrica_tune,
                                      choose_better=True, return_tuner=True,
                                      verbose=False)
    cv_ajustado = exp.pull()

    # (i) as dobras do modelo final, uma a uma. O DESVIO entre dobras é a
    #     informação que a média esconde: diferença de 0,002 entre dois modelos
    #     com desvio de 0,006 entre dobras não é diferença nenhuma.
    print("Desempenho do modelo final, dobra a dobra:")
    print(cv_ajustado.to_string())

    # (ii) o interior da busca: quais combinações foram testadas e como se
    #      saíram. É a evidência de que o ajuste foi feito na validação cruzada
    #      e não no teste.
    if buscador is not None and hasattr(buscador, "cv_results_"):
        import pandas as pd
        resultados = pd.DataFrame(buscador.cv_results_).sort_values("rank_test_score")
        print(f"\nAs 5 melhores das {len(resultados)} combinações sorteadas "
              f"({CFG.metrica_tune} médio nas {CFG.fold} dobras):\n")
        print(f"  {'#':>2}  {'AUC':>7}  {'desvio':>7}   hiperparâmetros")
        for _, linha in resultados.head(5).iterrows():
            print(f"  {int(linha['rank_test_score']):>2}  "
                  f"{linha['mean_test_score']:.4f}  {linha['std_test_score']:.4f}   "
                  f"{_formatar_parametros(linha['params'])}")
        print(f"\n  pior das {len(resultados)}: "
              f"{resultados['mean_test_score'].min():.4f}  |  "
              f"melhor: {buscador.best_score_:.4f}  |  "
              f"amplitude da busca: "
              f"{buscador.best_score_ - resultados['mean_test_score'].min():.4f}")

    # (iii) o veredito do choose_better, e o que exatamente mudou.
    mudou = {k: (parametros_antes.get(k), v) for k, v in modelo.get_params().items()
             if k in parametros_antes and parametros_antes[k] != v}
    auc_ajustado = float(cv_ajustado.loc["Mean", "AUC"]) if "Mean" in cv_ajustado.index \
        else float(cv_ajustado["AUC"].iloc[-1])
    print(f"\nAUC (validação cruzada) do modelo publicado: {auc_ajustado:.4f}")
    if mudou:
        print("A busca VENCEU o modelo de fábrica. Hiperparâmetros alterados:")
        for chave, (antes, depois) in sorted(mudou.items()):
            print(f"  {chave:<24} {str(antes):>12}  ->  {depois}")
    else:
        print("O modelo DE FÁBRICA venceu a busca — `choose_better=True` o "
              "manteve.\n"
              "Isso é comum e não é fracasso: a busca aleatória escolhe a melhor\n"
              "de 20 combinações sorteadas, que pode ser pior que o padrão da\n"
              "biblioteca. Sem choose_better, você publicaria a versão pior.")
    print("=" * 110 + "\n")

    metricas_cv = ranking if ranking is not None else cv_base

    # --- ETAPA 4c (opcional): combinar modelos ----------------------------
    if args.blend and len(topo) > 1:
        print(f"blend_models: voto suave entre os {len(topo)} melhores...")
        modelo = exp.blend_models(topo, method="soft", optimize=CFG.metrica_tune,
                                  choose_better=True, verbose=False)
        print(f"Resultado da combinação: {type(modelo).__name__}\n")

    # --- ETAPA 4d (opcional): calibrar as probabilidades ------------------
    if args.calibrar:
        print("calibrate_model: regressão isotônica sobre as dobras...")
        modelo = exp.calibrate_model(modelo, method="isotonic", verbose=False)
        print("(o AUC quase não muda: calibrar reescala as probabilidades sem "
              "mexer na ordem)\n")

    # --- ETAPA 5: validação e escolha do limiar ---------------------------
    y_val, p_val = probabilidades_validacao(exp, modelo)

    m_padrao = metricas_binarias(y_val, p_val, CFG.limiar_padrao)
    limiar, f1_limiar = melhor_limiar(y_val, p_val, criterio="f1")
    m_limiar = metricas_binarias(y_val, p_val, limiar)

    print("=" * 72)
    print(f"VALIDAÇÃO — {len(y_val)} pessoas (nunca usada para ajustar pesos)")
    print("=" * 72)
    print(f"AUC-ROC {m_padrao['auc_roc']:.4f} | precisão média "
          f"{m_padrao['auc_pr']:.4f} (linha de base {y_val.mean():.4f})")
    imprimir_metricas(f"Limiar {CFG.limiar_padrao:.2f} (o padrão do predict_model):",
                      m_padrao)
    imprimir_metricas(f"Limiar {limiar:.2f} (ótimo para F1 NA VALIDAÇÃO):", m_limiar)
    print(f"\nGanho de F1 ao escolher o limiar: "
          f"{f1_limiar - m_padrao['f1']:+.4f}")
    print("O AUC é IDÊNTICO nos dois casos — o limiar não muda o modelo,\n"
          "apenas onde você corta a mesma lista ordenada.")
    print("=" * 72)

    plotar_calibracao(y_val, p_val, OUT_DIR / CFG.arq_calibracao)

    # --- salvar modelo + limiar juntos ------------------------------------
    # O limiar É parte do modelo publicado. Salvar um sem o outro é o mesmo
    # erro de publicar pesos sem o pré-processador na aula de MLP.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp.save_model(modelo, str(OUT_DIR / ARQ_MODELO), verbose=False)
    print(f"\n[ok] Modelo salvo em {OUT_DIR / (ARQ_MODELO + '.pkl')}")

    historico = {
        "modelo": type(modelo).__name__,
        "modelo_base": type(base).__name__,
        "parametros": {k: str(v) for k, v in modelo.get_params().items()
                       if not k.startswith("estimator")},
        "limiar": limiar,
        "auc_validacao": m_padrao["auc_roc"],
        "metricas_validacao_limiar_padrao": m_padrao,
        "metricas_validacao_limiar_escolhido": m_limiar,
        "validacao_cruzada": metricas_cv.head(10).to_dict(orient="records"),
        "blend": args.blend,
        "calibrado": args.calibrar,
        "balanceado": args.balancear,
        "rapido": args.rapido,
        "segundos": round(time.time() - t0, 1),
        "config": CFG.to_dict(),
    }
    salvar_json(historico, OUT_DIR / CFG.arq_historico)
    print(f"[ok] Histórico salvo em {OUT_DIR / CFG.arq_historico}")
    print(f"Tempo total: {historico['segundos']:.1f} s")
    print("\nPróximo passo:  python src/evaluate.py")
    return historico


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ajuste fino e escolha do limiar")
    p.add_argument("--rapido", action="store_true",
                   help="subconjunto pequeno, para demonstração em aula")
    p.add_argument("--modelo", default=None,
                   help="sigla do PyCaret (lr, rf, lightgbm...); pula o compare_models")
    p.add_argument("--blend", action="store_true",
                   help="combina os 3 melhores modelos por voto suave")
    p.add_argument("--calibrar", action="store_true",
                   help="calibra as probabilidades (regressão isotônica)")
    p.add_argument("--balancear", action="store_true",
                   help="liga o SMOTE no setup (exercício 2.4)")
    treinar(p.parse_args())
