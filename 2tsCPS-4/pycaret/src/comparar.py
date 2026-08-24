"""
comparar.py — ETAPA 3: escolher o modelo (o que o AutoML realmente automatiza)
--------------------------------------------------------------------------------

Na aula de MLP, a ETAPA 3 era "arquitetura": você decidia quantas camadas, quantos
neurônios, qual ativação — e defendia cada escolha. Aqui a etapa tem outro nome e
outra natureza: `compare_models()` treina uma dúzia de famílias diferentes, com
validação cruzada de 5 dobras, e devolve a tabela ordenada.

O que isso muda de verdade:

  * A pergunta deixa de ser "essa arquitetura está boa?" e passa a ser
    "essa FAMÍLIA de modelo é a certa para este dado?".
  * O custo de testar uma hipótese cai de horas para minutos. Isso é
    genuinamente valioso: em dados tabulares, a escolha da família importa
    mais que o ajuste fino dentro dela.
  * Em compensação, fica MUITO barato produzir um número sem entender de onde
    ele veio — e é exatamente isso que a rubrica desta disciplina não aceita.

O 'dummy' ENTRA DE PROPÓSITO
----------------------------
`DummyClassifier` chuta sempre a classe majoritária: zero aprendizado, três
linhas de código, ~76% de acurácia neste dataset. Ele fica na tabela como régua.
Um relatório que celebra 87% de acurácia sem dizer que o trivial faz 76% está
incompleto, mesmo que todo o resto esteja certo.

POR QUE ORDENAR POR AUC E NÃO POR ACURÁCIA
------------------------------------------
Com 24% de positivos, acurácia recompensa quem ignora a classe minoritária. O
AUC mede a qualidade do ORDENAMENTO e não depende do limiar — é a métrica certa
para SELECIONAR modelo quando a decisão final ainda vai ser calibrada. Rode com
`--metrica Accuracy` e veja o dummy subir posições na tabela.

Uso:
    python src/comparar.py                 # ~4 min: 13 modelos, 5 dobras
    python src/comparar.py --rapido        # ~40 s, para demonstrar em sala
    python src/comparar.py --metrica F1
    python src/comparar.py --todos         # todos os modelos do PyCaret
"""

import argparse
import time

from config import CFG, OUT_DIR
from data import obter_particoes
from setup_experimento import criar_experimento
from utils import plotar_ranking, salvar_json, silenciar_bibliotecas

# Colunas da tabela do PyCaret que interessam nesta aula, na ordem de leitura.
COLUNAS_TABELA = ["Model", "AUC", "Accuracy", "Prec.", "Recall", "F1", "TT (Sec)"]


def modelos_incluidos(cfg=CFG) -> list:
    """Traduz a lista de siglas do config.py no que o `include=` espera.

    `include` aceita SIGLAS ('lr', 'rf', 'xgboost') e também INSTÂNCIAS de
    estimadores compatíveis com o scikit-learn. É por aí que você coloca um
    modelo seu na competição — inclusive um que não exista no PyCaret.

    Aqui usamos a instância só para calar o LightGBM: ele imprime dezenas de
    linhas de log por dobra, vindas do código C++, e como as dobras rodam em
    processos paralelos nem `register_logger` alcança. `verbose=-1` viaja junto
    com o estimador quando ele é enviado ao processo trabalhador.
    """
    incluidos = []
    for sigla in cfg.modelos_comparados:
        if sigla == "lightgbm":
            from lightgbm import LGBMClassifier
            incluidos.append(LGBMClassifier(random_state=cfg.semente,
                                            n_jobs=cfg.n_jobs, verbose=-1))
        else:
            incluidos.append(sigla)
    return incluidos


def comparar(rapido: bool = False, metrica: str | None = None,
             todos: bool = False, balancear: bool = False):
    """Roda o compare_models e salva a tabela + o gráfico do ranking."""
    silenciar_bibliotecas()
    metrica = metrica or CFG.metrica_ordenacao

    df_treino, df_val, _ = obter_particoes(CFG, rapido=rapido)
    print(f"Treino {df_treino.shape} | Validação {df_val.shape} "
          f"| Teste: guardado, não entra aqui\n")

    exp = criar_experimento(df_treino, df_val, balancear=balancear)

    incluir = None if todos else modelos_incluidos(CFG)
    print(f"Comparando {'todos os modelos disponíveis' if todos else f'{len(incluir)} modelos'} "
          f"por {metrica}, com {CFG.fold} dobras estratificadas...")
    print("(cada linha da tabela é a MÉDIA das 5 dobras; o desvio entre dobras "
          "aparece\n com exp.pull() logo após o compare_models)\n")

    t0 = time.time()
    melhores = exp.compare_models(
        include=incluir,
        sort=metrica,
        n_select=CFG.n_melhores,
        # cross_validation=True é o padrão: cada modelo é treinado 5 vezes.
        # O hold-out (nossa validação) NÃO é usado aqui — ele fica reservado
        # para a escolha do limiar em train.py.
        verbose=False,
    )
    segundos = time.time() - t0

    tabela = exp.pull()
    colunas = [c for c in COLUNAS_TABELA if c in tabela.columns]

    # Tabela COMPLETA: todas as métricas que o PyCaret mede, para todos os
    # modelos. Kappa e MCC entram aqui de graça e são úteis justamente em
    # problema desbalanceado — os dois descontam o acerto que se obteria por
    # acaso, coisa que a acurácia não faz.
    print("=" * 110)
    print(f"RANKING COMPLETO POR {metrica} — validação cruzada de {CFG.fold} "
          f"dobras estratificadas, no TREINO")
    print("=" * 110)
    print(tabela.to_string())
    print("=" * 110)
    print(f"Tempo total: {segundos:.1f} s "
          f"({len(tabela)} modelos x {CFG.fold} dobras = "
          f"{len(tabela) * CFG.fold} treinos)\n")

    print("Resumo das colunas que mais importam nesta aula:")
    print(tabela[colunas].to_string())
    print()

    # --- a leitura que interessa ------------------------------------------
    linha_dummy = tabela[tabela["Model"].str.contains("Dummy", case=False, na=False)]
    if not linha_dummy.empty:
        acc_dummy = float(linha_dummy["Accuracy"].iloc[0])
        topo = tabela.iloc[0]
        print(f"Linha de base trivial (Dummy): acurácia {acc_dummy:.2%}, AUC "
              f"{float(linha_dummy['AUC'].iloc[0]):.4f}")
        print(f"Melhor modelo ({topo['Model']}): acurácia "
              f"{float(topo['Accuracy']):.2%}, AUC {float(topo['AUC']):.4f}")
        print(f"Ganho real sobre o trivial: "
              f"{(float(topo['Accuracy']) - acc_dummy) * 100:+.2f} pontos de acurácia\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(OUT_DIR / CFG.arq_comparacao, index=False, encoding="utf-8")
    print(f"[ok] Tabela salva em {OUT_DIR / CFG.arq_comparacao}")
    plotar_ranking(tabela.reset_index(), OUT_DIR / CFG.arq_ranking, metrica)

    salvar_json({
        "metrica_ordenacao": metrica,
        "fold": CFG.fold,
        "rapido": rapido,
        "balancear": balancear,
        "segundos": round(segundos, 1),
        "ranking": tabela[colunas].to_dict(orient="records"),
        "melhores": [type(m).__name__ for m in
                     (melhores if isinstance(melhores, list) else [melhores])],
    }, OUT_DIR / "comparacao.json")

    return exp, melhores, tabela


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="compare_models do PyCaret")
    p.add_argument("--rapido", action="store_true",
                   help="subconjunto pequeno, para demonstração em aula")
    p.add_argument("--metrica", default=None,
                   help="AUC (padrão), Accuracy, F1, Recall, Prec.")
    p.add_argument("--todos", action="store_true",
                   help="compara TODOS os modelos do PyCaret (bem mais lento)")
    p.add_argument("--balancear", action="store_true",
                   help="liga o SMOTE no setup (exercício 2.4)")
    args = p.parse_args()
    comparar(args.rapido, args.metrica, args.todos, args.balancear)
