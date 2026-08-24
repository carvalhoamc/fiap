"""
predict.py — inferência em cadastros novos
-------------------------------------------

A ponte entre "o modelo funciona no notebook" e "o modelo funciona no mundo".

Na aula de MLP, este arquivo carregava um checkpoint, reconstruía o
pré-processador a partir do JSON e montava o tensor à mão — e o comentário
central era o erro clássico dos projetos tabulares: treino e inferência
PRECISAM usar exatamente o mesmo pré-processamento, e em tabelas o desastre é
silencioso (o código roda, devolve uma probabilidade bonita e ERRADA).

Aqui esse risco praticamente desaparece: `predict_model` aplica o mesmo
`Pipeline` que foi ajustado no treino, porque ele está dentro do arquivo. O que
sobra de responsabilidade sua são duas coisas — e as duas aparecem abaixo:

    1. Entregar as colunas com os NOMES certos (o pipeline procura por nome).
    2. Repetir as transformações que ficaram FORA do pipeline (o log1p).

Uso:
    python src/predict.py --exemplo
    python src/predict.py --json pessoa.json
    python src/predict.py --csv pessoas.csv --n 10
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from pycaret.classification import predict_model

from config import (ALVO, CATEGORIA_AUSENTE, CFG, CLASSES,
                    COLUNAS, COLUNAS_CATEGORICAS, COLUNAS_DESCARTADAS,
                    COLUNAS_NUMERICAS, EXEMPLO, OUT_DIR)
from data import aplicar_log1p
from evaluate import carregar_modelo
from utils import carregar_json, silenciar_bibliotecas


def preparar(registros: list[dict]) -> pd.DataFrame:
    """Lista de dicionários -> DataFrame no formato que o pipeline espera.

    Campos ausentes são criados como NaN de propósito: em produção, formulário
    real sempre chega incompleto. O imputador dentro do pipeline preenche os
    numéricos com a mediana DO TREINO; os categóricos viram "Desconhecido",
    exatamente como em data.py.

    Note o `aplicar_log1p`: ele é obrigatório aqui porque ficou fora do
    pipeline. Toda transformação que você deixa de fora vira uma linha que
    alguém precisa lembrar de repetir — e essa é a definição de dívida técnica
    em projeto de aprendizado de máquina.
    """
    df = pd.DataFrame(registros)
    for coluna in COLUNAS_NUMERICAS:
        if coluna not in df.columns:
            df[coluna] = pd.NA
    for coluna in COLUNAS_CATEGORICAS:
        if coluna not in df.columns:
            df[coluna] = CATEGORIA_AUSENTE
        df[coluna] = df[coluna].fillna(CATEGORIA_AUSENTE).astype(str)
    return aplicar_log1p(df[list(COLUNAS_NUMERICAS) + list(COLUNAS_CATEGORICAS)])


def prever(pipeline, df: pd.DataFrame, limiar: float):
    """Devolve (probabilidades, decisões)."""
    previsoes = predict_model(pipeline, data=df, raw_score=True, round=6, verbose=False)
    prob = previsoes["prediction_score_1"].to_numpy(dtype=float)
    return prob, prob >= limiar


def main():
    p = argparse.ArgumentParser(description="Classifica cadastros com o modelo treinado")
    p.add_argument("--json", type=Path, help="arquivo JSON com um objeto ou uma lista")
    p.add_argument("--csv", type=Path, help="CSV no formato do adult.data")
    p.add_argument("--n", type=int, default=5, help="quantas linhas do CSV usar")
    p.add_argument("--exemplo", action="store_true",
                   help="usa o cadastro de exemplo do config.py")
    p.add_argument("--limiar", type=float, default=None)
    args = p.parse_args()

    silenciar_bibliotecas()

    if args.json:
        conteudo = json.loads(args.json.read_text(encoding="utf-8"))
        registros = conteudo if isinstance(conteudo, list) else [conteudo]
    elif args.csv:
        df = pd.read_csv(args.csv, names=COLUNAS, sep=",", skipinitialspace=True,
                         na_values="?").head(args.n)
        df = df.drop(columns=[c for c in [ALVO] + COLUNAS_DESCARTADAS
                              if c in df.columns])
        registros = df.to_dict(orient="records")
    else:
        if not args.exemplo:
            print("Nenhuma entrada informada; usando --exemplo.\n")
        registros = [EXEMPLO]

    pipeline = carregar_modelo()
    historico = carregar_json(OUT_DIR / CFG.arq_historico)
    limiar = args.limiar if args.limiar is not None else historico.get(
        "limiar", CFG.limiar_padrao)

    df = preparar(registros)
    probabilidades, decisoes = prever(pipeline, df, limiar)

    print(f"Modelo: {historico.get('modelo')} | limiar de decisão: {limiar:.2f}\n")
    for registro, prob, decisao in zip(registros, probabilidades, decisoes):
        resumo = (f"{registro.get('age', '?')} anos, "
                  f"{registro.get('occupation', '?')}, "
                  f"{registro.get('hours-per-week', '?')} h/semana")
        barra = "#" * int(prob * 40)
        print(f"{resumo}")
        print(f"  P(>50K) = {prob:6.2%}  {barra}")
        print(f"  decisão : {CLASSES[1] if decisao else CLASSES[0]}")

        # Sinal de alerta útil em produção: probabilidade perto do limiar é
        # decisão frágil — um caso desses merece revisão humana, não automação.
        if abs(prob - limiar) < 0.05:
            print("  [aviso] probabilidade muito próxima do limiar — "
                  "decisão pouco confiável")
        print()


if __name__ == "__main__":
    main()
