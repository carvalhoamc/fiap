"""
predict.py — inferência em cadastros novos
-------------------------------------------

A ponte entre "o modelo funciona no notebook" e "o modelo funciona no mundo".
E aqui aparece o erro clássico dos projetos tabulares:

    TREINO e INFERÊNCIA precisam usar EXATAMENTE o mesmo pré-processamento.

Em imagens, esquecer a normalização produz um resultado visivelmente errado.
Em tabelas, o desastre é silencioso: se na hora de prever você padronizar com
a média do novo lote em vez da média do treino, ou trocar a ordem das colunas
one-hot, o código roda, devolve uma probabilidade bonita e ERRADA. Nenhuma
exceção é levantada.

Por isso o pré-processador viaja dentro do checkpoint e é reconstruído aqui a
partir dele — nunca refeito do zero.

Uso:
    python src/predict.py --exemplo
    python src/predict.py --json pessoa.json
    python src/predict.py --csv pessoas.csv --n 10
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from config import CFG, CLASSES, COLUNAS, COLUNAS_DESCARTADAS, EXEMPLO
from data import Preprocessador
from evaluate import carregar_checkpoint
from utils import obter_dispositivo


def preparar(registros: list[dict], preprocessador: Preprocessador) -> torch.Tensor:
    """Lista de dicionários -> tensor pronto para a rede.

    Campos ausentes viram NaN e são tratados exatamente como no treino: número
    ausente recebe a mediana do treino, categoria ausente vira "Desconhecido".
    Isso é obrigatório em produção — formulário real sempre chega incompleto.
    """
    df = pd.DataFrame(registros)
    for coluna in COLUNAS:
        if coluna not in df.columns and coluna not in COLUNAS_DESCARTADAS:
            df[coluna] = None
    return torch.from_numpy(preprocessador.transform(df))


@torch.no_grad()
def prever(modelo, tensor, dispositivo, limiar: float):
    """Devolve (probabilidades, decisões)."""
    # A sigmoide só entra AQUI: converte o logit (qualquer número real) em
    # probabilidade entre 0 e 1. Durante o treino isso acontece dentro da
    # BCEWithLogitsLoss.
    probabilidades = torch.sigmoid(modelo(tensor.to(dispositivo))).cpu().numpy()
    return probabilidades, (probabilidades >= limiar)


def main():
    p = argparse.ArgumentParser(description="Classifica cadastros com o modelo treinado")
    p.add_argument("--json", type=Path, help="arquivo JSON com um objeto ou uma lista")
    p.add_argument("--csv", type=Path, help="CSV no formato do adult.data")
    p.add_argument("--n", type=int, default=5, help="quantas linhas do CSV usar")
    p.add_argument("--exemplo", action="store_true", help="usa o cadastro de exemplo do config.py")
    p.add_argument("--limiar", type=float, default=None)
    args = p.parse_args()

    if args.json:
        conteudo = json.loads(args.json.read_text(encoding="utf-8"))
        registros = conteudo if isinstance(conteudo, list) else [conteudo]
    elif args.csv:
        df = pd.read_csv(args.csv, names=COLUNAS, sep=",", skipinitialspace=True,
                         na_values="?").head(args.n)
        registros = df.to_dict(orient="records")
    else:
        if not args.exemplo:
            print("Nenhuma entrada informada; usando --exemplo.\n")
        registros = [EXEMPLO]

    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo)
    preprocessador = Preprocessador.from_dict(ckpt["preprocessador"])
    limiar = args.limiar if args.limiar is not None else ckpt.get("limiar", CFG.limiar_padrao)

    tensor = preparar(registros, preprocessador)
    probabilidades, decisoes = prever(modelo, tensor, dispositivo, limiar)

    print(f"Limiar de decisão: {limiar:.2f}\n")
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
            print("  [aviso] probabilidade muito próxima do limiar — decisão pouco confiável")
        print()


if __name__ == "__main__":
    main()
