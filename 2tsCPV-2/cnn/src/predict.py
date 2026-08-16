"""
predict.py — inferência em imagens novas (uma ou várias)
---------------------------------------------------------

Este script é a ponte entre "o modelo funciona no notebook" e "o modelo
funciona no mundo". Aqui aparece o erro clássico de quem está começando:

    TREINO e INFERÊNCIA precisam usar EXATAMENTE o mesmo pré-processamento.

Se no treino você normalizou com média 0.286 e desvio 0.353, na inferência tem
que ser a mesma coisa. Qualquer diferença (tamanho, escala de cinza, ordem dos
canais, faixa de valores) derruba a acurácia sem gerar nenhum erro de execução.

Uso:
    python src/predict.py caminho/da/imagem.png
    python src/predict.py img1.png img2.jpg --inverter
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import CFG
from evaluate import carregar_checkpoint
from utils import obter_dispositivo


def preparar_imagem(caminho: Path, inverter: bool = False) -> torch.Tensor:
    """Lê um arquivo de imagem e devolve um tensor (1, 1, 28, 28) normalizado.

    inverter: o FashionMNIST tem objeto CLARO sobre fundo ESCURO. Fotos comuns
    normalmente são o contrário (roupa escura sobre fundo claro). Nesses casos
    use --inverter, senão o modelo vê algo muito diferente do que aprendeu.
    """
    imagem = Image.open(caminho).convert("L")            # força escala de cinza
    if inverter:
        from PIL import ImageOps
        imagem = ImageOps.invert(imagem)

    transformacao = transforms.Compose([
        transforms.Resize((CFG.tamanho_imagem, CFG.tamanho_imagem)),
        transforms.ToTensor(),
        transforms.Normalize((CFG.media,), (CFG.desvio,)),
    ])
    return transformacao(imagem).unsqueeze(0)            # adiciona a dimensão de lote


@torch.no_grad()
def prever(modelo, tensor, classes, dispositivo, k: int = 3):
    """Devolve as k classes mais prováveis com suas probabilidades."""
    logits = modelo(tensor.to(dispositivo))
    # Só AQUI aplicamos softmax: converte escores arbitrários em probabilidades
    # que somam 1. Durante o treino isso é feito dentro da CrossEntropyLoss.
    probabilidades = torch.softmax(logits, dim=1)[0]
    valores, indices = probabilidades.topk(min(k, len(classes)))
    return [(classes[i], v.item()) for v, i in zip(valores, indices)]


def main():
    p = argparse.ArgumentParser(description="Classifica imagens com o modelo treinado")
    p.add_argument("imagens", nargs="+", type=Path)
    p.add_argument("--inverter", action="store_true",
                   help="inverte preto/branco (use para fotos de fundo claro)")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo)
    classes = ckpt["classes"]

    for caminho in args.imagens:
        if not caminho.exists():
            print(f"[erro] arquivo não encontrado: {caminho}")
            continue
        tensor = preparar_imagem(caminho, args.inverter)
        resultado = prever(modelo, tensor, classes, dispositivo, args.topk)

        print(f"\n{caminho.name}")
        for posicao, (nome, prob) in enumerate(resultado, 1):
            barra = "#" * int(prob * 30)
            print(f"  {posicao}. {nome:<16} {prob * 100:6.2f}%  {barra}")

        # Sinal de alerta útil em produção: confiança baixa costuma indicar
        # imagem fora da distribuição de treino (out-of-distribution).
        if resultado[0][1] < 0.5:
            print("  [aviso] confiança baixa — imagem possivelmente fora do domínio")


if __name__ == "__main__":
    main()
