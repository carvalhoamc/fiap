"""
model.py — ETAPA 3 do pipeline: arquitetura
--------------------------------------------

A ideia central da CNN em uma frase: em vez de conectar cada pixel a cada
neurônio (o que exigiria milhões de pesos e ignoraria a estrutura espacial da
imagem), aprendemos pequenos FILTROS que deslizam sobre a imagem procurando
padrões locais — bordas, cantos, texturas — e empilhamos camadas para que
padrões simples se combinem em padrões complexos.

Três propriedades que justificam a convolução para imagens:
  * Conectividade local  : cada neurônio olha só uma vizinhança (ex.: 3x3).
  * Pesos compartilhados : o mesmo filtro é aplicado em toda a imagem, então
                           uma borda é detectada em qualquer posição.
  * Invariância a translação (via pooling): deslocar o objeto alguns pixels
                           não muda a resposta da rede.

Fórmula do tamanho da saída de uma convolução/pooling:
    saida = floor((entrada + 2*padding - kernel) / stride) + 1

Com kernel=3, padding=1, stride=1 -> a resolução NÃO muda (conv "same").
Com MaxPool kernel=2, stride=2    -> a resolução cai pela metade.
"""

import torch
import torch.nn as nn

from config import CFG


def bloco_conv(entrada: int, saida: int) -> nn.Sequential:
    """Bloco convolucional padrão: Conv -> BatchNorm -> ReLU -> MaxPool.

    Conv2d     : aprende `saida` filtros de 3x3 sobre `entrada` canais.
                 bias=False porque o BatchNorm logo em seguida já tem um termo
                 de deslocamento — o bias seria redundante.
    BatchNorm2d: normaliza as ativações dentro do lote. Estabiliza e acelera
                 muito o treino; permite taxas de aprendizado maiores.
    ReLU       : não-linearidade f(x)=max(0,x). Sem ela, empilhar camadas seria
                 equivalente a uma única transformação linear.
    MaxPool2d  : reduz a resolução pela metade, mantendo a ativação mais forte
                 de cada janela 2x2. Menos parâmetros à frente e mais robustez
                 a pequenos deslocamentos.
    """
    return nn.Sequential(
        nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(saida),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class CNNSimples(nn.Module):
    """CNN de 3 blocos para imagens 28x28 em escala de cinza.

    Evolução do formato do tensor (com lote N e canais (16, 32, 64)):

        entrada          (N,  1, 28, 28)
        bloco 1  ->      (N, 16, 14, 14)
        bloco 2  ->      (N, 32,  7,  7)
        bloco 3  ->      (N, 64,  3,  3)   # 7//2 = 3
        flatten  ->      (N, 576)
        fc1      ->      (N, 128)
        fc2      ->      (N, 10)  = logits, um escore por classe
    """

    def __init__(self, n_classes: int = CFG.n_classes, canais_entrada: int = CFG.canais,
                 canais_conv: tuple = CFG.canais_conv, neuronios_fc: int = CFG.neuronios_fc,
                 dropout: float = CFG.dropout, tamanho_imagem: int = CFG.tamanho_imagem):
        super().__init__()

        blocos = []
        c_ant = canais_entrada
        for c in canais_conv:
            blocos.append(bloco_conv(c_ant, c))
            c_ant = c
        self.extrator = nn.Sequential(*blocos)

        # Em vez de calcular o tamanho do flatten "na mão" (fonte clássica de
        # erro), descobrimos empiricamente com um tensor falso.
        with torch.no_grad():
            falso = torch.zeros(1, canais_entrada, tamanho_imagem, tamanho_imagem)
            n_achatado = self.extrator(falso).flatten(1).shape[1]

        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),          # desliga neurônios ao acaso: regularização
            nn.Linear(n_achatado, neuronios_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(neuronios_fc, n_classes),
        )

        self.hiperparametros = dict(
            n_classes=n_classes, canais_entrada=canais_entrada,
            canais_conv=tuple(canais_conv), neuronios_fc=neuronios_fc,
            dropout=dropout, tamanho_imagem=tamanho_imagem,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devolve LOGITS (sem softmax).

        Muito importante: nn.CrossEntropyLoss já aplica log-softmax
        internamente. Se você colocar um softmax aqui, estará aplicando duas
        vezes e o treino fica lento e instável. O softmax só entra na hora de
        interpretar a saída como probabilidade (predict.py / deploy).
        """
        x = self.extrator(x)
        return self.classificador(x)


def criar_modelo(**kwargs) -> CNNSimples:
    return CNNSimples(**kwargs)


if __name__ == "__main__":
    # Rode:  python src/model.py
    # Confere formatos e conta parâmetros ANTES de gastar tempo treinando.
    from utils import contar_parametros

    modelo = criar_modelo()
    print(modelo)
    print(f"\nParâmetros treináveis: {contar_parametros(modelo):,}")

    x = torch.randn(4, CFG.canais, CFG.tamanho_imagem, CFG.tamanho_imagem)
    with torch.no_grad():
        y = modelo(x)
    print(f"Entrada {tuple(x.shape)} -> saída {tuple(y.shape)} (4 amostras x 10 classes)")

    # Mostra o encolhimento espacial camada a camada
    print("\nFormato após cada bloco convolucional:")
    z = x
    for i, bloco in enumerate(modelo.extrator, 1):
        with torch.no_grad():
            z = bloco(z)
        print(f"  bloco {i}: {tuple(z.shape)}")
