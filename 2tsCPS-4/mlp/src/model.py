"""
model.py — ETAPA 3 do pipeline: arquitetura
--------------------------------------------

O perceptron multicamadas (MLP) em uma frase: uma pilha de transformações
lineares intercaladas por não-linearidades.

    h1 = f(W1 @ x  + b1)
    h2 = f(W2 @ h1 + b2)
    z  =   W3 @ h2 + b3        <- logit (um número, sem sigmoide)

POR QUE A NÃO-LINEARIDADE É OBRIGATÓRIA
---------------------------------------
Sem f, a conta acima seria W3(W2(W1 x)) = (W3 W2 W1) x = W x. Ou seja: mil
camadas empilhadas colapsariam em UMA transformação linear — um modelo com a
mesma capacidade de uma regressão logística, gastando mil vezes mais memória.
É a não-linearidade que faz a profundidade valer alguma coisa. Veja isso na
prática rodando `python src/xor.py`.

TEOREMA DA APROXIMAÇÃO UNIVERSAL
--------------------------------
Uma rede com UMA camada oculta suficientemente larga aproxima qualquer função
contínua com a precisão que você quiser. Cuidado com a leitura: o teorema diz
que existe tal rede, não que o gradiente descendente vai encontrá-la, nem com
quantos dados. Na prática, várias camadas estreitas costumam aprender melhor
que uma única muito larga.

POR QUE ISSO É UM MLP E NÃO UMA CNN
-----------------------------------
Numa imagem, pixels vizinhos têm relação espacial — daí o filtro que desliza.
Numa tabela, a coluna 3 (`occupation`) e a coluna 4 (`relationship`) não são
"vizinhas" em nenhum sentido: trocar a ordem das colunas não muda o problema.
Sem estrutura local para explorar, a camada totalmente conectada é a escolha
correta.
"""

import torch
import torch.nn as nn

from config import CFG


def bloco_denso(entrada: int, saida: int, dropout: float, usar_bn: bool) -> nn.Sequential:
    """Bloco padrão: Linear -> BatchNorm -> ReLU -> Dropout.

    Linear     : a transformação afim, onde estão os pesos aprendidos.
                 bias=False quando há BatchNorm em seguida, porque o BN já tem
                 um termo de deslocamento — o bias seria redundante.
    BatchNorm1d: normaliza as ativações dentro do lote. Estabiliza o treino e
                 permite taxas de aprendizado maiores.
    ReLU       : f(x) = max(0, x). Barata, não satura para x > 0 e por isso
                 sofre muito menos com gradiente que desaparece do que a
                 sigmoide (cuja derivada máxima é 0,25 — multiplicada camada a
                 camada, o gradiente some).
    Dropout    : desliga neurônios ao acaso durante o TREINO. Impede que a rede
                 dependa de um caminho único. Só age em modo treino, por isso
                 `modelo.eval()` é obrigatório na inferência.
    """
    camadas = [nn.Linear(entrada, saida, bias=not usar_bn)]
    if usar_bn:
        camadas.append(nn.BatchNorm1d(saida))
    camadas.append(nn.ReLU(inplace=True))
    if dropout > 0:
        camadas.append(nn.Dropout(dropout))
    return nn.Sequential(*camadas)


class MLP(nn.Module):
    """Perceptron multicamadas para classificação binária tabular.

    Formato do tensor (lote N, 95 features de entrada, camadas (64, 32)):

        entrada     (N, 95)
        oculta 1 -> (N, 64)
        oculta 2 -> (N, 32)
        saída    -> (N, 1)   = logit, um escore por amostra

    A saída tem UM neurônio, não dois. Em classificação binária basta um
    escore: P(classe 1) = sigmoide(z) e P(classe 0) = 1 - P(classe 1). Duas
    saídas com softmax dariam o mesmo resultado com o dobro de parâmetros na
    última camada.
    """

    def __init__(self, n_features: int, camadas_ocultas: tuple = CFG.camadas_ocultas,
                 dropout: float = CFG.dropout, usar_batchnorm: bool = CFG.usar_batchnorm):
        super().__init__()

        blocos, entrada = [], n_features
        for largura in camadas_ocultas:
            blocos.append(bloco_denso(entrada, largura, dropout, usar_batchnorm))
            entrada = largura
        self.oculta = nn.Sequential(*blocos)
        self.saida = nn.Linear(entrada, 1)

        self._inicializar()

        # Guardados junto com os pesos para que evaluate.py consiga recriar a
        # arquitetura sem depender de config.py continuar igual ao dia do treino.
        self.hiperparametros = dict(
            n_features=n_features,
            camadas_ocultas=tuple(camadas_ocultas),
            dropout=dropout,
            usar_batchnorm=usar_batchnorm,
        )

    def _inicializar(self) -> None:
        """Inicialização de He (Kaiming), apropriada para ReLU.

        Pesos todos iguais a zero seriam fatais: todos os neurônios de uma
        camada receberiam o mesmo gradiente e aprenderiam a mesma coisa para
        sempre — é o problema da simetria. Pesos grandes demais explodem as
        ativações; pequenos demais, somem. A inicialização de He escolhe a
        variância que mantém o sinal estável ao atravessar camadas ReLU.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devolve o LOGIT com formato (N,), SEM sigmoide.

        Muito importante: nn.BCEWithLogitsLoss já aplica a sigmoide
        internamente, de forma numericamente estável (log-sum-exp). Aplicar
        sigmoide aqui e usar BCELoss funciona em teoria, mas satura e produz
        NaN com logits grandes. A sigmoide só entra na hora de interpretar a
        saída como probabilidade (predict.py e deploy/api.py).
        """
        return self.saida(self.oculta(x)).squeeze(-1)


def criar_modelo(**kwargs) -> MLP:
    return MLP(**kwargs)


if __name__ == "__main__":
    # Rode:  python src/model.py
    # Confere formatos e conta parâmetros ANTES de gastar tempo treinando.
    from data import obter_dataloaders
    from utils import contar_parametros

    # Dataset completo de propósito: o número de features de entrada depende
    # de quantas categorias aparecem no treino. Com --rapido, o subconjunto não
    # contém todos os 41 países e a entrada encolhe de 95 para ~92 colunas.
    _, _, _, prep = obter_dataloaders(rapido=False)
    modelo = criar_modelo(n_features=prep.n_features)
    print(modelo)
    print(f"\nEntrada: {prep.n_features} features "
          f"({len(prep.nomes_features)} nomes registrados)")
    print(f"Parâmetros treináveis: {contar_parametros(modelo):,}")

    # Conferência manual da contagem: cada Linear tem entrada*saida pesos + saida
    # vieses; cada BatchNorm1d tem 2*largura parâmetros (escala e deslocamento).
    print("\nDe onde vêm os parâmetros:")
    total = 0
    entrada = prep.n_features
    for i, largura in enumerate(CFG.camadas_ocultas, 1):
        pesos = entrada * largura
        bn = 2 * largura if CFG.usar_batchnorm else largura  # BN (2) ou bias (1)
        print(f"  oculta {i}: {entrada} x {largura} = {pesos:,} pesos + {bn} "
              f"({'BatchNorm' if CFG.usar_batchnorm else 'bias'})")
        total += pesos + bn
        entrada = largura
    print(f"  saída   : {entrada} x 1 = {entrada} pesos + 1 bias")
    total += entrada + 1
    print(f"  total   : {total:,}")

    # Modo eval porque o BatchNorm1d recusa lote de tamanho 1 em modo treino.
    modelo.eval()
    x = torch.randn(4, prep.n_features)
    with torch.no_grad():
        z = modelo(x)
    print(f"\nEntrada {tuple(x.shape)} -> logits {tuple(z.shape)}")
    print(f"logits         : {z.numpy().round(3)}")
    print(f"probabilidades : {torch.sigmoid(z).numpy().round(3)}  (só após a sigmoide)")

    print("\nFormato após cada camada oculta:")
    h = x
    for i, bloco in enumerate(modelo.oculta, 1):
        with torch.no_grad():
            h = bloco(h)
        print(f"  oculta {i}: {tuple(h.shape)}")
