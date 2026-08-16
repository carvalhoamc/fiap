"""
xor.py — o experimento que criou (e quase matou) as redes neurais
------------------------------------------------------------------

Este script não faz parte do pipeline de produção. Ele existe para você VER,
em 10 segundos, por que existe a palavra "multicamadas" em perceptron
multicamadas.

A HISTÓRIA
----------
1958 — Rosenblatt apresenta o perceptron: uma única camada, saída = passo(w·x + b).
1969 — Minsky e Papert mostram que o perceptron não consegue aprender o XOR,
       porque XOR não é linearmente separável: não existe UMA reta que separe
       {(0,0), (1,1)} de {(0,1), (1,0)}. O financiamento da área evapora — o
       primeiro "inverno da IA".
1986 — Rumelhart, Hinton e Williams popularizam a retropropagação, que torna
       viável treinar camadas ocultas. Com UMA camada oculta de 2 neurônios o
       XOR sai de graça: a camada oculta redobra o espaço até que o problema
       fique linearmente separável.

A moral vale para o resto da aula: a camada oculta não é "mais do mesmo", ela
muda a REPRESENTAÇÃO do dado. É exatamente isso que a rede faz com as 95
colunas do censo em model.py.

Uso:
    python src/xor.py
"""

import torch
import torch.nn as nn

from config import OUT_DIR
from utils import definir_semente

# As quatro amostras do problema. É o dataset inteiro — sem treino/teste aqui,
# porque a pergunta não é "generaliza?", é "consegue sequer representar?".
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([0., 1., 1., 0.])          # XOR: 1 quando as entradas diferem


def treinar(modelo, epocas: int = 4000, lr: float = 0.1) -> float:
    criterio = nn.BCEWithLogitsLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    for _ in range(epocas):
        logits = modelo(X).squeeze(-1)
        perda = criterio(logits, Y)
        otimizador.zero_grad(set_to_none=True)
        perda.backward()
        otimizador.step()
    return perda.item()


def salvar_fronteiras(sem_oculta, com_oculta) -> None:
    """Desenha a fronteira de decisão dos dois modelos lado a lado."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando gráfico.")
        return

    passo = torch.linspace(-0.3, 1.3, 200)
    grade_x, grade_y = torch.meshgrid(passo, passo, indexing="xy")
    grade = torch.stack([grade_x.reshape(-1), grade_y.reshape(-1)], dim=1)

    fig, eixos = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (nome, modelo) in zip(eixos, [("Sem camada oculta (perceptron)", sem_oculta),
                                          ("Com 1 camada oculta de 2 neurônios", com_oculta)]):
        with torch.no_grad():
            z = torch.sigmoid(modelo(grade).squeeze(-1)).reshape(200, 200)
        ax.contourf(grade_x, grade_y, z, levels=20, cmap="RdBu_r", vmin=0, vmax=1)
        ax.contour(grade_x, grade_y, z, levels=[0.5], colors="k", linewidths=2)
        for (px, py), alvo in zip(X.tolist(), Y.tolist()):
            ax.scatter(px, py, s=900, marker="o", c="white", edgecolors="k",
                       linewidths=1.6, zorder=3)
            ax.text(px, py, f"alvo\n{int(alvo)}", ha="center", va="center",
                    fontsize=8, zorder=4)
        ax.set_title(nome)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    fig.suptitle("XOR: uma reta não resolve; duas regiões resolvem", fontsize=12)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    caminho = OUT_DIR / "xor_fronteira.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"\n[ok] Fronteiras salvas em {caminho}")


if __name__ == "__main__":
    definir_semente(42)

    # Modelo A: uma única camada linear. É o perceptron de 1958 — só consegue
    # traçar UMA reta no plano.
    sem_oculta = nn.Linear(2, 1)

    # Modelo B: uma camada oculta com 2 neurônios e uma não-linearidade.
    # Tente trocar nn.Tanh() por nn.Identity(): volta a falhar, porque sem
    # não-linearidade duas camadas lineares colapsam em uma só.
    com_oculta = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))

    perda_a = treinar(sem_oculta)
    perda_b = treinar(com_oculta)

    print("Problema XOR — 4 amostras, dataset completo\n")
    print(f"{'x1':>4}{'x2':>4}{'alvo':>6}{'sem oculta':>13}{'com oculta':>13}")
    with torch.no_grad():
        pa = torch.sigmoid(sem_oculta(X).squeeze(-1))
        pb = torch.sigmoid(com_oculta(X).squeeze(-1))
    for i in range(4):
        print(f"{X[i,0]:>4.0f}{X[i,1]:>4.0f}{Y[i]:>6.0f}"
              f"{pa[i]:>13.3f}{pb[i]:>13.3f}")

    acertos_a = int(((pa >= 0.5).float() == Y).sum())
    acertos_b = int(((pb >= 0.5).float() == Y).sum())
    print(f"\nsem camada oculta: perda final {perda_a:.4f} | acertos {acertos_a}/4")
    print(f"com camada oculta: perda final {perda_b:.4f} | acertos {acertos_b}/4")
    print("\nO perceptron empaca em 2/4 (equivalente a chutar) por mais que você\n"
          "treine: o problema não é o otimizador, é a CAPACIDADE do modelo.\n"
          "Nenhuma reta separa esses quatro pontos.")

    salvar_fronteiras(sem_oculta, com_oculta)
