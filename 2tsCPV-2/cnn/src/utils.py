"""
utils.py
--------
Funções de apoio usadas por vários scripts: semente aleatória, escolha de
dispositivo, salvamento de JSON e gráficos.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch


def definir_semente(semente: int = 42) -> None:
    """Fixa a semente de TODOS os geradores aleatórios envolvidos.

    Sem isso, dois treinos com o mesmo código dão acurácias diferentes, porque
    a inicialização dos pesos, o embaralhamento do DataLoader e o dropout são
    aleatórios. Reprodutibilidade é requisito científico, não capricho.
    """
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def obter_dispositivo() -> torch.device:
    """Usa GPU se existir; caso contrário, CPU. O código não muda."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def salvar_json(dados: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


def carregar_json(caminho: Path) -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def contar_parametros(modelo: torch.nn.Module) -> int:
    """Número de parâmetros treináveis — a 'capacidade' do modelo."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def plotar_curvas(historico: dict, caminho: Path) -> None:
    """Desenha perda e acurácia de treino x validação ao longo das épocas.

    Este é o gráfico mais importante da disciplina: é nele que você diagnostica
    underfitting (ambas as curvas ruins) e overfitting (treino melhora,
    validação piora).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend sem janela, funciona em servidor
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando gráficos.")
        return

    epocas = range(1, len(historico["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epocas, historico["train_loss"], "o-", label="treino")
    ax1.plot(epocas, historico["val_loss"], "s-", label="validação")
    ax1.set_xlabel("época"); ax1.set_ylabel("perda (cross-entropy)")
    ax1.set_title("Curva de perda"); ax1.legend(); ax1.grid(alpha=.3)

    ax2.plot(epocas, [a * 100 for a in historico["train_acc"]], "o-", label="treino")
    ax2.plot(epocas, [a * 100 for a in historico["val_acc"]], "s-", label="validação")
    ax2.set_xlabel("época"); ax2.set_ylabel("acurácia (%)")
    ax2.set_title("Curva de acurácia"); ax2.legend(); ax2.grid(alpha=.3)

    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Curvas salvas em {caminho}")


def plotar_matriz_confusao(matriz, classes, caminho: Path) -> None:
    """Matriz de confusão normalizada por linha (recall por classe)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando matriz de confusão.")
        return

    m = np.array(matriz, dtype=float)
    linhas = m.sum(axis=1, keepdims=True)
    linhas[linhas == 0] = 1  # evita divisão por zero
    m_norm = m / linhas

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Classe prevista"); ax.set_ylabel("Classe verdadeira")
    ax.set_title("Matriz de confusão (normalizada por linha)")

    for i in range(len(classes)):
        for j in range(len(classes)):
            if m_norm[i, j] > 0.005:
                ax.text(j, i, f"{m_norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if m_norm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Matriz de confusão salva em {caminho}")
