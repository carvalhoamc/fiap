"""
utils.py — funções de apoio
----------------------------
Semente, dispositivo, leitura/escrita de JSON, métricas e gráficos.

As métricas (matriz de confusão, precisão, revocação, F1, AUC) estão
implementadas à mão, com numpy. Existe scikit-learn para isso, e em projeto
real você usaria — mas entender COMO cada número sai da matriz de confusão faz
parte do objetivo da aula. Métrica que você não sabe calcular é métrica que
você não sabe interpretar.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch


def definir_semente(semente: int = 42) -> None:
    """Fixa a semente de TODOS os geradores aleatórios envolvidos.

    Sem isso, dois treinos com o mesmo código dão resultados diferentes: a
    inicialização dos pesos, o embaralhamento do DataLoader, o dropout e o
    sorteio treino/validação são aleatórios.
    """
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def obter_dispositivo() -> torch.device:
    """Usa GPU se existir; caso contrário, CPU. O código não muda.

    Para um MLP deste tamanho a CPU é suficiente — e muitas vezes mais rápida
    que a GPU, porque o gargalo passa a ser a transferência de lotes pequenos,
    não a conta em si.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def salvar_json(dados, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


def carregar_json(caminho: Path):
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def contar_parametros(modelo: torch.nn.Module) -> int:
    """Número de parâmetros treináveis — a capacidade do modelo."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Métricas de classificação binária
# ---------------------------------------------------------------------------
def matriz_confusao_binaria(y, previsto):
    """Devolve (vn, fp, fn, vp).

        vn = disse não, era não      fp = disse sim, era não  (alarme falso)
        fn = disse não, era sim      vp = disse sim, era sim
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(previsto).astype(int)
    vp = int(((p == 1) & (y == 1)).sum())
    vn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return vn, fp, fn, vp


def metricas_binarias(y, probabilidades, limiar: float = 0.5) -> dict:
    """Todas as métricas que importam num problema desbalanceado.

    Acurácia sozinha mente: neste dataset, um modelo que chuta sempre "<=50K"
    acerta cerca de 76%. Precisão e revocação da classe positiva é que dizem
    se o modelo aprendeu alguma coisa.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(probabilidades, dtype=float)
    previsto = (prob >= limiar).astype(int)

    vn, fp, fn, vp = matriz_confusao_binaria(y, previsto)
    total = vn + fp + fn + vp

    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    especificidade = vn / (vn + fp) if vn + fp else 0.0

    return {
        "limiar": float(limiar),
        "acuracia": (vp + vn) / total if total else 0.0,
        "precisao": precisao,
        "revocacao": revocacao,
        "f1": f1,
        "especificidade": especificidade,
        "auc_roc": auc_roc(y, prob),
        "auc_pr": auc_precisao_revocacao(y, prob),
        "matriz": {"vn": vn, "fp": fp, "fn": fn, "vp": vp},
        "n": int(total),
    }


def auc_roc(y, escores) -> float:
    """Área sob a curva ROC, pela identidade com a estatística de Mann-Whitney.

    Interpretação direta e útil em sala: é a probabilidade de o modelo dar um
    escore maior a um positivo sorteado ao acaso do que a um negativo sorteado
    ao acaso. 0,5 = moeda; 1,0 = separação perfeita.

    Vantagem sobre a acurácia: NÃO depende do limiar escolhido nem da proporção
    entre as classes — mede a qualidade do ORDENAMENTO.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(escores, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ordem = np.argsort(s, kind="mergesort")
    postos = np.empty(len(s), dtype=float)
    postos[ordem] = np.arange(1, len(s) + 1)

    # Empates recebem o posto médio; sem isso o AUC passaria a depender da
    # ordem em que as amostras chegaram.
    s_ord = s[ordem]
    i = 0
    while i < len(s_ord):
        j = i
        while j + 1 < len(s_ord) and s_ord[j + 1] == s_ord[i]:
            j += 1
        if j > i:
            postos[ordem[i:j + 1]] = (i + j + 2) / 2
        i = j + 1

    soma_postos_pos = postos[y == 1].sum()
    return float((soma_postos_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def curva_roc(y, escores):
    """Devolve (fpr, tpr) percorrendo todos os limiares possíveis."""
    y = np.asarray(y).astype(int)
    s = np.asarray(escores, dtype=float)
    ordem = np.argsort(-s, kind="mergesort")
    y_ord = y[ordem]
    tp = np.cumsum(y_ord)
    fp = np.cumsum(1 - y_ord)
    n_pos, n_neg = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    return fpr, tpr


def curva_precisao_revocacao(y, escores):
    """Devolve (revocacao, precisao).

    Mais informativa que a ROC quando a classe positiva é rara, porque a
    precisão depende da prevalência e a taxa de falsos positivos não.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(escores, dtype=float)
    ordem = np.argsort(-s, kind="mergesort")
    y_ord = y[ordem]
    tp = np.cumsum(y_ord)
    previstos_pos = np.arange(1, len(y_ord) + 1)
    precisao = tp / previstos_pos
    revocacao = tp / max(int(y.sum()), 1)
    return revocacao, precisao


def auc_precisao_revocacao(y, escores) -> float:
    """Precisão média (área sob a curva P-R).

    A linha de base aqui é a prevalência da classe positiva (~0,24 neste
    dataset), e não 0,5 como na ROC.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(escores, dtype=float)
    ordem = np.argsort(-s, kind="mergesort")
    y_ord = y[ordem]
    tp = np.cumsum(y_ord)
    precisao = tp / np.arange(1, len(y_ord) + 1)
    n_pos = max(int(y.sum()), 1)
    return float((precisao * y_ord).sum() / n_pos)


def melhor_limiar(y, probabilidades, criterio: str = "f1"):
    """Varre limiares e devolve (limiar, valor do critério).

    ATENÇÃO METODOLÓGICA: esta função só pode ser chamada com dados de
    VALIDAÇÃO. Escolher o limiar olhando o teste é a mesma coisa que ajustar
    hiperparâmetro no teste — o número final vira propaganda, não estimativa.
    """
    melhor_valor, melhor_t = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        m = metricas_binarias(y, probabilidades, float(t))
        if m[criterio] > melhor_valor:
            melhor_valor, melhor_t = m[criterio], float(t)
    return round(melhor_t, 2), melhor_valor


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _plt():
    """Importa matplotlib com backend sem janela (funciona em servidor)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando gráfico.")
        return None


def plotar_curvas(historico: dict, caminho: Path) -> None:
    """Perda de treino x validação e métricas de validação por época.

    É o gráfico de diagnóstico: underfitting (as duas curvas ruins e paradas),
    overfitting (treino melhora enquanto a validação piora) e taxa de
    aprendizado inadequada (curva serrilhada) aparecem todos aqui.
    """
    plt = _plt()
    if plt is None:
        return
    epocas = range(1, len(historico["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epocas, historico["train_loss"], "o-", label="treino", ms=3)
    ax1.plot(epocas, historico["val_loss"], "s-", label="validação", ms=3)
    ax1.set_xlabel("época")
    ax1.set_ylabel("perda (BCE ponderada)")
    ax1.set_title("Curva de perda")
    ax1.legend()
    ax1.grid(alpha=.3)

    ax2.plot(epocas, historico["val_auc"], "s-", color="tab:green", ms=3,
             label="AUC-ROC (validação)")
    ax2.plot(epocas, historico["val_f1"], "^-", color="tab:orange", ms=3,
             label="F1 (validação, limiar 0,5)")
    ax2.set_xlabel("época")
    ax2.set_ylabel("métrica")
    ax2.set_title("Desempenho na validação")
    ax2.legend()
    ax2.grid(alpha=.3)

    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Curvas salvas em {caminho}")


def plotar_matriz_confusao(matriz: dict, classes, caminho: Path, limiar: float) -> None:
    """Matriz 2x2 com contagens absolutas e percentual por linha."""
    plt = _plt()
    if plt is None:
        return
    m = np.array([[matriz["vn"], matriz["fp"]],
                  [matriz["fn"], matriz["vp"]]], dtype=float)
    linhas = m.sum(axis=1, keepdims=True)
    linhas[linhas == 0] = 1
    m_norm = m / linhas

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(m_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], [f"previsto\n{classes[0]}", f"previsto\n{classes[1]}"])
    ax.set_yticks([0, 1], [f"real\n{classes[0]}", f"real\n{classes[1]}"])
    ax.set_title(f"Matriz de confusão (limiar = {limiar:.2f})")

    rotulos = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{rotulos[i][j]}\n{int(m[i, j])}\n{m_norm[i, j]:.1%}",
                    ha="center", va="center", fontsize=11,
                    color="white" if m_norm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Matriz de confusão salva em {caminho}")


def plotar_roc_pr(y, probabilidades, caminho: Path) -> None:
    """ROC e Precisão-Revocação lado a lado, com as respectivas linhas de base."""
    plt = _plt()
    if plt is None:
        return
    y = np.asarray(y).astype(int)
    fpr, tpr = curva_roc(y, probabilidades)
    rev, prec = curva_precisao_revocacao(y, probabilidades)
    prevalencia = float(y.mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fpr, tpr, color="tab:blue", label=f"AUC = {auc_roc(y, probabilidades):.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="chute aleatório (0,5)")
    ax1.set_xlabel("taxa de falsos positivos")
    ax1.set_ylabel("taxa de verdadeiros positivos (revocação)")
    ax1.set_title("Curva ROC")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=.3)

    ax2.plot(rev, prec, color="tab:red",
             label=f"precisão média = {auc_precisao_revocacao(y, probabilidades):.3f}")
    ax2.axhline(prevalencia, ls="--", color="gray",
                label=f"linha de base = {prevalencia:.3f}")
    ax2.set_xlabel("revocação")
    ax2.set_ylabel("precisão")
    ax2.set_title("Curva Precisão-Revocação")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=.3)

    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Curvas ROC/PR salvas em {caminho}")


def plotar_importancia(nomes, quedas, caminho: Path, n: int = 15) -> None:
    """Barras horizontais da importância por permutação (ver evaluate.py)."""
    plt = _plt()
    if plt is None:
        return
    ordem = np.argsort(quedas)[-n:]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(ordem) + 1.8))
    ax.barh([nomes[i] for i in ordem], [quedas[i] for i in ordem], color="tab:purple")
    ax.set_xlabel("queda no AUC ao embaralhar a coluna")
    ax.set_title(f"Importância por permutação (top {len(ordem)}, validação)")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Importâncias salvas em {caminho}")
