"""
utils.py — métricas, limiar e gráficos
---------------------------------------

DIFERENÇA EM RELAÇÃO À AULA DE MLP
----------------------------------
Lá, as métricas estavam implementadas à mão com numpy, de propósito: métrica
que você não sabe calcular é métrica que você não sabe interpretar. Aquele
exercício já foi feito.

Aqui usamos `sklearn.metrics`, que é o que se usa em projeto real — e o que o
próprio PyCaret usa por baixo. O que continua sendo nosso é a ESCOLHA das
métricas e a leitura delas: acurácia sozinha mente num problema com 24% de
positivos, e nenhuma biblioteca vai avisar disso.

Os gráficos também são feitos à mão em vez de `plot_model(...)`. Motivo: os
gráficos do PyCaret são ótimos para explorar no notebook, mas dependem do
objeto de experimento vivo. Estes aqui recebem apenas (y, probabilidade) e por
isso funcionam igual no relatório, no teste de integração da API e num script
de monitoramento em produção.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)


# ---------------------------------------------------------------------------
# Ruído de terminal
# ---------------------------------------------------------------------------
def silenciar_bibliotecas() -> None:
    """Desliga o log tagarela do LightGBM e os avisos de versão do sklearn.

    Não é frescura de estética: com 13 modelos x 5 dobras, o LightGBM sozinho
    imprime centenas de linhas e empurra a tabela de resultados para fora da
    tela. Aluno que não vê a tabela não lê a tabela.
    """
    import logging
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        import lightgbm
        registro = logging.getLogger("lightgbm_silencioso")
        registro.setLevel(logging.CRITICAL)
        registro.addHandler(logging.NullHandler())
        lightgbm.register_logger(registro)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entrada e saída
# ---------------------------------------------------------------------------
def salvar_json(dados, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False, default=str)


def carregar_json(caminho: Path):
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Métricas de classificação binária
# ---------------------------------------------------------------------------
def metricas_binarias(y, probabilidades, limiar: float = 0.5) -> dict:
    """Todas as métricas que importam num problema desbalanceado.

        vn = disse não, era não      fp = disse sim, era não  (alarme falso)
        fn = disse não, era sim      vp = disse sim, era sim

    AUC-ROC não depende do limiar: mede a qualidade do ORDENAMENTO. Já
    precisão, revocação, F1 e acurácia mudam TODAS quando o limiar muda,
    porque dependem de onde você corta a lista ordenada.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(probabilidades, dtype=float)
    previsto = (prob >= limiar).astype(int)

    vn, fp, fn, vp = confusion_matrix(y, previsto, labels=[0, 1]).ravel()
    vn, fp, fn, vp = int(vn), int(fp), int(fn), int(vp)
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
        "auc_roc": float(roc_auc_score(y, prob)),
        "auc_pr": float(average_precision_score(y, prob)),
        "matriz": {"vn": vn, "fp": fp, "fn": fn, "vp": vp},
        "n": int(total),
    }


def melhor_limiar(y, probabilidades, criterio: str = "f1"):
    """Varre limiares de 0,05 a 0,95 e devolve (limiar, valor do critério).

    ATENÇÃO METODOLÓGICA: esta função só pode ser chamada com dados de
    VALIDAÇÃO. Escolher o limiar olhando o teste é a mesma coisa que ajustar
    hiperparâmetro no teste — o número final vira propaganda, não estimativa.

    O PyCaret tem `optimize_threshold()`, que faz a mesma varredura com um
    gráfico bonito. Fazemos à mão porque o limiar precisa ir junto com o
    modelo para produção (ver metadados.json), e não ficar só na tela.
    """
    melhor_valor, melhor_t = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        m = metricas_binarias(y, probabilidades, float(t))
        if m[criterio] > melhor_valor:
            melhor_valor, melhor_t = m[criterio], float(t)
    return round(melhor_t, 2), melhor_valor


def imprimir_metricas(titulo: str, m: dict) -> None:
    mat = m["matriz"]
    print(f"\n{titulo}")
    print(f"  acurácia      {m['acuracia'] * 100:6.2f}%")
    print(f"  precisão      {m['precisao']:6.3f}   (dos previstos >50K, quantos eram)")
    print(f"  revocação     {m['revocacao']:6.3f}   (dos >50K reais, quantos achei)")
    print(f"  F1            {m['f1']:6.3f}")
    print(f"  especificidade{m['especificidade']:6.3f}   (dos <=50K reais, quantos acertei)")
    print(f"  matriz        VN={mat['vn']}  FP={mat['fp']}  FN={mat['fn']}  VP={mat['vp']}")


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


def plotar_ranking(df_comparacao, caminho: Path, metrica: str = "AUC") -> None:
    """Barras horizontais com o AUC de cada modelo do compare_models.

    É o gráfico que substitui a curva de treino da aula de MLP: lá o eixo era o
    tempo (épocas), aqui é o espaço de modelos. A pergunta também muda — de
    "meu treino está saudável?" para "escolhi a família certa de modelo?".
    """
    plt = _plt()
    if plt is None:
        return
    df = df_comparacao.sort_values(metrica)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(df) + 1.6))
    cores = ["tab:red" if nome.lower().startswith("dummy") else "tab:blue"
             for nome in df["Model"]]
    ax.barh(df["Model"], df[metrica], color=cores)
    for i, valor in enumerate(df[metrica]):
        ax.text(valor, i, f"  {valor:.4f}", va="center", fontsize=8)
    ax.set_xlabel(f"{metrica} (validação cruzada de 5 dobras, no treino)")
    ax.set_title(f"compare_models — ranking por {metrica}\n"
                 "(em vermelho: o modelo trivial, a régua de todos os outros)")
    ax.set_xlim(0.4, 1.0)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Ranking salvo em {caminho}")


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
    """ROC e Precisão-Revocação lado a lado, com as respectivas linhas de base.

    A linha de base da P-R é a PREVALÊNCIA (~0,236 aqui), não 0,5 como na ROC.
    É por isso que a curva P-R é mais honesta quando a classe positiva é rara.
    """
    plt = _plt()
    if plt is None:
        return
    y = np.asarray(y).astype(int)
    fpr, tpr, _ = roc_curve(y, probabilidades)
    prec, rev, _ = precision_recall_curve(y, probabilidades)
    prevalencia = float(y.mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fpr, tpr, color="tab:blue",
             label=f"AUC = {roc_auc_score(y, probabilidades):.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="chute aleatório (0,5)")
    ax1.set_xlabel("taxa de falsos positivos")
    ax1.set_ylabel("taxa de verdadeiros positivos (revocação)")
    ax1.set_title("Curva ROC")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=.3)

    ax2.plot(rev, prec, color="tab:red",
             label=f"precisão média = {average_precision_score(y, probabilidades):.3f}")
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
    """Barras horizontais da importância por permutação."""
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


def plotar_calibracao(y, probabilidades, caminho: Path, faixas: int = 10) -> None:
    """Diagrama de confiabilidade: probabilidade prevista x frequência real.

    Um modelo CALIBRADO acerta a probabilidade, não só a ordem: entre os casos
    a que ele deu 30%, cerca de 30% devem ser positivos de verdade. Isso importa
    quando a probabilidade vira decisão de negócio (preço, limite de crédito,
    priorização de fila) e não apenas um ranking.

    Uma curva abaixo da diagonal = o modelo é otimista (promete mais do que
    entrega). Acima = pessimista. Reponderar classes (pos_weight, SMOTE) tende
    a estragar a calibração, mesmo mantendo o AUC — ver o exercício 3.5.
    """
    plt = _plt()
    if plt is None:
        return
    y = np.asarray(y).astype(int)
    prob = np.asarray(probabilidades, dtype=float)
    bordas = np.linspace(0, 1, faixas + 1)
    indice = np.clip(np.digitize(prob, bordas) - 1, 0, faixas - 1)

    x, obs, tamanho = [], [], []
    for b in range(faixas):
        mascara = indice == b
        if mascara.sum() == 0:
            continue
        x.append(prob[mascara].mean())
        obs.append(y[mascara].mean())
        tamanho.append(int(mascara.sum()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="calibração perfeita")
    ax1.plot(x, obs, "o-", color="tab:blue", label="modelo")
    ax1.set_xlabel("probabilidade média prevista")
    ax1.set_ylabel("frequência real de >50K")
    ax1.set_title("Diagrama de confiabilidade")
    ax1.legend()
    ax1.grid(alpha=.3)

    ax2.bar(x, tamanho, width=0.08, color="tab:orange")
    ax2.set_xlabel("probabilidade prevista")
    ax2.set_ylabel("nº de casos na faixa")
    ax2.set_title("Quantas amostras sustentam cada ponto")
    ax2.grid(axis="y", alpha=.3)

    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Calibração salva em {caminho}")
