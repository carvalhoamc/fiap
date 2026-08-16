"""
evaluate.py — ETAPA 6 do pipeline: teste
-----------------------------------------

Aqui o conjunto de TESTE é usado pela primeira e única vez. Ele responde a
pergunta que interessa: "quanto o modelo acerta em dados que ninguém — nem o
otimizador, nem eu escolhendo hiperparâmetros — jamais viu?"

Acurácia sozinha engana. Também calculamos:

    Matriz de confusão : quem é confundido com quem.
    Precisão (por classe): dos que EU disse que eram X, quantos eram mesmo X?
    Revocação (recall)  : dos que ERAM X, quantos eu encontrei?
    F1                  : média harmônica das duas.

Uso:
    python src/evaluate.py
    python src/evaluate.py --rapido
"""

import argparse

import torch
import torch.nn as nn

from config import CFG, OUT_DIR
from data import obter_dataloaders
from model import criar_modelo
from utils import (obter_dispositivo, plotar_matriz_confusao, salvar_json)


def carregar_checkpoint(dispositivo):
    """Recria a arquitetura a partir dos metadados salvos e carrega os pesos.

    Salvamos os hiperparâmetros junto com o state_dict justamente para não
    depender de o código-fonte continuar igual ao do dia do treino.
    """
    caminho = OUT_DIR / CFG.arq_checkpoint
    if not caminho.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado em {caminho}. Rode antes: python src/train.py")

    ckpt = torch.load(caminho, map_location=dispositivo, weights_only=False)
    modelo = criar_modelo(**ckpt["hiperparametros"]).to(dispositivo)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()   # NUNCA esqueça: sem isso o dropout continua ativo na inferência
    return modelo, ckpt


@torch.no_grad()
def prever_conjunto(modelo, carregador, dispositivo):
    """Devolve (rotulos_verdadeiros, rotulos_previstos, perda_media)."""
    criterio = nn.CrossEntropyLoss(reduction="sum")
    verdadeiros, previstos = [], []
    soma_perda, total = 0.0, 0

    for x, y in carregador:
        x, y = x.to(dispositivo), y.to(dispositivo)
        logits = modelo(x)
        soma_perda += criterio(logits, y).item()
        total += x.size(0)
        verdadeiros.append(y.cpu())
        previstos.append(logits.argmax(dim=1).cpu())

    return torch.cat(verdadeiros), torch.cat(previstos), soma_perda / total


def matriz_confusao(verdadeiros, previstos, n_classes: int):
    """Matriz C onde C[i][j] = nº de amostras da classe i previstas como j.

    Implementada na mão (sem scikit-learn) porque entender a construção da
    matriz é parte do objetivo de aprendizagem.
    """
    m = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for v, p in zip(verdadeiros, previstos):
        m[v, p] += 1
    return m


def metricas_por_classe(m, classes):
    """Precisão, revocação e F1 derivadas da matriz de confusão."""
    resultado = {}
    for i, nome in enumerate(classes):
        vp = m[i, i].item()                      # verdadeiros positivos
        fp = (m[:, i].sum() - m[i, i]).item()    # falsos positivos
        fn = (m[i, :].sum() - m[i, i]).item()    # falsos negativos
        precisao = vp / (vp + fp) if vp + fp else 0.0
        revocacao = vp / (vp + fn) if vp + fn else 0.0
        f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
        resultado[nome] = {"precisao": precisao, "revocacao": revocacao,
                           "f1": f1, "suporte": int(m[i, :].sum().item())}
    return resultado


def salvar_erros(modelo, carregador, classes, dispositivo, caminho, n: int = 12):
    """Salva um mosaico com os erros mais 'confiantes' do modelo.

    Analisar os erros é mais informativo que celebrar a acurácia: eles mostram
    se o problema é do modelo, do dado ou da própria definição das classes.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    erros = []
    modelo.eval()
    with torch.no_grad():
        for x, y in carregador:
            logits = modelo(x.to(dispositivo))
            probs = torch.softmax(logits, dim=1).cpu()
            pred = probs.argmax(dim=1)
            conf = probs.max(dim=1).values
            for i in range(len(y)):
                if pred[i] != y[i]:
                    erros.append((conf[i].item(), x[i], y[i].item(), pred[i].item()))

    if not erros:
        return
    erros.sort(key=lambda e: -e[0])   # os mais confiantes primeiro
    erros = erros[:n]

    linhas, colunas = 3, (n + 2) // 3
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 1.9, linhas * 2.2))
    for ax, (conf, img, verdadeiro, previsto) in zip(eixos.ravel(), erros):
        img = img.squeeze() * CFG.desvio + CFG.media
        ax.imshow(img.numpy(), cmap="gray")
        ax.set_title(f"real: {classes[verdadeiro]}\nprev: {classes[previsto]} ({conf:.0%})",
                     fontsize=7, color="darkred")
    for ax in eixos.ravel():
        ax.axis("off")
    fig.suptitle("Erros com maior confiança", fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Erros salvos em {caminho}")


def avaliar(rapido: bool = False) -> dict:
    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo)
    classes = ckpt["classes"]
    print(f"Modelo da época {ckpt['epoca']} "
          f"(acc. validação {ckpt['val_acc'] * 100:.2f}%)\n")

    _, _, carregador_teste, _ = obter_dataloaders(CFG, rapido=rapido)
    verdadeiros, previstos, perda = prever_conjunto(modelo, carregador_teste, dispositivo)

    acuracia = (verdadeiros == previstos).float().mean().item()
    m = matriz_confusao(verdadeiros, previstos, len(classes))
    por_classe = metricas_por_classe(m, classes)

    print("=" * 66)
    print(f"TESTE — {len(verdadeiros)} imagens")
    print(f"Acurácia global: {acuracia * 100:.2f}%   |   Perda: {perda:.4f}")
    print("=" * 66)
    print(f"{'classe':<16}{'precisão':>10}{'revocação':>12}{'F1':>8}{'n':>7}")
    for nome, met in por_classe.items():
        print(f"{nome:<16}{met['precisao']:>10.3f}{met['revocacao']:>12.3f}"
              f"{met['f1']:>8.3f}{met['suporte']:>7}")

    piores = sorted(por_classe.items(), key=lambda kv: kv[1]["f1"])[:3]
    print("\nClasses mais difíceis: " + ", ".join(f"{n} (F1={m_['f1']:.2f})" for n, m_ in piores))

    plotar_matriz_confusao(m.tolist(), classes, OUT_DIR / CFG.arq_confusao)
    salvar_erros(modelo, carregador_teste, classes, dispositivo, OUT_DIR / "erros_teste.png")

    resultado = {"acuracia": acuracia, "perda": perda,
                 "por_classe": por_classe, "matriz_confusao": m.tolist(),
                 "classes": classes}
    salvar_json(resultado, OUT_DIR / CFG.arq_metricas)
    salvar_json(classes, OUT_DIR / CFG.arq_classes)  # usado no deploy
    return resultado


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Avaliação no conjunto de teste")
    p.add_argument("--rapido", action="store_true")
    avaliar(p.parse_args().rapido)
