"""
evaluate.py — ETAPA 6 do pipeline: teste final
-----------------------------------------------

Aqui o conjunto de TESTE é usado pela primeira e única vez. Ele responde à
pergunta que interessa: "quanto o modelo acerta em dados que ninguém — nem o
otimizador, nem eu escolhendo hiperparâmetros e limiar — jamais viu?"

Acurácia sozinha engana, e neste dataset engana muito: um modelo que responde
"<=50K" para todo mundo acerta 76,4%. Por isso reportamos:

    Matriz de confusão : VN, FP, FN, VP — a estrutura do erro.
    Precisão   : dos que EU disse que ganham mais de 50k, quantos ganham mesmo?
    Revocação  : dos que GANHAM mais de 50k, quantos eu encontrei?
    F1         : média harmônica das duas.
    AUC-ROC    : qualidade do ordenamento, independente do limiar.
    Precisão média (AUC-PR): idem, mas sensível à raridade da classe positiva.

E, no fim, a importância por permutação: quais colunas o modelo realmente usa.

Uso:
    python src/evaluate.py
    python src/evaluate.py --limiar 0.5     # compara com o limiar padrão
    python src/evaluate.py --rapido
"""

import argparse

import numpy as np
import torch

from config import CFG, CLASSES, OUT_DIR
from data import Preprocessador, obter_dataloaders
from model import criar_modelo
from utils import (auc_roc, metricas_binarias, obter_dispositivo,
                   plotar_importancia, plotar_matriz_confusao, plotar_roc_pr,
                   salvar_json)


def carregar_checkpoint(dispositivo):
    """Recria a arquitetura a partir dos metadados salvos e carrega os pesos.

    Salvamos hiperparâmetros e pré-processador junto com o state_dict
    justamente para não depender de o código-fonte continuar igual ao do dia
    do treino.
    """
    caminho = OUT_DIR / CFG.arq_checkpoint
    if not caminho.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado em {caminho}. Rode antes: python src/train.py")

    ckpt = torch.load(caminho, map_location=dispositivo, weights_only=False)
    modelo = criar_modelo(**ckpt["hiperparametros"]).to(dispositivo)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()   # NUNCA esqueça: sem isso o dropout continua ativo e o
                    # BatchNorm segue atualizando estatísticas na inferência.
    return modelo, ckpt


@torch.no_grad()
def prever_conjunto(modelo, carregador, dispositivo):
    """Devolve (rotulos_verdadeiros, probabilidades)."""
    todos_y, todas_probs = [], []
    for x, y in carregador:
        logits = modelo(x.to(dispositivo))
        todas_probs.append(torch.sigmoid(logits).cpu().numpy())
        todos_y.append(y.numpy())
    return np.concatenate(todos_y), np.concatenate(todas_probs)


@torch.no_grad()
def importancia_por_permutacao(modelo, X, y, nomes, dispositivo, repeticoes: int = 3,
                               semente: int = 42):
    """Quanto o AUC cai quando embaralhamos UMA coluna de cada vez.

    A ideia é simples e independe do tipo de modelo: se destruir a relação
    entre a coluna e o alvo não muda nada, o modelo não estava usando aquela
    coluna. É o análogo tabular de visualizar os filtros de uma CNN.

    Calculada na VALIDAÇÃO: o teste é usado uma vez só, para o número final.

    Cuidado ao interpretar: colunas correlacionadas dividem a importância entre
    si (embaralhar uma delas ainda deixa a informação disponível na outra), e
    importância não é causalidade.
    """
    gerador = np.random.default_rng(semente)
    X_t = torch.from_numpy(X).to(dispositivo)
    base = auc_roc(y, torch.sigmoid(modelo(X_t)).cpu().numpy())

    quedas = np.zeros(X.shape[1])
    for coluna in range(X.shape[1]):
        perdas = []
        for _ in range(repeticoes):
            X_emb = X.copy()
            gerador.shuffle(X_emb[:, coluna])
            probs = torch.sigmoid(modelo(torch.from_numpy(X_emb).to(dispositivo)))
            perdas.append(base - auc_roc(y, probs.cpu().numpy()))
        quedas[coluna] = float(np.mean(perdas))
    return base, quedas


def _imprimir_metricas(titulo: str, m: dict) -> None:
    mat = m["matriz"]
    print(f"\n{titulo}")
    print(f"  acurácia      {m['acuracia'] * 100:6.2f}%")
    print(f"  precisão      {m['precisao']:6.3f}   (dos previstos >50K, quantos eram)")
    print(f"  revocação     {m['revocacao']:6.3f}   (dos >50K reais, quantos achei)")
    print(f"  F1            {m['f1']:6.3f}")
    print(f"  especificidade{m['especificidade']:6.3f}   (dos <=50K reais, quantos acertei)")
    print(f"  matriz        VN={mat['vn']}  FP={mat['fp']}  FN={mat['fn']}  VP={mat['vp']}")


def avaliar(rapido: bool = False, limiar_forcado: float | None = None) -> dict:
    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo)
    limiar = limiar_forcado if limiar_forcado is not None else ckpt.get("limiar", CFG.limiar_padrao)

    print(f"Modelo da época {ckpt['epoca']} (AUC de validação {ckpt['val_auc']:.4f})")
    print(f"Limiar de decisão: {limiar:.2f}"
          f"{' (forçado pela linha de comando)' if limiar_forcado is not None else ' (escolhido na validação)'}")

    _, carregador_val, carregador_teste, _ = obter_dataloaders(CFG, rapido=rapido)
    y, prob = prever_conjunto(modelo, carregador_teste, dispositivo)

    m = metricas_binarias(y, prob, limiar)
    prevalencia = float(y.mean())

    print("\n" + "=" * 72)
    print(f"TESTE — {len(y)} pessoas (usado UMA única vez)")
    print("=" * 72)
    print(f"AUC-ROC                {m['auc_roc']:.4f}   (0,5 = chute)")
    print(f"Precisão média (AUC-PR){m['auc_pr']:8.4f}   (linha de base = {prevalencia:.4f})")
    _imprimir_metricas(f"No limiar {limiar:.2f}:", m)

    # --- referências honestas de comparação -------------------------------
    print("\nLinhas de base:")
    print(f"  chutar sempre '<=50K'  -> acurácia {1 - prevalencia:.2%}, "
          f"revocação 0,000 (não encontra ninguém)")
    print(f"  chutar ao acaso        -> AUC 0,500")

    # O mesmo modelo com o limiar padrão, para deixar visível que o limiar
    # move precisão e revocação em direções opostas sem mudar o AUC.
    if limiar_forcado is None and abs(limiar - CFG.limiar_padrao) > 1e-9:
        _imprimir_metricas(f"Para comparação, no limiar padrão {CFG.limiar_padrao:.2f}:",
                           metricas_binarias(y, prob, CFG.limiar_padrao))
        print("\n  Note que o AUC é idêntico nos dois casos: o limiar não muda o\n"
              "  modelo, apenas onde você corta a mesma lista ordenada.")

    plotar_matriz_confusao(m["matriz"], CLASSES, OUT_DIR / CFG.arq_confusao, limiar)
    plotar_roc_pr(y, prob, OUT_DIR / CFG.arq_roc)

    # --- importância por permutação, na validação -------------------------
    prep = Preprocessador.from_dict(ckpt["preprocessador"])
    X_val = carregador_val.dataset.tensors[0].numpy()
    y_val = carregador_val.dataset.tensors[1].numpy()
    base, quedas = importancia_por_permutacao(modelo, X_val, y_val,
                                              prep.nomes_features, dispositivo)
    ordem = np.argsort(-quedas)[:10]
    print(f"\nImportância por permutação (validação, AUC base {base:.4f}) — top 10:")
    for i in ordem:
        print(f"  {prep.nomes_features[i]:<34} queda de AUC {quedas[i]:+.4f}")
    plotar_importancia(prep.nomes_features, quedas.tolist(), OUT_DIR / CFG.arq_importancia)

    resultado = {
        "limiar": limiar,
        "metricas_teste": m,
        "prevalencia_teste": prevalencia,
        "importancia": {prep.nomes_features[i]: float(quedas[i])
                        for i in np.argsort(-quedas)[:20]},
        "classes": CLASSES,
    }
    salvar_json(resultado, OUT_DIR / CFG.arq_metricas)
    return resultado


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Avaliação no conjunto de teste")
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--limiar", type=float, default=None,
                   help="força um limiar específico (padrão: o escolhido na validação)")
    args = p.parse_args()
    avaliar(args.rapido, args.limiar)
