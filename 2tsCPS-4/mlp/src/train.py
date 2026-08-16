"""
train.py — ETAPAS 4 e 5 do pipeline: treinamento e validação
-------------------------------------------------------------

O laço de treino do PyTorch tem sempre os mesmos 5 passos:

    1. forward   : logits = modelo(x)
    2. perda     : loss = criterio(logits, y)
    3. zerar     : otimizador.zero_grad()      <- se esquecer, gradientes acumulam
    4. backward  : loss.backward()             <- calcula dLoss/dPeso
    5. passo     : otimizador.step()           <- atualiza os pesos

Validação usa o MESMO forward, porém:
    * modelo.eval()      -> desliga dropout e congela as estatísticas do BatchNorm
    * torch.no_grad()    -> não constrói o grafo de derivadas (mais rápido, menos RAM)
    * sem backward/step  -> os pesos NÃO são atualizados

O QUE MUDA EM RELAÇÃO A UM PROBLEMA BALANCEADO
----------------------------------------------
Só 24% das pessoas ganham mais de 50k. Duas consequências práticas:

  * A perda usa `pos_weight`: cada positivo pesa ~3,15 vezes mais que um
    negativo, senão o caminho mais fácil para a rede é dizer "não" para todo
    mundo e acertar 76%.
  * O checkpoint é escolhido pelo AUC-ROC de validação, não pela acurácia. AUC
    mede a qualidade do ORDENAMENTO e não depende do limiar — é a métrica certa
    para selecionar modelo quando a decisão final ainda vai ser calibrada.

Uso:
    python src/train.py                  # treino completo
    python src/train.py --rapido         # demonstração em sala (subconjunto)
    python src/train.py --epocas 60 --lr 5e-4
    python src/train.py --sem-peso       # ablação: sem pos_weight
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from config import CFG, OUT_DIR
from data import obter_dataloaders
from model import criar_modelo
from utils import (auc_roc, contar_parametros, definir_semente, melhor_limiar,
                   metricas_binarias, obter_dispositivo, plotar_curvas, salvar_json)


def executar_epoca(modelo, carregador, criterio, dispositivo, otimizador=None):
    """Executa uma passada completa sobre `carregador`.

    Se `otimizador` for None, a função opera em modo avaliação (validação/teste).
    Devolve (perda_media, rotulos, probabilidades).
    """
    treinando = otimizador is not None
    modelo.train() if treinando else modelo.eval()

    soma_perda, total = 0.0, 0
    todas_probs, todos_y = [], []
    contexto = torch.enable_grad() if treinando else torch.no_grad()

    with contexto:
        for x, y in carregador:
            x, y = x.to(dispositivo), y.to(dispositivo)

            logits = modelo(x)              # 1. forward
            perda = criterio(logits, y)     # 2. perda

            if treinando:
                otimizador.zero_grad(set_to_none=True)  # 3. zerar gradientes
                perda.backward()                        # 4. retropropagação
                otimizador.step()                       # 5. atualizar pesos

            # Multiplicamos pelo tamanho do lote porque o último lote pode ser
            # menor que os demais — a média simples das médias seria enviesada.
            soma_perda += perda.item() * x.size(0)
            total += x.size(0)
            # A sigmoide entra AQUI, só para registrar a probabilidade; a perda
            # continua sendo calculada a partir do logit.
            todas_probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
            todos_y.append(y.detach().cpu().numpy())

    return soma_perda / total, np.concatenate(todos_y), np.concatenate(todas_probs)


def treinar(args) -> dict:
    definir_semente(CFG.semente)
    dispositivo = obter_dispositivo()
    print(f"Dispositivo: {dispositivo}\n")

    # --- dados ------------------------------------------------------------
    carregador_treino, carregador_val, _, preprocessador = obter_dataloaders(
        CFG, rapido=args.rapido)
    y_treino = carregador_treino.dataset.tensors[1].numpy()
    n_pos, n_neg = int(y_treino.sum()), int(len(y_treino) - y_treino.sum())
    print(f"Treino: {len(y_treino)} linhas ({n_pos} positivos, {n_neg} negativos)")
    print(f"Validação: {len(carregador_val.dataset)} linhas")
    print(f"Features de entrada: {preprocessador.n_features}\n")

    # --- modelo -----------------------------------------------------------
    modelo = criar_modelo(n_features=preprocessador.n_features).to(dispositivo)
    print(f"Parâmetros treináveis: {contar_parametros(modelo):,}\n")

    # --- função de perda --------------------------------------------------
    # BCEWithLogitsLoss = sigmoide + entropia cruzada binária, fundidas em uma
    # única operação numericamente estável. NUNCA use sigmoide no modelo + BCELoss.
    if args.sem_peso:
        peso = None
        print("Perda: BCEWithLogitsLoss SEM pos_weight (ablação)")
    else:
        valor = n_neg / max(n_pos, 1) if CFG.pos_weight == "auto" else float(CFG.pos_weight)
        peso = torch.tensor([valor], device=dispositivo)
        print(f"Perda: BCEWithLogitsLoss com pos_weight = {valor:.2f}")
    criterio = nn.BCEWithLogitsLoss(pos_weight=peso)

    # --- otimizador -------------------------------------------------------
    # Adam adapta a taxa de aprendizado por parâmetro; é a escolha segura para
    # quem está começando. weight_decay penaliza pesos grandes (regularização L2).
    otimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr,
                                  weight_decay=CFG.weight_decay)

    # Reduz a taxa de aprendizado quando a validação empaca: passos grandes no
    # começo para explorar, passos pequenos no fim para refinar.
    escalonador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.5, patience=2)

    historico = {"train_loss": [], "val_loss": [], "train_auc": [],
                 "val_auc": [], "val_f1": [], "val_acc": [], "lr": []}
    melhor_auc, melhor_epoca, epocas_sem_melhora = 0.0, 0, 0
    caminho_ckpt = OUT_DIR / CFG.arq_checkpoint
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    for epoca in range(1, args.epocas + 1):
        t0 = time.time()

        perda_tr, y_tr, p_tr = executar_epoca(modelo, carregador_treino, criterio,
                                              dispositivo, otimizador)
        perda_va, y_va, p_va = executar_epoca(modelo, carregador_val, criterio, dispositivo)

        auc_tr = auc_roc(y_tr, p_tr)
        m_va = metricas_binarias(y_va, p_va, CFG.limiar_padrao)
        lr_atual = otimizador.param_groups[0]["lr"]
        escalonador.step(m_va["auc_roc"])

        historico["train_loss"].append(perda_tr)
        historico["val_loss"].append(perda_va)
        historico["train_auc"].append(auc_tr)
        historico["val_auc"].append(m_va["auc_roc"])
        historico["val_f1"].append(m_va["f1"])
        historico["val_acc"].append(m_va["acuracia"])
        historico["lr"].append(lr_atual)

        marca = ""
        # --- checkpoint: guardamos o MELHOR modelo, não o último -----------
        # A última época quase nunca é a melhor. Selecionar pelo desempenho de
        # VALIDAÇÃO (nunca de teste!) é o que caracteriza um protocolo honesto.
        if m_va["auc_roc"] > melhor_auc:
            melhor_auc, melhor_epoca, epocas_sem_melhora = m_va["auc_roc"], epoca, 0
            torch.save({
                "model_state": modelo.state_dict(),
                "hiperparametros": modelo.hiperparametros,
                "preprocessador": preprocessador.to_dict(),
                "epoca": epoca,
                "val_auc": m_va["auc_roc"],
                "config": CFG.to_dict(),
            }, caminho_ckpt)
            marca = "  <- melhor"
        else:
            epocas_sem_melhora += 1

        print(f"época {epoca:>3}/{args.epocas} | perda tr {perda_tr:.4f} va {perda_va:.4f} | "
              f"AUC va {m_va['auc_roc']:.4f} | F1 va {m_va['f1']:.4f} | "
              f"acc va {m_va['acuracia'] * 100:.2f}% | {time.time() - t0:.1f}s{marca}")

        # --- early stopping ------------------------------------------------
        # Interrompe quando a validação para de melhorar: economiza tempo e
        # evita continuar decorando o conjunto de treino (overfitting).
        if epocas_sem_melhora >= CFG.paciencia:
            print(f"\nEarly stopping na época {epoca} (paciência = {CFG.paciencia}).")
            break
    print("=" * 78)

    # --- escolha do limiar de decisão -------------------------------------
    # O modelo devolve probabilidade; a DECISÃO exige um corte. O corte é
    # escolhido na VALIDAÇÃO — nunca no teste — recarregando o melhor
    # checkpoint, para que o limiar corresponda ao modelo que será publicado.
    ckpt = torch.load(caminho_ckpt, map_location=dispositivo, weights_only=False)
    modelo.load_state_dict(ckpt["model_state"])
    _, y_va, p_va = executar_epoca(modelo, carregador_val, criterio, dispositivo)
    limiar, f1_limiar = melhor_limiar(y_va, p_va, criterio="f1")
    m_padrao = metricas_binarias(y_va, p_va, CFG.limiar_padrao)

    ckpt["limiar"] = limiar
    torch.save(ckpt, caminho_ckpt)

    print(f"\nMelhor AUC de validação: {melhor_auc:.4f} (época {melhor_epoca})")
    print(f"Limiar 0,50 (padrão) -> F1 {m_padrao['f1']:.4f} | "
          f"precisão {m_padrao['precisao']:.3f} | revocação {m_padrao['revocacao']:.3f}")
    print(f"Limiar {limiar:.2f} (ótimo na validação) -> F1 {f1_limiar:.4f}")
    print(f"Checkpoint: {caminho_ckpt}")

    salvar_json({"historico": historico, "melhor_val_auc": melhor_auc,
                 "melhor_epoca": melhor_epoca, "limiar": limiar,
                 "config": CFG.to_dict()},
                OUT_DIR / CFG.arq_historico)
    plotar_curvas(historico, OUT_DIR / CFG.arq_curvas)
    return historico


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Treinamento do MLP")
    p.add_argument("--epocas", type=int, default=CFG.epocas)
    p.add_argument("--lr", type=float, default=CFG.lr)
    p.add_argument("--rapido", action="store_true",
                   help="subconjunto pequeno, para demonstração em aula")
    p.add_argument("--sem-peso", action="store_true", dest="sem_peso",
                   help="desliga o pos_weight (experimento do exercício 2.4)")
    treinar(p.parse_args())
