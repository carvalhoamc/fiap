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

Uso:
    python src/train.py                 # treino completo (10 épocas)
    python src/train.py --rapido        # demonstração em sala (subconjunto)
    python src/train.py --epocas 20 --lr 5e-4
"""

import argparse
import time

import torch
import torch.nn as nn

from config import CFG, OUT_DIR
from data import obter_dataloaders
from model import criar_modelo
from utils import (contar_parametros, definir_semente, obter_dispositivo,
                   plotar_curvas, salvar_json)


def executar_epoca(modelo, carregador, criterio, dispositivo, otimizador=None,
                   log_a_cada: int = 100, rotulo: str = ""):
    """Executa uma passada completa sobre `carregador`.

    Se `otimizador` for None, a função opera em modo avaliação (validação/teste).
    Devolve (perda_media, acuracia).
    """
    treinando = otimizador is not None
    modelo.train() if treinando else modelo.eval()

    soma_perda, acertos, total = 0.0, 0, 0
    contexto = torch.enable_grad() if treinando else torch.no_grad()

    with contexto:
        for i, (x, y) in enumerate(carregador, start=1):
            x, y = x.to(dispositivo), y.to(dispositivo)

            logits = modelo(x)              # 1. forward
            perda = criterio(logits, y)     # 2. perda

            if treinando:
                otimizador.zero_grad(set_to_none=True)  # 3. zerar gradientes
                perda.backward()                        # 4. retropropagação
                otimizador.step()                       # 5. atualizar pesos

            # .item() traz o escalar para a CPU; multiplicamos pelo tamanho do
            # lote porque o último lote pode ser menor que os demais.
            soma_perda += perda.item() * x.size(0)
            acertos += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

            if treinando and i % log_a_cada == 0:
                print(f"    {rotulo} lote {i:>4}/{len(carregador)} | "
                      f"perda {soma_perda / total:.4f} | acc {acertos / total:.3f}")

    return soma_perda / total, acertos / total


def treinar(args) -> dict:
    definir_semente(CFG.semente)
    dispositivo = obter_dispositivo()
    print(f"Dispositivo: {dispositivo}\n")

    # --- dados ------------------------------------------------------------
    carregador_treino, carregador_val, _, classes = obter_dataloaders(CFG, rapido=args.rapido)
    print(f"Treino: {len(carregador_treino.dataset)} | Validação: {len(carregador_val.dataset)}")

    # --- modelo -----------------------------------------------------------
    modelo = criar_modelo().to(dispositivo)
    print(f"Parâmetros treináveis: {contar_parametros(modelo):,}\n")

    # --- função de perda --------------------------------------------------
    # CrossEntropyLoss = log_softmax + negative log likelihood. É a perda padrão
    # para classificação multiclasse com rótulos inteiros (0..9).
    criterio = nn.CrossEntropyLoss()

    # --- otimizador -------------------------------------------------------
    # Adam adapta a taxa de aprendizado por parâmetro; é a escolha segura para
    # quem está começando. weight_decay penaliza pesos grandes (regularização).
    otimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr,
                                  weight_decay=CFG.weight_decay)

    # Reduz a taxa de aprendizado quando a validação empaca: passos grandes no
    # começo para explorar, passos pequenos no fim para refinar.
    escalonador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="max", factor=0.5, patience=1)

    historico = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    melhor_acc, melhor_epoca, epocas_sem_melhora = 0.0, 0, 0
    caminho_ckpt = OUT_DIR / CFG.arq_checkpoint
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoca in range(1, args.epocas + 1):
        t0 = time.time()
        print(f"Época {epoca}/{args.epocas}")

        perda_tr, acc_tr = executar_epoca(modelo, carregador_treino, criterio,
                                          dispositivo, otimizador, rotulo="treino")
        perda_va, acc_va = executar_epoca(modelo, carregador_val, criterio, dispositivo)

        lr_atual = otimizador.param_groups[0]["lr"]
        escalonador.step(acc_va)

        historico["train_loss"].append(perda_tr)
        historico["train_acc"].append(acc_tr)
        historico["val_loss"].append(perda_va)
        historico["val_acc"].append(acc_va)
        historico["lr"].append(lr_atual)

        print(f"  treino   -> perda {perda_tr:.4f} | acc {acc_tr * 100:.2f}%")
        print(f"  validação-> perda {perda_va:.4f} | acc {acc_va * 100:.2f}%"
              f"  (lr={lr_atual:.1e}, {time.time() - t0:.1f}s)")

        # --- checkpoint: guardamos o MELHOR modelo, não o último -----------
        # A última época quase nunca é a melhor. Selecionar pelo desempenho de
        # VALIDAÇÃO (nunca de teste!) é o que caracteriza um protocolo honesto.
        if acc_va > melhor_acc:
            melhor_acc, melhor_epoca, epocas_sem_melhora = acc_va, epoca, 0
            torch.save({
                "model_state": modelo.state_dict(),
                "hiperparametros": modelo.hiperparametros,
                "classes": classes,
                "epoca": epoca,
                "val_acc": acc_va,
                "config": CFG.to_dict(),
            }, caminho_ckpt)
            print(f"  [ok] novo melhor modelo salvo ({acc_va * 100:.2f}%)")
        else:
            epocas_sem_melhora += 1
            print(f"  sem melhora há {epocas_sem_melhora} época(s)")

        # --- early stopping ------------------------------------------------
        # Interrompe quando a validação para de melhorar: economiza tempo e
        # evita continuar decorando o conjunto de treino (overfitting).
        if epocas_sem_melhora >= CFG.paciencia:
            print(f"\nEarly stopping na época {epoca} "
                  f"(paciência = {CFG.paciencia}).")
            break
        print()

    print(f"\nMelhor acurácia de validação: {melhor_acc * 100:.2f}% (época {melhor_epoca})")
    print(f"Checkpoint: {caminho_ckpt}")

    salvar_json({"historico": historico, "melhor_val_acc": melhor_acc,
                 "melhor_epoca": melhor_epoca, "config": CFG.to_dict()},
                OUT_DIR / CFG.arq_historico)
    plotar_curvas(historico, OUT_DIR / CFG.arq_curvas)
    return historico


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Treinamento da CNN")
    p.add_argument("--epocas", type=int, default=CFG.epocas)
    p.add_argument("--lr", type=float, default=CFG.lr)
    p.add_argument("--rapido", action="store_true",
                   help="subconjunto pequeno, para demonstração em aula")
    treinar(p.parse_args())
