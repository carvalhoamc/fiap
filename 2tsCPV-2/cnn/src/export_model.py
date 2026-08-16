"""
export_model.py — ETAPA 7 (parte A): empacotar o modelo para produção
----------------------------------------------------------------------

Um checkpoint `.pt` com state_dict é ótimo para pesquisa, mas ruim para
produção: para carregá-lo é preciso ter a classe Python `CNNSimples`
disponível e idêntica à do dia do treino. Se alguém renomear um atributo, o
serviço quebra.

TorchScript resolve isso: ele serializa a ARQUITETURA E OS PESOS juntos, em um
formato que roda sem o código-fonte original — inclusive em C++ e em servidores
sem o seu repositório instalado.

    torch.jit.trace  : executa o modelo com uma entrada de exemplo e grava as
                       operações. Simples e suficiente para redes sem
                       if/for dependentes dos dados (o nosso caso).
    torch.jit.script : compila o código Python de verdade; necessário quando há
                       fluxo de controle dinâmico.

Uso:
    python src/export_model.py
    python src/export_model.py --onnx     # exporta também em ONNX (opcional)
"""

import argparse

import torch

from config import CFG, OUT_DIR
from evaluate import carregar_checkpoint
from utils import obter_dispositivo, salvar_json


def exportar_torchscript(modelo, classes, dispositivo):
    modelo.eval()
    exemplo = torch.randn(1, CFG.canais, CFG.tamanho_imagem,
                          CFG.tamanho_imagem, device=dispositivo)

    with torch.no_grad():
        modelo_scriptado = torch.jit.trace(modelo, exemplo)

    # torch.jit.freeze embute os pesos como constantes e funde Conv+BatchNorm,
    # deixando a inferência mais rápida. (Já a otimização mais agressiva,
    # torch.jit.optimize_for_inference, gera um módulo que não pode ser
    # recarregado de arquivo — se quiser usá-la, aplique-a no servidor, logo
    # após o torch.jit.load.)
    modelo_scriptado = torch.jit.freeze(modelo_scriptado)

    caminho = OUT_DIR / CFG.arq_torchscript
    modelo_scriptado.save(str(caminho))
    salvar_json(classes, OUT_DIR / CFG.arq_classes)

    # --- verificação obrigatória -----------------------------------------
    # Nunca confie em uma exportação sem comparar as saídas. Diferenças acima
    # de ~1e-4 indicam que algo mudou (ex.: modelo em modo train durante o trace).
    recarregado = torch.jit.load(str(caminho), map_location=dispositivo)
    with torch.no_grad():
        original = modelo(exemplo)
        exportado = recarregado(exemplo)
    diferenca = (original - exportado).abs().max().item()

    print(f"[ok] TorchScript salvo em {caminho}")
    print(f"[ok] Classes salvas em {OUT_DIR / CFG.arq_classes}")
    print(f"     diferença máxima original vs. exportado: {diferenca:.2e} "
          f"({'OK' if diferenca < 1e-4 else 'ATENÇÃO: divergência!'})")
    print(f"     tamanho do arquivo: {caminho.stat().st_size / 1024:.1f} KB")
    return caminho


def exportar_onnx(modelo, dispositivo):
    """ONNX = formato aberto, lido por ONNX Runtime, TensorRT, navegador, etc.

    Útil quando o time de produção não usa Python/PyTorch.
    """
    caminho = OUT_DIR / "modelo.onnx"
    exemplo = torch.randn(1, CFG.canais, CFG.tamanho_imagem,
                          CFG.tamanho_imagem, device=dispositivo)
    torch.onnx.export(
        modelo, exemplo, str(caminho),
        input_names=["entrada"], output_names=["logits"],
        # eixo 0 dinâmico: o serviço poderá enviar lotes de qualquer tamanho
        dynamic_axes={"entrada": {0: "lote"}, "logits": {0: "lote"}},
        opset_version=17,
    )
    print(f"[ok] ONNX salvo em {caminho}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exportação do modelo para produção")
    p.add_argument("--onnx", action="store_true")
    args = p.parse_args()

    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo)
    exportar_torchscript(modelo, ckpt["classes"], dispositivo)

    if args.onnx:
        try:
            exportar_onnx(modelo, dispositivo)
        except Exception as e:  # onnx nem sempre está instalado
            print(f"[aviso] falha ao exportar ONNX: {e}")
