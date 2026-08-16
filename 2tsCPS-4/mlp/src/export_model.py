"""
export_model.py — ETAPA 7 (parte A): empacotar o modelo para produção
----------------------------------------------------------------------

Um checkpoint `.pt` com state_dict é ótimo para pesquisa e ruim para produção:
para carregá-lo é preciso ter a classe Python `MLP` disponível e idêntica à do
dia do treino. Se alguém renomear um atributo, o serviço quebra.

TorchScript resolve isso: serializa ARQUITETURA E PESOS juntos, num formato
que roda sem o código-fonte original — inclusive em C++.

    torch.jit.trace  : executa o modelo com uma entrada de exemplo e grava as
                       operações. Simples e suficiente para redes sem if/for
                       dependentes dos dados (o nosso caso).
    torch.jit.script : compila o código Python de verdade; necessário quando há
                       fluxo de controle dinâmico.

MAS — e este é O ponto da aula em dados tabulares — o modelo sozinho não serve
para nada. Ele espera 95 números padronizados numa ordem específica. Quem
converte {"age": 39, "occupation": "Adm-clerical", ...} nesses 95 números é o
PRÉ-PROCESSADOR. Por isso o pacote de produção tem três arquivos:

    modelo_scriptado.pt   pesos + arquitetura
    preprocessador.json   medianas, categorias, médias e desvios do TREINO
    metadados.json        limiar de decisão, nomes das features, versão

Esses três, juntos, são "o modelo". Publicar só o primeiro é o erro nº 1 de
deploy em projetos tabulares.

Uso:
    python src/export_model.py
    python src/export_model.py --onnx     # opcional: formato aberto ONNX
"""

import argparse

import torch

from config import CFG, CLASSES, OUT_DIR
from evaluate import carregar_checkpoint
from utils import obter_dispositivo, salvar_json


def exportar_torchscript(modelo, ckpt, dispositivo):
    modelo.eval()   # OBRIGATÓRIO antes do trace: em modo treino, o dropout
                    # ficaria gravado no grafo e a inferência sairia aleatória.
    n_features = ckpt["hiperparametros"]["n_features"]
    exemplo = torch.randn(4, n_features, device=dispositivo)

    with torch.no_grad():
        modelo_scriptado = torch.jit.trace(modelo, exemplo)

    # freeze embute os pesos como constantes e funde operações (por exemplo
    # Linear + BatchNorm), deixando a inferência mais rápida.
    modelo_scriptado = torch.jit.freeze(modelo_scriptado)

    caminho = OUT_DIR / CFG.arq_torchscript
    modelo_scriptado.save(str(caminho))

    salvar_json(ckpt["preprocessador"], OUT_DIR / CFG.arq_preprocessador)
    salvar_json({
        "versao_modelo": "1.0.0",
        "descricao": "MLP binário — prevê renda anual acima de US$ 50k (Adult/UCI 1994)",
        "classes": CLASSES,
        "limiar": ckpt.get("limiar", CFG.limiar_padrao),
        "n_features": n_features,
        "nomes_features": ckpt["preprocessador"]["nomes_features"],
        "epoca_treino": ckpt["epoca"],
        "auc_validacao": ckpt["val_auc"],
    }, OUT_DIR / CFG.arq_metadados)

    # --- verificação obrigatória -----------------------------------------
    # Nunca confie em uma exportação sem comparar as saídas. Diferenças acima
    # de ~1e-4 indicam que algo mudou entre o modelo original e o exportado.
    recarregado = torch.jit.load(str(caminho), map_location=dispositivo)
    with torch.no_grad():
        original = modelo(exemplo)
        exportado = recarregado(exemplo)
    diferenca = (original - exportado).abs().max().item()

    print(f"[ok] TorchScript salvo em {caminho}")
    print(f"[ok] Pré-processador salvo em {OUT_DIR / CFG.arq_preprocessador}")
    print(f"[ok] Metadados salvos em {OUT_DIR / CFG.arq_metadados}")
    print(f"     limiar de decisão publicado: {ckpt.get('limiar', CFG.limiar_padrao):.2f}")
    print(f"     diferença máxima original vs. exportado: {diferenca:.2e} "
          f"({'OK' if diferenca < 1e-4 else 'ATENÇÃO: divergência!'})")
    print(f"     tamanho do modelo: {caminho.stat().st_size / 1024:.1f} KB")
    return caminho


def exportar_onnx(modelo, ckpt, dispositivo):
    """ONNX = formato aberto, lido por ONNX Runtime, TensorRT, navegador etc.

    Útil quando o time de produção não usa Python/PyTorch.
    """
    caminho = OUT_DIR / "modelo.onnx"
    exemplo = torch.randn(1, ckpt["hiperparametros"]["n_features"], device=dispositivo)
    torch.onnx.export(
        modelo, exemplo, str(caminho),
        input_names=["entrada"], output_names=["logito"],
        # eixo 0 dinâmico: o serviço poderá enviar lotes de qualquer tamanho
        dynamic_axes={"entrada": {0: "lote"}, "logito": {0: "lote"}},
        opset_version=17,
    )
    print(f"[ok] ONNX salvo em {caminho}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exportação do modelo para produção")
    p.add_argument("--onnx", action="store_true")
    args = p.parse_args()

    dispositivo = obter_dispositivo()
    modelo, ckpt = carregar_checkpoint(dispositivo)
    exportar_torchscript(modelo, ckpt, dispositivo)

    if args.onnx:
        try:
            exportar_onnx(modelo, ckpt, dispositivo)
        except Exception as e:   # onnx nem sempre está instalado
            print(f"[aviso] falha ao exportar ONNX: {e}")
