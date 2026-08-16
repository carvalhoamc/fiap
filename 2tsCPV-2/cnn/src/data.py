"""
data.py — ETAPAS 1 e 2 do pipeline: aquisição e pré-processamento
------------------------------------------------------------------

Conceitos trabalhados aqui:

1. Aquisição      -> baixar/organizar o dataset bruto em disco.
2. Particionamento -> treino / validação / teste, SEM vazamento de dados.
3. Transformações  -> converter imagem PIL em tensor, normalizar e aumentar.
4. DataLoader      -> entregar os dados em lotes (batches) para a GPU/CPU.

Regra de ouro: aumento de dados (data augmentation) só no TREINO.
Validação e teste têm que ser sempre a mesma coisa, senão a métrica oscila
por motivo aleatório e você não sabe se o modelo melhorou ou teve sorte.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import CFG, DATA_DIR, CLASSES


# ---------------------------------------------------------------------------
# 2. Pré-processamento
# ---------------------------------------------------------------------------
def construir_transformacoes(cfg=CFG):
    """Devolve (transform_treino, transform_avaliacao).

    ToTensor()  : converte PIL [0,255] em tensor float [0,1] no formato
                  (C, H, W) — atenção: PyTorch usa canal-primeiro, ao
                  contrário do OpenCV/Matplotlib que usam (H, W, C).
    Normalize() : centraliza os dados, (x - média) / desvio. Redes convergem
                  muito mais rápido com entradas de média ~0 e desvio ~1.
    Augmentation: cria variações plausíveis da mesma peça de roupa. É a forma
                  mais barata de combater overfitting: mais dados de graça.
    """
    transform_treino = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),   # roupa espelhada continua sendo a mesma roupa
        transforms.RandomCrop(cfg.tamanho_imagem, padding=2),  # pequenos deslocamentos
        transforms.RandomRotation(degrees=8),     # pequenas rotações
        transforms.ToTensor(),
        transforms.Normalize((cfg.media,), (cfg.desvio,)),
    ])

    transform_avaliacao = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg.media,), (cfg.desvio,)),
    ])
    return transform_treino, transform_avaliacao


# ---------------------------------------------------------------------------
# 1. Aquisição + 3. Particionamento
# ---------------------------------------------------------------------------
def obter_dataloaders(cfg=CFG, rapido: bool = False):
    """Baixa o FashionMNIST e devolve os três DataLoaders + nomes das classes.

    Sobre o split treino/validação:
    O torchvision entrega apenas 'train' (60.000) e 'test' (10.000). O conjunto
    de teste é sagrado: só pode ser olhado UMA vez, no final. Portanto tiramos
    a validação de dentro do treino.

    Detalhe que quase todo iniciante erra: se você usar random_split direto,
    treino e validação compartilham o MESMO objeto dataset e, portanto, a mesma
    transformação — sua validação ficaria com augmentation aleatório. A solução
    abaixo é carregar o dataset duas vezes (uma com cada transform) e aplicar
    os mesmos índices em cada cópia via Subset.
    """
    t_treino, t_aval = construir_transformacoes(cfg)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base_treino_aug = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=t_treino)
    base_treino_limpo = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=t_aval)
    conjunto_teste = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=t_aval)

    n_total = len(base_treino_aug)
    gerador = torch.Generator().manual_seed(cfg.semente)
    indices = torch.randperm(n_total, generator=gerador).tolist()

    n_val = int(n_total * cfg.frac_validacao)
    idx_val, idx_treino = indices[:n_val], indices[n_val:]

    if rapido:
        # Modo demonstração em sala: subconjunto pequeno, roda em segundos.
        idx_treino, idx_val = idx_treino[:6000], idx_val[:1000]
        conjunto_teste = Subset(conjunto_teste, list(range(2000)))

    conjunto_treino = Subset(base_treino_aug, idx_treino)
    conjunto_val = Subset(base_treino_limpo, idx_val)

    args_comuns = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                       pin_memory=torch.cuda.is_available())

    # shuffle=True só no treino: a ordem dos lotes deve mudar a cada época para
    # que o gradiente estocástico não fique viciado na sequência dos dados.
    carregador_treino = DataLoader(conjunto_treino, shuffle=True, drop_last=False, **args_comuns)
    carregador_val = DataLoader(conjunto_val, shuffle=False, **args_comuns)
    carregador_teste = DataLoader(conjunto_teste, shuffle=False, **args_comuns)

    return carregador_treino, carregador_val, carregador_teste, CLASSES


# ---------------------------------------------------------------------------
# Utilitário didático: inspecionar os dados ANTES de treinar
# ---------------------------------------------------------------------------
def salvar_amostra(carregador, classes, caminho: Path, n: int = 16) -> None:
    """Salva um mosaico com n imagens do lote — sempre olhe seus dados!

    Modelo que treina com dados errados não reclama: ele simplesmente aprende
    a coisa errada com 99% de acurácia.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando amostra.")
        return

    imagens, rotulos = next(iter(carregador))
    imagens = imagens[:n] * CFG.desvio + CFG.media  # desfaz a normalização p/ visualizar

    linhas = int(n ** 0.5)
    colunas = (n + linhas - 1) // linhas
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 1.6, linhas * 1.8))
    for i, ax in enumerate(eixos.ravel()):
        if i < n:
            ax.imshow(imagens[i].squeeze().numpy(), cmap="gray")
            ax.set_title(classes[rotulos[i]], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Amostra salva em {caminho}")


if __name__ == "__main__":
    # Rode:  python src/data.py
    # Serve para conferir tamanhos, formatos e visualizar um lote.
    p = argparse.ArgumentParser(description="Inspeção do dataset")
    p.add_argument("--rapido", action="store_true", help="usa subconjunto pequeno")
    args = p.parse_args()

    from config import OUT_DIR

    tr, va, te, classes = obter_dataloaders(rapido=args.rapido)

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Treino    : {len(tr.dataset):>6} imagens | {len(tr)} lotes")
    print(f"Validação : {len(va.dataset):>6} imagens | {len(va)} lotes")
    print(f"Teste     : {len(te.dataset):>6} imagens | {len(te)} lotes")

    x, y = next(iter(tr))
    print(f"\nFormato do lote x: {tuple(x.shape)}  (N, C, H, W)")
    print(f"Formato do lote y: {tuple(y.shape)}  dtype={y.dtype}")
    print(f"Faixa de valores após normalizar: [{x.min():.2f}, {x.max():.2f}]")

    salvar_amostra(tr, classes, OUT_DIR / "amostra_treino.png")
