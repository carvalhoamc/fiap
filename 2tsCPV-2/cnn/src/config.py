"""
config.py
---------
Ponto único de configuração do projeto.

Por que centralizar? Porque um experimento de aprendizado de máquina só é
reprodutível se todos os "botões" (hiperparâmetros, caminhos, semente
aleatória) estiverem escritos em um lugar só, versionados junto com o código.
Espalhar números mágicos pelos scripts é a causa nº 1 de resultados que
"funcionavam ontem" e não funcionam hoje.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos: sempre relativos à raiz do projeto, nunca ao diretório de onde
# você chamou o script. parents[1] sobe de src/ para cnn/.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"        # onde o dataset bruto é baixado/armazenado
OUT_DIR = ROOT / "outputs"      # onde salvamos pesos, gráficos e métricas


@dataclass
class Config:
    # --- Dados ---------------------------------------------------------
    dataset: str = "FashionMNIST"
    tamanho_imagem: int = 28          # imagens 28x28
    canais: int = 1                   # 1 = escala de cinza, 3 = RGB
    n_classes: int = 10
    frac_validacao: float = 0.1       # 10% do treino vira validação
    num_workers: int = 0              # 0 é o mais seguro no Windows

    # Estatísticas do FashionMNIST (calculadas sobre o conjunto de treino).
    # Usadas para normalizar: x = (x - media) / desvio
    media: float = 0.2860
    desvio: float = 0.3530

    # --- Treinamento ---------------------------------------------------
    batch_size: int = 128
    epocas: int = 10
    lr: float = 1e-3                  # taxa de aprendizado do Adam
    weight_decay: float = 1e-4        # regularização L2
    paciencia: int = 3                # early stopping
    semente: int = 42

    # --- Arquitetura ---------------------------------------------------
    canais_conv: tuple = (16, 32, 64)  # filtros de cada bloco convolucional
    neuronios_fc: int = 128
    dropout: float = 0.3

    # --- Arquivos de saída ---------------------------------------------
    arq_checkpoint: str = "melhor_modelo.pt"
    arq_historico: str = "historico.json"
    arq_curvas: str = "curvas_treino.png"
    arq_confusao: str = "matriz_confusao.png"
    arq_metricas: str = "metricas_teste.json"
    arq_torchscript: str = "modelo_scriptado.pt"
    arq_classes: str = "classes.json"

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()

# Nomes legíveis das 10 classes do FashionMNIST, na ordem dos rótulos 0..9.
CLASSES = [
    "Camiseta/Top",
    "Calça",
    "Pulôver",
    "Vestido",
    "Casaco",
    "Sandália",
    "Camisa",
    "Tênis",
    "Bolsa",
    "Bota",
]
