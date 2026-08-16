"""
config.py — ETAPA 0: ponto único de configuração
-------------------------------------------------
Todos os "botões" do projeto moram aqui: caminhos, colunas, hiperparâmetros e
semente aleatória. Espalhar números mágicos pelos scripts é a causa nº 1 de
experimentos que "funcionavam ontem" e hoje não reproduzem.

Em dados tabulares, a configuração inclui algo que não existe em imagens: a
DEFINIÇÃO DAS COLUNAS. Quais são numéricas, quais são categóricas, quais devem
ser descartadas e por quê — essa é uma decisão de modelagem, não um detalhe
técnico, e por isso está versionada junto com o código.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos relativos à raiz do projeto (parents[1] sobe de src/ para mlp/),
# nunca ao diretório de onde você chamou o script.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"      # dataset bruto baixado
OUT_DIR = ROOT / "outputs"    # pesos, gráficos e métricas gerados

# ---------------------------------------------------------------------------
# ETAPA 1 — origem dos dados
# Adult / Census Income (UCI, 1994): prever se a renda anual passa de US$ 50k.
# ---------------------------------------------------------------------------
URL_DADOS = "https://archive.ics.uci.edu/static/public/2/adult.zip"

# O arquivo do UCI não tem cabeçalho: os nomes vêm do adult.names.
COLUNAS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

ALVO = "income"          # coluna original (texto)
CLASSE_POSITIVA = ">50K"  # o que chamamos de "1"
CLASSES = ["<=50K", ">50K"]

# Descartadas de propósito — cada uma por um motivo diferente:
#   fnlwgt    : peso amostral do censo, não é atributo da pessoa. Manter seria
#               dar ao modelo uma informação sobre o PROCESSO de amostragem.
#   education : redundante com education-num, que já é a mesma informação em
#               escala ordinal (1=pré-escola ... 16=doutorado).
COLUNAS_DESCARTADAS = ["fnlwgt", "education"]

COLUNAS_NUMERICAS = [
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]

# 92% das pessoas têm capital-gain = 0 e o máximo é 99.999. Sem log1p, a
# padronização é dominada por meia dúzia de valores extremos.
COLUNAS_LOG = ["capital-gain", "capital-loss"]

COLUNAS_CATEGORICAS = [
    "workclass", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
]

# O UCI marca ausente como "?" (aparece em workclass, occupation e
# native-country). Viram uma categoria própria — ver data.py.
CATEGORIA_AUSENTE = "Desconhecido"


@dataclass
class Config:
    # --- Dados ----------------------------------------------------------
    frac_validacao: float = 0.2   # 20% do treino vira validação (estratificado)
    num_workers: int = 0          # 0 é o mais seguro no Windows

    # --- Treinamento ----------------------------------------------------
    batch_size: int = 256
    epocas: int = 40
    lr: float = 1e-3              # taxa de aprendizado do Adam
    weight_decay: float = 1e-4    # regularização L2
    paciencia: int = 6            # early stopping
    semente: int = 42
    # A classe positiva é ~24% dos dados. pos_weight multiplica a perda dos
    # positivos; com "auto" usamos n_negativos/n_positivos (~3,15).
    pos_weight: str | float = "auto"

    # --- Arquitetura ----------------------------------------------------
    camadas_ocultas: tuple = (64, 32)
    dropout: float = 0.3
    usar_batchnorm: bool = True

    # --- Decisão --------------------------------------------------------
    # O modelo devolve uma PROBABILIDADE; virar "sim/não" exige um limiar.
    # 0,5 é apenas o padrão — train.py escolhe o melhor limiar na validação.
    limiar_padrao: float = 0.5

    # --- Arquivos de saída ----------------------------------------------
    arq_dados: str = "adult.zip"
    arq_checkpoint: str = "melhor_modelo.pt"
    arq_historico: str = "historico.json"
    arq_curvas: str = "curvas_treino.png"
    arq_confusao: str = "matriz_confusao.png"
    arq_roc: str = "curvas_roc_pr.png"
    arq_importancia: str = "importancia_permutacao.png"
    arq_metricas: str = "metricas_teste.json"
    arq_torchscript: str = "modelo_scriptado.pt"
    arq_preprocessador: str = "preprocessador.json"
    arq_metadados: str = "metadados.json"

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()

# Exemplo usado por predict.py e pela página de teste da API.
EXEMPLO = {
    "age": 39,
    "workclass": "State-gov",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}
