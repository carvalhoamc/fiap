"""
config.py — ETAPA 0: ponto único de configuração
-------------------------------------------------
Todos os "botões" do projeto moram aqui: caminhos, colunas, parâmetros do
`setup()` do PyCaret e semente aleatória. Espalhar números mágicos pelos
scripts é a causa nº 1 de experimentos que "funcionavam ontem" e hoje não
reproduzem.

DIFERENÇA EM RELAÇÃO À AULA DE MLP
----------------------------------
Lá, a definição das colunas alimentava um `Preprocessador` escrito à mão.
Aqui, ela alimenta um único `setup()`: as MESMAS decisões de modelagem
(o que é numérico, o que é categórico, o que descartar, como imputar, como
escalar) continuam existindo — elas só mudaram de endereço. O PyCaret não
tomou nenhuma decisão por você; ele apenas escreveu o código no seu lugar.

Ou seja: a configuração continua sendo a parte do projeto que exige o
especialista. Automatizar o "como" não dispensa o "o quê".
"""

from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos relativos à raiz do projeto (parents[1] sobe de src/ para pycaret/),
# nunca ao diretório de onde você chamou o script.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"      # dataset bruto baixado
OUT_DIR = ROOT / "outputs"    # modelos, gráficos e métricas gerados

# ---------------------------------------------------------------------------
# ETAPA 1 — origem dos dados
# Adult / Census Income (UCI, 1994): prever se a renda anual passa de US$ 50k.
# É o MESMO dataset da aula de MLP, de propósito: só assim a comparação entre
# as duas abordagens mede a abordagem, e não o dataset.
# ---------------------------------------------------------------------------
URL_DADOS = "https://archive.ics.uci.edu/static/public/2/adult.zip"

# O arquivo do UCI não tem cabeçalho: os nomes vêm do adult.names.
COLUNAS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

ALVO = "income"           # coluna original (texto)
CLASSE_POSITIVA = ">50K"  # o que chamamos de "1"
CLASSES = ["<=50K", ">50K"]
ALVO_BINARIO = "y"        # coluna 0/1 que entregamos ao PyCaret

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
# padronização é decidida por meia dúzia de valores extremos.
#
# ATENÇÃO METODOLÓGICA: log1p é uma transformação SEM ESTADO — ela não aprende
# nada dos dados, é a mesma função para qualquer linha. Por isso pode ser
# aplicada ANTES da separação treino/teste, sem risco de vazamento. Já mediana,
# média, desvio e lista de categorias APRENDEM dos dados: essas ficam dentro do
# pipeline do PyCaret, que só as ajusta no treino. Saber distinguir os dois
# tipos é o que separa "cuidado com vazamento" de superstição.
COLUNAS_LOG = ["capital-gain", "capital-loss"]

COLUNAS_CATEGORICAS = [
    "workclass", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
]

# O UCI marca ausente como "?" (aparece em workclass, occupation e
# native-country). Viram uma categoria própria, e não a moda: "não informado"
# muitas vezes É informação.
CATEGORIA_AUSENTE = "Desconhecido"


@dataclass
class Config:
    # --- Dados ----------------------------------------------------------
    frac_validacao: float = 0.2   # 20% do treino vira validação (estratificado)

    # --- setup() do PyCaret ---------------------------------------------
    semente: int = 42             # session_id: fixa TODA a aleatoriedade
    fold: int = 5                 # k da validação cruzada estratificada
    fold_strategy: str = "stratifiedkfold"
    normalize: bool = True        # padroniza os numéricos (z-score)
    normalize_method: str = "zscore"
    # PyCaret usa one-hot só até `max_encoding_ohe` categorias; acima disso ele
    # troca silenciosamente para target encoding. native-country tem 41 valores.
    # Colocamos 50 para forçar one-hot em TODAS as colunas — igual à aula de MLP,
    # que é o que torna as duas comparáveis. Ver a discussão no README (§5.2).
    max_encoding_ohe: int = 50
    # SMOTE: sintetiza positivos até equilibrar as classes. É o análogo do
    # `pos_weight` da aula de MLP — mas com um efeito colateral diferente.
    # Ligue com `python src/train.py --balancear` e compare (exercício 2.4).
    fix_imbalance: bool = False
    n_jobs: int = -1
    use_gpu: bool = False

    # --- Seleção de modelos ---------------------------------------------
    # 'dummy' entra de propósito: é a linha de base trivial ("chute sempre a
    # maioria"), e o número dele é a régua contra a qual todos os outros são
    # julgados. Um relatório que mostra 85% de acurácia sem dizer que o dummy
    # faz 76% está incompleto.
    # 'mlp' é o MLPClassifier do sklearn — a mesma família da aula anterior,
    # aqui como só mais uma linha da tabela.
    modelos_comparados: tuple = (
        "dummy", "lr", "ridge", "nb", "dt", "knn",
        "rf", "et", "ada", "gbc", "lightgbm", "xgboost", "mlp",
    )
    metrica_ordenacao: str = "AUC"   # ordena o compare_models por AUC, não por acurácia
    n_melhores: int = 3              # quantos modelos guardar do compare_models

    # --- Ajuste fino -----------------------------------------------------
    n_iter_tune: int = 20            # sorteios da busca aleatória do tune_model
    metrica_tune: str = "AUC"

    # --- Decisão ---------------------------------------------------------
    # O modelo devolve uma PROBABILIDADE; virar "sim/não" exige um limiar.
    # 0,5 é apenas o padrão — train.py escolhe o melhor limiar na validação.
    limiar_padrao: float = 0.5

    # --- Arquivos de saída ----------------------------------------------
    arq_dados: str = "adult.zip"
    arq_comparacao: str = "comparacao_modelos.csv"
    arq_ranking: str = "ranking_modelos.png"
    arq_exploracao: str = "exploracao_dados.png"
    arq_curvas: str = "curvas_roc_pr.png"
    arq_confusao: str = "matriz_confusao.png"
    arq_importancia: str = "importancia_permutacao.png"
    arq_calibracao: str = "calibracao.png"
    arq_metricas: str = "metricas_teste.json"
    arq_historico: str = "historico.json"
    arq_pipeline: str = "pipeline_renda"      # save_model acrescenta .pkl
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
