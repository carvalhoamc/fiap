"""
data.py — ETAPA 1 do pipeline: aquisição e particionamento
------------------------------------------------------------

Compare este arquivo com o `data.py` da aula de MLP. Lá, a classe
`Preprocessador` custava 77 linhas de código (imputação, one-hot, padronização,
serialização) — mais 32 reescritas em numpy dentro do servidor, uma segunda
implementação da mesma regra. Aqui esse trabalho todo cabe em 41 linhas e sem
duplicação, porque o pré-processamento migrou para uma única chamada de
`setup()` — ver `setup_experimento.py`.

O que NÃO migrou, e por isso continua aqui:

  1. Baixar e ler o arquivo, com as armadilhas do formato.
  2. Construir o alvo binário.
  3. Separar treino / validação / teste — a decisão metodológica que nenhum
     AutoML toma por você de forma satisfatória.
  4. Transformações SEM ESTADO (log1p), que podem ocorrer antes da separação
     porque não aprendem nada dos dados.

REGRA DE OURO — a mesma da aula anterior
----------------------------------------
Tudo o que é APRENDIDO dos dados (mediana, lista de categorias, média e desvio)
tem que ser aprendido SOMENTE no treino. A diferença é que agora quem garante
isso é o pipeline do PyCaret, e não você. Isso é um ganho de segurança — desde
que você entregue ao `setup()` apenas o conjunto de treino, que é exatamente o
que este arquivo prepara.

Uso:
    python src/data.py
    python src/data.py --rapido
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import (ALVO, ALVO_BINARIO, CATEGORIA_AUSENTE, CFG,
                    CLASSE_POSITIVA, CLASSES, COLUNAS, COLUNAS_CATEGORICAS,
                    COLUNAS_DESCARTADAS, COLUNAS_LOG, COLUNAS_NUMERICAS,
                    DATA_DIR, URL_DADOS)


# ---------------------------------------------------------------------------
# ETAPA 1 — Aquisição
# ---------------------------------------------------------------------------
def baixar_dados(destino: Path | None = None) -> Path:
    """Baixa adult.zip do repositório da UCI (só na primeira vez).

    Em projeto real esta etapa seria uma consulta ao banco, um dump do data
    warehouse ou uma API — e é normalmente onde vai a maior parte do esforço.
    Note que nenhum AutoML do mundo faz esta parte por você.
    """
    destino = destino or (DATA_DIR / CFG.arq_dados)
    if destino.exists():
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {URL_DADOS} ...")
    with urllib.request.urlopen(URL_DADOS, timeout=120) as resposta:
        destino.write_bytes(resposta.read())
    print(f"[ok] salvo em {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
    return destino


def aplicar_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """Comprime a cauda longa de capital-gain / capital-loss.

    POR QUE ISSO PODE ACONTECER ANTES DA SEPARAÇÃO
    ----------------------------------------------
    log1p(x) = log(1 + x) é uma função fixa: não estima média, nem mediana, nem
    quantil. Aplicada linha a linha, ela dá o mesmo resultado independentemente
    das outras linhas do conjunto. Logo, não há como informação do teste vazar
    para o treino.

    Já `normalize=True` do setup() calcula média e desvio — esses APRENDEM, e
    por isso ficam dentro do pipeline, ajustados só no treino.

    Esta distinção (transformação com estado x sem estado) é a forma correta de
    raciocinar sobre vazamento. Sem ela, o aluno decora "nunca transforme antes
    de separar", o que é falso, ou transforma tudo antes, o que é perigoso.
    """
    df = df.copy()
    for coluna in COLUNAS_LOG:
        df[coluna] = np.log1p(pd.to_numeric(df[coluna], errors="coerce").clip(lower=0))
    return df


def carregar_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (treino_completo, teste) já com a coluna alvo binária 'y'.

    O UCI entrega a partição oficial pronta: adult.data (32.561 linhas) e
    adult.test (16.281). Vamos respeitá-la — usar a partição oficial é o que
    permite comparar o seu resultado com o de qualquer outro trabalho, inclusive
    com o MLP da aula anterior.

    Duas armadilhas do arquivo, ambas clássicas em dados públicos:
      * adult.test tem uma linha de comentário no topo (skiprows=1);
      * nele o rótulo vem com um ponto final: ">50K." em vez de ">50K".
    Quem não trata a segunda acaba com 100% de "<=50K" no teste e não entende
    por que a acurácia deu exatamente a prevalência.
    """
    caminho_zip = baixar_dados()
    with zipfile.ZipFile(caminho_zip) as z:
        bruto_treino = z.read("adult.data")
        bruto_teste = z.read("adult.test")

    leitura = dict(names=COLUNAS, sep=",", skipinitialspace=True, na_values="?")
    df_treino = pd.read_csv(io.BytesIO(bruto_treino), **leitura)
    df_teste = pd.read_csv(io.BytesIO(bruto_teste), skiprows=1, **leitura)

    saida = []
    for df in (df_treino, df_teste):
        # .str.rstrip(".") resolve o ">50K." do arquivo de teste.
        df[ALVO_BINARIO] = (df[ALVO].str.rstrip(".") == CLASSE_POSITIVA).astype(int)
        df = df.drop(columns=[ALVO] + COLUNAS_DESCARTADAS)
        # Ausentes categóricos viram uma categoria explícita ANTES do setup().
        # Poderíamos deixar para o `categorical_imputation` do PyCaret; fazemos
        # aqui porque queremos "Desconhecido" como um valor de verdade, com sua
        # própria coluna one-hot, e não a moda substituindo o dado faltante.
        for coluna in COLUNAS_CATEGORICAS:
            df[coluna] = df[coluna].fillna(CATEGORIA_AUSENTE).astype(str)
        saida.append(aplicar_log1p(df).reset_index(drop=True))

    return saida[0], saida[1]


def separar_treino_validacao(df: pd.DataFrame, cfg=CFG):
    """Divide o treino em (treino, validação) de forma ESTRATIFICADA.

    Estratificar = manter a mesma proporção de classes nas duas partes. Com
    24% de positivos, uma divisão aleatória comum pode entregar uma validação
    com 21% e outra com 27%, e aí você compara experimentos medindo coisas
    diferentes.

    "MAS O PYCARET NÃO FAZ ISSO SOZINHO?"
    -------------------------------------
    Faz: `setup(train_size=0.8)` separa um hold-out automaticamente, e a
    validação cruzada de k dobras acontece dentro do treino. Ainda assim
    fazemos a separação à mão por dois motivos:

      * O teste tem que ser a partição OFICIAL do UCI, não uma fatia aleatória
        — senão o número não é comparável com a literatura nem com a aula de MLP.
      * Queremos um conjunto de validação nomeado e estável para escolher o
        LIMIAR de decisão. Escolher limiar dentro das dobras da validação
        cruzada é possível, mas embaralha duas coisas que o aluno precisa ver
        separadas.

    O conjunto de TESTE não aparece aqui: ele fica intocado até evaluate.py.
    """
    gerador = np.random.default_rng(cfg.semente)
    indices_val = []
    for classe in (0, 1):
        idx = df.index[df[ALVO_BINARIO] == classe].to_numpy()
        gerador.shuffle(idx)
        indices_val.append(idx[:int(len(idx) * cfg.frac_validacao)])
    indices_val = np.concatenate(indices_val)

    mascara_val = df.index.isin(indices_val)
    return (df[~mascara_val].reset_index(drop=True),
            df[mascara_val].reset_index(drop=True))


def obter_particoes(cfg=CFG, rapido: bool = False):
    """Devolve (df_treino, df_validacao, df_teste), prontos para o setup().

    A ordem das operações é o ponto da etapa:
        carregar -> log1p (sem estado) -> separar -> só então setup() no treino.
    Chamar setup() no dataframe completo é o vazamento clássico desta aula.
    """
    df_treino_completo, df_teste = carregar_dataframes()
    df_treino, df_val = separar_treino_validacao(df_treino_completo, cfg)

    if rapido:
        # Modo demonstração em sala: o compare_models roda em ~1 minuto.
        df_treino = df_treino.sample(n=min(6000, len(df_treino)),
                                     random_state=cfg.semente).reset_index(drop=True)
        df_val = df_val.sample(n=min(2000, len(df_val)),
                               random_state=cfg.semente).reset_index(drop=True)
        df_teste = df_teste.sample(n=min(4000, len(df_teste)),
                                   random_state=cfg.semente).reset_index(drop=True)

    return df_treino, df_val, df_teste


# ---------------------------------------------------------------------------
# Utilitário didático: OLHAR os dados antes de treinar
# ---------------------------------------------------------------------------
def salvar_exploracao(df: pd.DataFrame, caminho: Path) -> None:
    """Quatro painéis que respondem perguntas que todo projeto tabular deve fazer.

    O PyCaret tem `eda()` e vários `plot_model(...)`, e você deveria usá-los.
    Mas um gráfico feito à mão obriga a decidir O QUE olhar — e é essa decisão,
    não a biblioteca, que impede um projeto de treinar com dado errado.

    Modelo alimentado com dado errado não reclama: ele aprende a coisa errada
    com toda a confiança do mundo.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando exploração.")
        return

    fig, eixos = plt.subplots(2, 2, figsize=(12, 8))

    # (1) Desbalanceamento: quanto vale "chutar sempre a maioria"?
    contagem = df[ALVO_BINARIO].value_counts().sort_index()
    eixos[0, 0].bar([CLASSES[0], CLASSES[1]], contagem.values,
                    color=["tab:blue", "tab:red"])
    for i, v in enumerate(contagem.values):
        eixos[0, 0].text(i, v, f"{v}\n{v / len(df):.1%}", ha="center", va="bottom")
    eixos[0, 0].set_title("Distribuição do alvo (classe positiva é a minoria)")
    eixos[0, 0].set_ylim(0, contagem.max() * 1.2)

    # (2) Uma variável contínua separa as classes?
    for classe, cor in ((0, "tab:blue"), (1, "tab:red")):
        eixos[0, 1].hist(df.loc[df[ALVO_BINARIO] == classe, "age"], bins=30,
                         alpha=.6, color=cor, label=CLASSES[classe], density=True)
    eixos[0, 1].set_title("Idade por classe")
    eixos[0, 1].set_xlabel("idade")
    eixos[0, 1].legend()

    # (3) Efeito quase monotônico: escolaridade x renda.
    taxa = df.groupby("education-num")[ALVO_BINARIO].mean()
    eixos[1, 0].plot(taxa.index, taxa.values, "o-", color="tab:green")
    eixos[1, 0].set_title("Proporção de >50K por anos de estudo")
    eixos[1, 0].set_xlabel("education-num")
    eixos[1, 0].set_ylabel("P(>50K)")
    eixos[1, 0].grid(alpha=.3)

    # (4) Categórica forte — e o gancho para a discussão de viés.
    taxa_civil = df.groupby("marital-status")[ALVO_BINARIO].mean().sort_values()
    eixos[1, 1].barh(taxa_civil.index, taxa_civil.values, color="tab:purple")
    eixos[1, 1].set_title("Proporção de >50K por estado civil")
    eixos[1, 1].set_xlabel("P(>50K)")

    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Exploração salva em {caminho}")


if __name__ == "__main__":
    # Rode:  python src/data.py
    # Serve para conferir tamanhos, ausentes, balanceamento e cardinalidade —
    # ANTES de gastar tempo comparando doze modelos.
    p = argparse.ArgumentParser(description="Inspeção do dataset")
    p.add_argument("--rapido", action="store_true", help="usa subconjunto pequeno")
    args = p.parse_args()

    from config import OUT_DIR

    # Recarregamos o bruto só para contar os ausentes ANTES de preenchê-los.
    caminho_zip = baixar_dados()
    with zipfile.ZipFile(caminho_zip) as z:
        bruto = z.read("adult.data")
    df_bruto = pd.read_csv(io.BytesIO(bruto), names=COLUNAS, sep=",",
                           skipinitialspace=True, na_values="?")

    df_treino, df_val, df_teste = obter_particoes(rapido=args.rapido)

    print("=" * 72)
    print(f"Treino    : {len(df_treino):>6} linhas | "
          f"{df_treino[ALVO_BINARIO].mean():.1%} da classe >50K")
    print(f"Validação : {len(df_val):>6} linhas | "
          f"{df_val[ALVO_BINARIO].mean():.1%} da classe >50K")
    print(f"Teste     : {len(df_teste):>6} linhas | "
          f"{df_teste[ALVO_BINARIO].mean():.1%} da classe >50K")
    print("=" * 72)

    ausentes = df_bruto.isna().sum()
    ausentes = ausentes[ausentes > 0]
    print("\nValores ausentes no arquivo original (marcados como '?'):")
    for coluna, n in ausentes.items():
        print(f"  {coluna:<18} {n:>5}  ({n / len(df_bruto):.1%})")
    print(f"  -> preenchidos com a categoria '{CATEGORIA_AUSENTE}' em data.py")

    print("\nCardinalidade das colunas categóricas (no treino):")
    total_ohe = 0
    for coluna in COLUNAS_CATEGORICAS:
        n = df_treino[coluna].nunique()
        total_ohe += n
        print(f"  {coluna:<18} {n:>3} categorias distintas")
    print(f"  -> {total_ohe} colunas one-hot + {len(COLUNAS_NUMERICAS)} numéricas "
          f"= {total_ohe + len(COLUNAS_NUMERICAS)} features (o setup() confirma)")

    print("\nEfeito do log1p (transformação SEM ESTADO, aplicada antes de separar):")
    bruto_gain = pd.to_numeric(df_bruto["capital-gain"], errors="coerce")
    print(f"  capital-gain bruto : min {bruto_gain.min():.0f} | "
          f"máx {bruto_gain.max():.0f} | {(bruto_gain == 0).mean():.1%} de zeros")
    print(f"  capital-gain log1p : min {df_treino['capital-gain'].min():.2f} | "
          f"máx {df_treino['capital-gain'].max():.2f}")

    salvar_exploracao(df_treino, OUT_DIR / CFG.arq_exploracao)
