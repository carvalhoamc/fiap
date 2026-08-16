"""
data.py — ETAPAS 1 e 2 do pipeline: aquisição e pré-processamento
------------------------------------------------------------------

Em visão computacional, "pré-processar" é redimensionar e normalizar pixels.
Em dados tabulares o trabalho é bem maior, e é aqui que mora a maior parte dos
erros de um projeto real:

1. Aquisição       -> baixar e ler o CSV bruto, com os tipos certos.
2. Ausentes        -> decidir o que fazer com "?" e células vazias.
3. Categóricas     -> texto não entra em rede neural; precisa virar número.
4. Escala          -> padronizar, senão "capital-gain" (até 99.999) domina
                      "hours-per-week" (até 99) só por ser numericamente maior.
5. Particionamento -> treino / validação / teste, SEM vazamento.

REGRA DE OURO DESTA AULA
------------------------
Tudo o que é APRENDIDO dos dados (mediana para preencher, lista de categorias,
média e desvio para padronizar) tem que ser aprendido SOMENTE no conjunto de
treino, e depois APLICADO à validação e ao teste.

Calcular a média usando o dataset inteiro antes de separar é o vazamento mais
comum — e o mais silencioso — em dados tabulares. Ninguém vê erro nenhum: a
validação só fica otimista, e o modelo decepciona em produção.

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
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import (ALVO, CATEGORIA_AUSENTE, CFG, CLASSE_POSITIVA, CLASSES,
                    COLUNAS, COLUNAS_CATEGORICAS, COLUNAS_DESCARTADAS,
                    COLUNAS_LOG, COLUNAS_NUMERICAS, DATA_DIR, URL_DADOS)


# ---------------------------------------------------------------------------
# ETAPA 1 — Aquisição
# ---------------------------------------------------------------------------
def baixar_dados(destino: Path | None = None) -> Path:
    """Baixa adult.zip do repositório da UCI (só na primeira vez).

    Em projeto real esta etapa seria uma consulta ao banco, um dump do data
    warehouse ou uma API — e é normalmente onde vai a maior parte do esforço.
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


def carregar_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (treino_completo, teste) já com a coluna alvo binária 'y'.

    O UCI entrega a partição oficial pronta: adult.data (32.561 linhas) e
    adult.test (16.281). Vamos respeitá-la — usar a partição oficial é o que
    permite comparar o seu resultado com o de qualquer outro trabalho.

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

    for df in (df_treino, df_teste):
        # .str.rstrip(".") resolve o ">50K." do arquivo de teste.
        df["y"] = (df[ALVO].str.rstrip(".") == CLASSE_POSITIVA).astype(np.int64)
        df.drop(columns=[ALVO] + COLUNAS_DESCARTADAS, inplace=True)

    return df_treino.reset_index(drop=True), df_teste.reset_index(drop=True)


def separar_treino_validacao(df: pd.DataFrame, cfg=CFG):
    """Divide o treino em (treino, validação) de forma ESTRATIFICADA.

    Estratificar = manter a mesma proporção de classes nas duas partes. Com
    24% de positivos, uma divisão aleatória comum pode entregar uma validação
    com 21% e outra com 27%, e aí você compara experimentos medindo coisas
    diferentes.

    O conjunto de TESTE não aparece aqui: ele fica intocado até evaluate.py.
    """
    gerador = np.random.default_rng(cfg.semente)
    indices_val = []
    for classe in (0, 1):
        idx = df.index[df["y"] == classe].to_numpy()
        gerador.shuffle(idx)
        indices_val.append(idx[:int(len(idx) * cfg.frac_validacao)])
    indices_val = np.concatenate(indices_val)

    mascara_val = df.index.isin(indices_val)
    return df[~mascara_val].reset_index(drop=True), df[mascara_val].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ETAPA 2 — Pré-processamento
# ---------------------------------------------------------------------------
class Preprocessador:
    """Converte o DataFrame bruto em uma matriz numérica para a rede.

    Existe `sklearn.compose.ColumnTransformer` que faz isso em cinco linhas, e
    em projeto real você provavelmente o usaria. Aqui está escrito à mão por
    dois motivos pedagógicos:

      1. Você vê exatamente O QUE é aprendido do treino (medianas, categorias,
         médias e desvios) e o que é apenas aplicado.
      2. O estado cabe num JSON legível, que viaja junto com o modelo até o
         servidor. O deploy então reimplementa a transformação em ~20 linhas de
         numpy, sem depender de sklearn nem de pickle — que é frágil a versão.

    O pré-processador É PARTE DO MODELO. Modelo sem o pré-processador que o
    acompanha é um arquivo de pesos inútil.
    """

    def __init__(self):
        self.medianas: dict[str, float] = {}
        self.categorias: dict[str, list[str]] = {}
        self.media: dict[str, float] = {}
        self.desvio: dict[str, float] = {}
        self.nomes_features: list[str] = []

    # --- aprendizado (SOMENTE no treino) ----------------------------------
    def fit(self, df: pd.DataFrame) -> "Preprocessador":
        for coluna in COLUNAS_NUMERICAS:
            # Mediana e não média: é robusta a valores extremos, e
            # capital-gain tem cauda longa até 99.999.
            self.medianas[coluna] = float(df[coluna].median())

        for coluna in COLUNAS_CATEGORICAS:
            valores = df[coluna].fillna(CATEGORIA_AUSENTE).astype(str)
            categorias = sorted(valores.unique().tolist())
            if CATEGORIA_AUSENTE not in categorias:
                # Sempre existe, mesmo que não tenha aparecido no treino: em
                # produção pode chegar um cadastro com o campo em branco.
                categorias.append(CATEGORIA_AUSENTE)
            self.categorias[coluna] = categorias

        # Média e desvio são calculados DEPOIS do log1p, na mesma ordem em que
        # a transformação será aplicada. Trocar a ordem muda os números.
        numerico = self._numerico_bruto(df)
        for coluna in COLUNAS_NUMERICAS:
            self.media[coluna] = float(numerico[coluna].mean())
            desvio = float(numerico[coluna].std())
            self.desvio[coluna] = desvio if desvio > 1e-8 else 1.0

        self.nomes_features = list(COLUNAS_NUMERICAS) + [
            f"{coluna}={categoria}"
            for coluna in COLUNAS_CATEGORICAS
            for categoria in self.categorias[coluna]
        ]
        return self

    # --- aplicação (treino, validação, teste e produção) ------------------
    def _numerico_bruto(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preenche ausentes e aplica log1p, sem padronizar ainda."""
        saida = pd.DataFrame(index=df.index)
        for coluna in COLUNAS_NUMERICAS:
            valores = pd.to_numeric(df[coluna], errors="coerce").fillna(self.medianas[coluna])
            if coluna in COLUNAS_LOG:
                # log1p(x) = log(1+x): comprime a cauda longa e aceita zero.
                valores = np.log1p(valores.clip(lower=0))
            saida[coluna] = valores.astype(float)
        return saida

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrame bruto -> matriz float32 (n_amostras, n_features)."""
        numerico = self._numerico_bruto(df)
        blocos = [((numerico[c] - self.media[c]) / self.desvio[c]).to_numpy()[:, None]
                  for c in COLUNAS_NUMERICAS]

        for coluna in COLUNAS_CATEGORICAS:
            categorias = self.categorias[coluna]
            valores = df[coluna].fillna(CATEGORIA_AUSENTE).astype(str).to_numpy()
            bloco = np.zeros((len(df), len(categorias)), dtype=float)
            posicao = {c: i for i, c in enumerate(categorias)}
            for linha, valor in enumerate(valores):
                indice = posicao.get(valor)
                if indice is not None:
                    bloco[linha, indice] = 1.0
                # Categoria nunca vista no treino (ex.: um país novo) fica com
                # a linha toda zerada. É uma decisão consciente: o modelo trata
                # como "nenhuma das conhecidas". A alternativa seria mapear para
                # Desconhecido — vale discutir qual é melhor no seu domínio.
            blocos.append(bloco)

        return np.concatenate(blocos, axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # --- serialização: o estado viaja com o modelo ------------------------
    def to_dict(self) -> dict:
        return {
            "versao": 1,
            "colunas_numericas": list(COLUNAS_NUMERICAS),
            "colunas_log": list(COLUNAS_LOG),
            "colunas_categoricas": list(COLUNAS_CATEGORICAS),
            "categoria_ausente": CATEGORIA_AUSENTE,
            "medianas": self.medianas,
            "categorias": self.categorias,
            "media": self.media,
            "desvio": self.desvio,
            "nomes_features": self.nomes_features,
        }

    @classmethod
    def from_dict(cls, estado: dict) -> "Preprocessador":
        p = cls()
        p.medianas = {k: float(v) for k, v in estado["medianas"].items()}
        p.categorias = {k: list(v) for k, v in estado["categorias"].items()}
        p.media = {k: float(v) for k, v in estado["media"].items()}
        p.desvio = {k: float(v) for k, v in estado["desvio"].items()}
        p.nomes_features = list(estado["nomes_features"])
        return p

    @property
    def n_features(self) -> int:
        return len(self.nomes_features)


# ---------------------------------------------------------------------------
# Montagem dos DataLoaders
# ---------------------------------------------------------------------------
def _para_loader(X: np.ndarray, y: np.ndarray, embaralhar: bool, cfg=CFG) -> DataLoader:
    conjunto = TensorDataset(torch.from_numpy(X),
                             torch.from_numpy(y.astype(np.float32)))
    return DataLoader(conjunto, batch_size=cfg.batch_size, shuffle=embaralhar,
                      num_workers=cfg.num_workers)


def obter_dataloaders(cfg=CFG, rapido: bool = False):
    """Devolve (loader_treino, loader_val, loader_teste, preprocessador).

    A ordem das operações é o ponto da etapa:
        separar  ->  ajustar o preprocessador SÓ no treino  ->  aplicar nos três.
    Inverter isso é vazamento.
    """
    df_treino_completo, df_teste = carregar_dataframes()
    df_treino, df_val = separar_treino_validacao(df_treino_completo, cfg)

    if rapido:
        # Modo demonstração em sala: roda em segundos.
        df_treino = df_treino.sample(n=min(5000, len(df_treino)),
                                     random_state=cfg.semente).reset_index(drop=True)
        df_val = df_val.sample(n=min(1500, len(df_val)),
                               random_state=cfg.semente).reset_index(drop=True)
        df_teste = df_teste.sample(n=min(3000, len(df_teste)),
                                   random_state=cfg.semente).reset_index(drop=True)

    preprocessador = Preprocessador().fit(df_treino)     # <- só o treino!

    X_treino = preprocessador.transform(df_treino)
    X_val = preprocessador.transform(df_val)
    X_teste = preprocessador.transform(df_teste)

    # shuffle=True apenas no treino: a ordem dos lotes deve mudar a cada época
    # para o gradiente estocástico não ficar viciado na sequência dos dados.
    return (
        _para_loader(X_treino, df_treino["y"].to_numpy(), True, cfg),
        _para_loader(X_val, df_val["y"].to_numpy(), False, cfg),
        _para_loader(X_teste, df_teste["y"].to_numpy(), False, cfg),
        preprocessador,
    )


# ---------------------------------------------------------------------------
# Utilitário didático: OLHAR os dados antes de treinar
# ---------------------------------------------------------------------------
def salvar_exploracao(df: pd.DataFrame, caminho: Path) -> None:
    """Quatro painéis que respondem perguntas que todo projeto tabular deve fazer.

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
    contagem = df["y"].value_counts().sort_index()
    eixos[0, 0].bar([CLASSES[0], CLASSES[1]], contagem.values,
                    color=["tab:blue", "tab:red"])
    for i, v in enumerate(contagem.values):
        eixos[0, 0].text(i, v, f"{v}\n{v / len(df):.1%}", ha="center", va="bottom")
    eixos[0, 0].set_title("Distribuição do alvo (classe positiva é a minoria)")
    eixos[0, 0].set_ylim(0, contagem.max() * 1.2)

    # (2) Uma variável contínua separa as classes?
    for classe, cor in ((0, "tab:blue"), (1, "tab:red")):
        eixos[0, 1].hist(df.loc[df["y"] == classe, "age"], bins=30, alpha=.6,
                         color=cor, label=CLASSES[classe], density=True)
    eixos[0, 1].set_title("Idade por classe")
    eixos[0, 1].set_xlabel("idade")
    eixos[0, 1].legend()

    # (3) Efeito quase monotônico: escolaridade x renda.
    taxa = df.groupby("education-num")["y"].mean()
    eixos[1, 0].plot(taxa.index, taxa.values, "o-", color="tab:green")
    eixos[1, 0].set_title("Proporção de >50K por anos de estudo")
    eixos[1, 0].set_xlabel("education-num")
    eixos[1, 0].set_ylabel("P(>50K)")
    eixos[1, 0].grid(alpha=.3)

    # (4) Categórica forte — e o gancho para a discussão de viés.
    taxa_civil = df.groupby("marital-status")["y"].mean().sort_values()
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
    # Serve para conferir tamanhos, ausentes, balanceamento e o resultado da
    # transformação — ANTES de gastar tempo treinando.
    p = argparse.ArgumentParser(description="Inspeção do dataset")
    p.add_argument("--rapido", action="store_true", help="usa subconjunto pequeno")
    args = p.parse_args()

    from config import OUT_DIR

    df_treino_completo, df_teste = carregar_dataframes()
    df_treino, df_val = separar_treino_validacao(df_treino_completo)

    print("=" * 70)
    print(f"Treino    : {len(df_treino):>6} linhas | {df_treino['y'].mean():.1%} da classe >50K")
    print(f"Validação : {len(df_val):>6} linhas | {df_val['y'].mean():.1%} da classe >50K")
    print(f"Teste     : {len(df_teste):>6} linhas | {df_teste['y'].mean():.1%} da classe >50K")
    print("=" * 70)

    ausentes = df_treino.isna().sum()
    ausentes = ausentes[ausentes > 0]
    print("\nValores ausentes no treino (marcados como '?' no arquivo original):")
    for coluna, n in ausentes.items():
        print(f"  {coluna:<18} {n:>5}  ({n / len(df_treino):.1%})")

    print("\nCardinalidade das colunas categóricas:")
    for coluna in COLUNAS_CATEGORICAS:
        print(f"  {coluna:<18} {df_treino[coluna].nunique():>3} categorias distintas")

    prep = Preprocessador().fit(df_treino)
    X_treino = prep.transform(df_treino)
    X_val = prep.transform(df_val)

    print(f"\nApós o pré-processamento: {X_treino.shape[1]} colunas numéricas")
    print(f"  {len(COLUNAS_NUMERICAS)} numéricas padronizadas + "
          f"{X_treino.shape[1] - len(COLUNAS_NUMERICAS)} colunas one-hot")
    print(f"  matriz de treino    : {X_treino.shape}")
    print(f"  matriz de validação : {X_val.shape}")

    print("\nConferência da padronização (deve dar média ~0 e desvio ~1 NO TREINO):")
    for i, coluna in enumerate(COLUNAS_NUMERICAS):
        print(f"  {coluna:<16} treino: média {X_treino[:, i].mean():+.3f} "
              f"desvio {X_treino[:, i].std():.3f}   |   "
              f"validação: média {X_val[:, i].mean():+.3f} desvio {X_val[:, i].std():.3f}")
    print("  (na validação NÃO dá exatamente 0 e 1 — e isso está certo: as "
          "estatísticas\n   vieram do treino. Se desse exatamente, haveria vazamento.)")

    print(f"\nPrimeiras 8 features: {prep.nomes_features[:8]}")
    salvar_exploracao(df_treino, OUT_DIR / "exploracao_dados.png")
