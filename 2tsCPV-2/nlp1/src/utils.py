"""
utils.py
--------
Funções de apoio usadas por vários scripts: leitura/escrita de JSONL, hashes
estáveis, contagem de palavras e gráficos.

Por que JSONL (uma linha JSON por documento) e não um JSON gigante?

  * Você pode processar um corpus de 10 GB lendo linha a linha, sem carregar
    tudo na memória.
  * `head -1 corpus.jsonl` já mostra um documento inteiro — inspeção é trivial.
  * Um arquivo corrompido no meio não invalida as linhas anteriores.

É o formato padrão de fato para corpora de texto (o C4, o OSCAR e o The Pile
são distribuídos assim).
"""

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------
def configurar_terminal() -> None:
    """Garante que o terminal aceite os caracteres acentuados que imprimimos.

    Problema real, e a primeira coisa que quebra numa aula sobre codificação:
    no Windows, o Python escolhe a codificação da saída padrão pelo LOCALE do
    sistema (tipicamente cp1252), e **não** pela página de código do console.
    Ou seja, rodar `chcp 65001` antes não resolve — o `print` continua tentando
    codificar em cp1252 e levanta `UnicodeEncodeError` no primeiro caractere
    fora dessa tabela, derrubando o script no meio do relatório.

    `reconfigure` resolve na origem, de dentro do próprio programa, sem
    depender de variável de ambiente que alguém vai esquecer de definir.
    `errors="replace"` é a rede de segurança: se ainda assim aparecer um
    caractere impossível, ele vira "?" em vez de matar o processo. Um pipeline
    de dados nunca deve morrer por causa da FORMATAÇÃO de um log.

    É o mesmo princípio da cadeia de codificações em `readers.decodificar`:
    tente o certo, degrade com elegância, nunca trave.
    """
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # Saída redirecionada para um objeto que não suporta reconfigure
            # (pytest, algumas IDEs). Seguir sem reconfigurar é o certo aqui.
            pass


# ---------------------------------------------------------------------------
# Entrada e saída
# ---------------------------------------------------------------------------
def salvar_json(dados, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


def carregar_json(caminho: Path):
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def salvar_jsonl(registros: Iterable[dict], caminho: Path) -> int:
    """Escreve um registro por linha. Devolve quantos foram escritos."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(caminho, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def ler_jsonl(caminho: Path) -> Iterator[dict]:
    """Lê preguiçosamente (lazy): não carrega o arquivo inteiro na memória."""
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                yield json.loads(linha)


# ---------------------------------------------------------------------------
# Identidade e texto
# ---------------------------------------------------------------------------
def hash_curto(*partes: str, n: int = 12) -> str:
    """Identificador estável e reprodutível a partir de strings.

    Estável importa: se você reprocessar o mesmo corpus amanhã, o documento
    precisa receber o MESMO id, senão é impossível comparar duas execuções ou
    rastrear de onde veio um trecho que apareceu na resposta de um modelo.
    Por isso usamos hash do conteúdo/origem, e nunca um contador incremental.
    """
    bruto = "\x1f".join(partes).encode("utf-8")
    return hashlib.sha256(bruto).hexdigest()[:n]


def normalizar_para_comparacao(texto: str) -> str:
    """Forma agressivamente reduzida do texto, usada só para COMPARAR.

    Remove acentos, pontuação, maiúsculas e espaços redundantes. Dois textos
    que diferem apenas por formatação passam a ter exatamente a mesma forma —
    é isso que permite detectar a duplicata que um hash do texto cru não pega.
    Esta forma NUNCA é salva como conteúdo: ela é só uma chave.
    """
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


PALAVRA = re.compile(r"[^\W\d_]+", re.UNICODE)  # sequências de letras, sem dígitos


def palavras(texto: str) -> list[str]:
    """Tokenização simples por expressão regular.

    Suficiente para ESTATÍSTICA (contar, medir, filtrar). Não confunda com a
    tokenização de um modelo de linguagem, que é subpalavra e depende do
    vocabulário treinado.
    """
    return PALAVRA.findall(texto.lower())


def contar_palavras(texto: str) -> int:
    return len(palavras(texto))


def formatar_bytes(n: int) -> str:
    for unidade in ("B", "KB", "MB", "GB"):
        if n < 1024 or unidade == "GB":
            return f"{n:.0f} {unidade}" if unidade == "B" else f"{n:.1f} {unidade}"
        n /= 1024
    return f"{n:.1f} GB"


def truncar(texto: str, n: int = 90) -> str:
    """Versão de uma linha, para caber no terminal."""
    limpo = re.sub(r"\s+", " ", texto).strip()
    return limpo if len(limpo) <= n else limpo[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Gráficos do relatório (ETAPA 6)
# ---------------------------------------------------------------------------
def _plt():
    """Importa matplotlib com backend sem janela; devolve None se faltar."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # backend sem janela, funciona em servidor
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[aviso] matplotlib não instalado; pulando gráficos.")
        return None


def plotar_barras(contagens: dict, titulo: str, rotulo_x: str, caminho: Path,
                  cor: str = "#2563eb") -> None:
    """Gráfico de barras horizontais ordenado — o formato mais legível para
    comparar categorias com nomes longos (os motivos de rejeição)."""
    plt = _plt()
    if plt is None or not contagens:
        return

    itens = sorted(contagens.items(), key=lambda kv: kv[1])
    nomes = [k for k, _ in itens]
    valores = [v for _, v in itens]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.42 * len(nomes) + 1)))
    barras = ax.barh(nomes, valores, color=cor)
    ax.bar_label(barras, padding=3, fontsize=8)
    ax.set_xlabel(rotulo_x)
    ax.set_title(titulo)
    ax.margins(x=0.12)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Gráfico salvo em {caminho}")


def plotar_histograma(valores: list, titulo: str, rotulo_x: str, caminho: Path,
                      limiar: float | None = None) -> None:
    """Distribuição dos tamanhos dos documentos, com o limiar de corte marcado.

    Este gráfico responde a pergunta que todo filtro levanta: "meu limiar está
    cortando a cauda de lixo ou está cortando o meio da distribuição?"
    """
    plt = _plt()
    if plt is None or not valores:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valores, bins=min(40, max(5, len(valores) // 2)),
            color="#0d9488", edgecolor="white")
    if limiar is not None:
        ax.axvline(limiar, color="#dc2626", linestyle="--", linewidth=1.5)
        ax.text(limiar, ax.get_ylim()[1] * .92, f" limiar = {limiar:g}",
                color="#dc2626", fontsize=8)
    ax.set_xlabel(rotulo_x)
    ax.set_ylabel("nº de documentos")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Gráfico salvo em {caminho}")


def plotar_pizza(contagens: dict, titulo: str, caminho: Path) -> None:
    """Composição do corpus por formato de origem."""
    plt = _plt()
    if plt is None or not contagens:
        return

    itens = sorted(contagens.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie([v for _, v in itens], labels=[k for k, _ in itens],
           autopct=lambda p: f"{p:.0f}%", startangle=90,
           colors=["#2563eb", "#0d9488", "#f59e0b", "#dc2626", "#7c3aed", "#64748b"],
           textprops={"fontsize": 9})
    ax.set_title(titulo)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"[ok] Gráfico salvo em {caminho}")
