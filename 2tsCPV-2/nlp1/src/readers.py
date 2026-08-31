"""
readers.py — ETAPA 1 do pipeline: aquisição (um leitor por formato)
--------------------------------------------------------------------

Aqui mora a parte do trabalho que ninguém mostra nos tutoriais e que consome a
maior parte do tempo real de um projeto de PLN: **transformar arquivos
bagunçados em texto com procedência**.

Cada formato tem uma patologia própria. Saber qual é a de cada um é metade do
trabalho:

  TXT  -> CODIFICAÇÃO. O arquivo não diz qual é a sua. Ler UTF-8 um arquivo
          cp1252 gera "informaÃ§Ã£o" (mojibake) ou uma exceção. Sistemas
          legados brasileiros produzem cp1252 até hoje.

  PDF  -> NÃO EXISTE "linha" NEM "parágrafo". O PDF guarda posições de
          glifos na página. O extrator reconstrói a leitura por heurística, e
          por isso vêm juntos: quebras no meio da frase, hifenização de fim de
          linha, cabeçalho e rodapé repetidos em toda página e colunas
          intercaladas. Pior: se o PDF for uma imagem escaneada, o texto
          extraído é VAZIO — e nenhum erro é levantado.

  DOCX -> É um ZIP de XML. O conteúdo está em parágrafos E em tabelas. Quem
          itera só `documento.paragraphs` perde todas as tabelas em silêncio —
          e tabelas costumam ser exatamente onde estão os dados.

  PLANILHA -> O texto não flui: ele mora em células, misturado com números,
          datas e fórmulas. É preciso DECIDIR quais colunas são texto. Empilhar
          a planilha inteira produz um amontoado numérico que envenena o corpus.

Princípio de projeto desta etapa: **todo trecho carrega a sua procedência**
(arquivo, formato e localizador — página 3, aba "Vendas" linha 12, tabela 1).
Sem procedência você não consegue auditar o corpus, atender a um pedido de
remoção, nem citar a fonte de uma resposta gerada por um modelo.

Uso:
    python src/readers.py                      # inspeciona tudo em data/brutos
    python src/readers.py data/brutos/x.pdf    # inspeciona um arquivo
    python src/readers.py data/brutos/x.pdf --texto   # despeja o texto extraído
"""

import argparse
import csv
import io
from dataclasses import dataclass, asdict
from pathlib import Path

from config import CFG, ROOT, BRUTOS_DIR
from utils import (configurar_terminal, contar_palavras, formatar_bytes,
                   hash_curto, truncar)


class ErroDeLeitura(Exception):
    """Falha ao ler um arquivo. O pipeline registra e segue para o próximo.

    Um corpus real tem arquivos corrompidos, protegidos por senha e vazios.
    Se um deles derrubar o processo inteiro, você nunca termina a ingestão.
    """


# ---------------------------------------------------------------------------
# A unidade de trabalho do pipeline
# ---------------------------------------------------------------------------
@dataclass
class Trecho:
    """Um pedaço de texto com procedência.

    Por que não um documento por arquivo? Porque a granularidade certa é a
    unidade natural do formato: uma página de PDF, uma linha de planilha, uma
    tabela do DOCX. Isso permite descartar a página 1 (capa) sem descartar o
    relatório inteiro, e permite dizer de onde veio cada frase.
    """
    texto: str
    arquivo: str          # caminho relativo à raiz do projeto
    formato: str          # pdf | txt | docx | xlsx | csv
    localizador: str      # "página 3", "aba Vendas · linha 12", "tabela 1"
    ordem: int            # posição dentro do arquivo, para reconstruir a ordem
    codificacao: str = ""  # só faz sentido para formatos de texto puro

    def to_dict(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        d["n_caracteres"] = len(self.texto)
        d["n_palavras"] = contar_palavras(self.texto)
        return d

    @property
    def id(self) -> str:
        # Identidade = origem, não conteúdo: assim o mesmo trecho mantém o
        # mesmo id mesmo depois de a limpeza mudar o texto.
        return hash_curto(self.arquivo, self.localizador, str(self.ordem))


def _relativo(caminho: Path) -> str:
    """Caminho relativo à raiz, com barras normais — comparável entre máquinas."""
    try:
        return caminho.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return caminho.as_posix()


# ---------------------------------------------------------------------------
# TXT / MD — o problema é a codificação
# ---------------------------------------------------------------------------
def decodificar(bruto: bytes, cfg=CFG) -> tuple[str, str]:
    """Tenta as codificações de `cfg.codificacoes` em ordem.

    Devolve (texto, nome_da_codificacao).

    Por que uma cadeia de tentativas e não `chardet`? Porque a cadeia é
    determinística e explicável. `utf-8` FALHA ruidosamente ao ler bytes
    inválidos — é justamente essa falha que usamos como teste. Já `latin-1`
    NUNCA falha (todo byte é um caractere válido), então ele fica por último,
    como rede de segurança: se chegou nele, o resultado pode estar errado, mas
    pelo menos o pipeline não trava.

    O `utf-8-sig` vem primeiro para consumir o BOM (os três bytes EF BB BF que
    o Bloco de Notas do Windows insere e que, lidos como `utf-8` puro, viram um
    caractere invisível no começo do texto).
    """
    for codec in cfg.codificacoes:
        try:
            return bruto.decode(codec), codec
        except UnicodeDecodeError:
            continue
    # Último recurso: substitui os bytes problemáticos por U+FFFD em vez de
    # perder o arquivo inteiro. O documento fica marcado e auditável.
    return bruto.decode("utf-8", errors="replace"), "utf-8/substituído"


def ler_txt(caminho: Path, cfg=CFG) -> list[Trecho]:
    """Arquivo de texto puro: um trecho por arquivo."""
    texto, codec = decodificar(caminho.read_bytes(), cfg)
    return [Trecho(texto=texto, arquivo=_relativo(caminho),
                   formato=caminho.suffix.lstrip(".").lower(),
                   localizador="arquivo inteiro", ordem=0, codificacao=codec)]


# ---------------------------------------------------------------------------
# PDF — o problema é que não existe estrutura de texto
# ---------------------------------------------------------------------------
def ler_pdf(caminho: Path, cfg=CFG) -> list[Trecho]:
    """Um trecho por página.

    Páginas SEM texto extraível viram trechos vazios em vez de sumirem. Isso é
    proposital: um PDF escaneado (imagem sem camada de texto) não gera erro
    nenhum, ele simplesmente devolve "". Se a gente descartasse em silêncio,
    o pipeline diria "10 documentos lidos" e você nunca saberia que 4 deles
    são páginas cegas precisando de OCR. Trecho vazio é rejeitado na ETAPA 3
    com o motivo `vazio`, e aparece no relatório.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ErroDeLeitura("pypdf não instalado (pip install pypdf)") from e

    try:
        leitor = PdfReader(str(caminho))
        # PDFs "protegidos" com senha vazia são comuns em documentos públicos:
        # a criptografia existe só para bloquear impressão/cópia.
        if leitor.is_encrypted:
            try:
                leitor.decrypt("")
            except Exception as e:
                raise ErroDeLeitura(f"PDF protegido por senha: {e}") from e
        paginas = leitor.pages
    except ErroDeLeitura:
        raise
    except Exception as e:
        raise ErroDeLeitura(f"PDF ilegível: {e}") from e

    limite = cfg.max_paginas_pdf or len(paginas)
    trechos = []
    for i, pagina in enumerate(paginas[:limite]):
        try:
            texto = pagina.extract_text() or ""
        except Exception as e:
            # Uma página quebrada não pode custar o documento inteiro.
            texto = ""
            print(f"    [aviso] página {i + 1} de {caminho.name} falhou: {e}")
        trechos.append(Trecho(texto=texto, arquivo=_relativo(caminho),
                              formato="pdf", localizador=f"página {i + 1}",
                              ordem=i))
    return trechos


# ---------------------------------------------------------------------------
# DOCX — o problema é o conteúdo que não está nos parágrafos
# ---------------------------------------------------------------------------
def _iterar_corpo_docx(documento):
    """Percorre o corpo do DOCX na ORDEM REAL, devolvendo parágrafos e tabelas.

    `documento.paragraphs` devolve só os parágrafos e `documento.tables` só as
    tabelas — e nenhum dos dois preserva a ordem entre eles. Para um contrato
    onde a cláusula explica a tabela logo abaixo, essa ordem é o significado.

    A solução é descer ao XML: o corpo do documento é uma sequência de
    elementos `w:p` (parágrafo) e `w:tbl` (tabela), e basta iterar os filhos.
    """
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for filho in documento.element.body.iterchildren():
        if isinstance(filho, CT_P):
            yield "paragrafo", Paragraph(filho, documento)
        elif isinstance(filho, CT_Tbl):
            yield "tabela", Table(filho, documento)


def _tabela_para_texto(tabela) -> str:
    """Serializa a tabela preservando as linhas.

    Uma tabela virou texto: como? Aqui usamos ` | ` entre células e quebra de
    linha entre linhas — legível para humanos e para modelos, e reversível o
    bastante para auditoria. Achatar tudo em uma linha só destruiria a relação
    coluna-valor, que costuma ser a informação toda.
    """
    linhas = []
    for linha in tabela.rows:
        celulas = [c.text.strip().replace("\n", " ") for c in linha.cells]
        if any(celulas):
            linhas.append(" | ".join(celulas))
    return "\n".join(linhas)


def ler_docx(caminho: Path, cfg=CFG) -> list[Trecho]:
    """Um trecho por bloco de parágrafos consecutivos e um por tabela."""
    try:
        import docx
    except ImportError as e:
        raise ErroDeLeitura("python-docx não instalado (pip install python-docx)") from e

    try:
        documento = docx.Document(str(caminho))
    except Exception as e:
        # O .doc antigo (binário, pré-2007) NÃO é lido por python-docx: ele é
        # outro formato, apesar do nome parecido. Converta antes com o Word ou
        # com o LibreOffice (`soffice --convert-to docx`).
        raise ErroDeLeitura(f"DOCX ilegível ({e}). É um .doc antigo renomeado?") from e

    trechos: list[Trecho] = []
    acumulado: list[str] = []
    contador = {"bloco": 0, "tabela": 0}

    def descarregar():
        """Fecha o bloco de parágrafos acumulado e o transforma em trecho."""
        if acumulado:
            contador["bloco"] += 1
            trechos.append(Trecho(
                texto="\n".join(acumulado), arquivo=_relativo(caminho),
                formato="docx", localizador=f"bloco {contador['bloco']}",
                ordem=len(trechos)))
            acumulado.clear()

    for tipo, elemento in _iterar_corpo_docx(documento):
        if tipo == "paragrafo":
            texto = elemento.text.strip()
            if texto:
                acumulado.append(texto)
        else:
            descarregar()   # a tabela interrompe o fluxo de parágrafos
            contador["tabela"] += 1
            texto = _tabela_para_texto(elemento)
            if texto:
                trechos.append(Trecho(
                    texto=texto, arquivo=_relativo(caminho), formato="docx",
                    localizador=f"tabela {contador['tabela']}", ordem=len(trechos)))
    descarregar()
    return trechos


# ---------------------------------------------------------------------------
# PLANILHAS — o problema é decidir o que é texto
# ---------------------------------------------------------------------------
def _e_numero(valor) -> bool:
    if isinstance(valor, bool):
        return True
    if isinstance(valor, (int, float)):
        return True
    if isinstance(valor, str):
        try:
            float(valor.replace(".", "").replace(",", ".").strip())
            return True
        except ValueError:
            return False
    return False


def detectar_colunas_textuais(linhas: list[list], cfg=CFG) -> list[int]:
    """Decide QUAIS colunas contêm texto natural, olhando uma amostra.

    Duas evidências, ambas necessárias:

      1. comprimento médio do conteúdo acima de `min_comprimento_medio_coluna`
         — uma coluna "Comentário do cliente" tem dezenas de caracteres; uma
         coluna "UF" ou "Status" tem 2 a 8;
      2. proporção de células numéricas abaixo de `max_prop_numerica_coluna`
         — datas, IDs, preços e códigos não são texto natural.

    Por que isso importa tanto: empilhar a planilha inteira em texto produz
    documentos como "1023 | 2024-03-11 | 45,90 | SP | OK", que passam raspando
    por qualquer filtro de tamanho e envenenam o corpus com ruído tabular.
    Preferimos escolher as colunas explicitamente — e o critério fica auditável
    porque a decisão é impressa no relatório de ingestão.

    Nota honesta: isto é uma heurística. Em produção você declararia as colunas
    em um arquivo de configuração por planilha. É exatamente o exercício 3.1.
    """
    if not linhas:
        return []

    n_colunas = max(len(l) for l in linhas)
    amostra = linhas[: cfg.linhas_amostra_planilha]
    escolhidas = []

    for c in range(n_colunas):
        valores = [l[c] for l in amostra if c < len(l) and l[c] not in (None, "")]
        if not valores:
            continue
        comprimento_medio = sum(len(str(v)) for v in valores) / len(valores)
        prop_numerica = sum(_e_numero(v) for v in valores) / len(valores)
        if (comprimento_medio >= cfg.min_comprimento_medio_coluna
                and prop_numerica <= cfg.max_prop_numerica_coluna):
            escolhidas.append(c)
    return escolhidas


def _linhas_para_trechos(linhas: list[list], caminho: Path, formato: str,
                         aba: str, ordem_inicial: int, cfg=CFG) -> list[Trecho]:
    """Converte as linhas de uma aba/CSV em trechos, um por linha de dados.

    A primeira linha é tratada como cabeçalho quando é toda textual e curta —
    e os nomes das colunas são preservados como rótulo no texto ("Comentário:
    ..."). Esse rótulo é contexto barato e valioso: sem ele, o modelo recebe
    uma frase solta sem saber o que ela responde.
    """
    if not linhas:
        return []

    cabecalho, corpo = None, linhas
    primeira = linhas[0]
    if primeira and all(isinstance(v, str) and 0 < len(v) <= 60
                        for v in primeira if v not in (None, "")):
        cabecalho, corpo = [str(v) if v is not None else "" for v in primeira], linhas[1:]

    colunas = detectar_colunas_textuais(corpo, cfg)
    if not colunas:
        return []   # aba puramente numérica: nada a extrair (e o relatório dirá)

    trechos = []
    for i, linha in enumerate(corpo, start=2 if cabecalho else 1):
        partes = []
        for c in colunas:
            valor = linha[c] if c < len(linha) else None
            if valor in (None, ""):
                continue
            rotulo = cabecalho[c].strip() if cabecalho and c < len(cabecalho) else ""
            partes.append(f"{rotulo}: {valor}" if rotulo else str(valor))
        if not partes:
            continue
        localizador = f"aba {aba} · linha {i}" if aba else f"linha {i}"
        trechos.append(Trecho(texto="\n".join(partes), arquivo=_relativo(caminho),
                              formato=formato, localizador=localizador,
                              ordem=ordem_inicial + len(trechos)))
    return trechos


def ler_xlsx(caminho: Path, cfg=CFG) -> list[Trecho]:
    """Planilha do Excel: percorre todas as abas."""
    try:
        from openpyxl import load_workbook
    except ImportError as e:
        raise ErroDeLeitura("openpyxl não instalado (pip install openpyxl)") from e

    try:
        # data_only=True devolve o VALOR calculado da fórmula, não o texto
        # "=SOMA(A1:A9)". Cuidado clássico: esse valor é o que o Excel deixou
        # em cache no último salvamento. Planilha gerada por script e nunca
        # aberta no Excel devolve None aqui.
        # read_only=True evita carregar a planilha inteira na memória.
        planilha = load_workbook(str(caminho), read_only=True, data_only=True)
    except Exception as e:
        raise ErroDeLeitura(f"Planilha ilegível: {e}") from e

    trechos: list[Trecho] = []
    for aba in planilha.worksheets:
        linhas = [list(l) for l in aba.iter_rows(values_only=True)]
        linhas = [l for l in linhas if any(v not in (None, "") for v in l)]
        trechos += _linhas_para_trechos(linhas, caminho, "xlsx", aba.title,
                                        len(trechos), cfg)
    planilha.close()
    return trechos


def ler_csv(caminho: Path, cfg=CFG) -> list[Trecho]:
    """CSV: codificação incerta E separador incerto.

    No Brasil o separador é quase sempre `;`, porque a vírgula já é o separador
    decimal. O `csv.Sniffer` descobre isso a partir de uma amostra; se ele
    falhar (arquivo com uma coluna só, por exemplo), caímos na vírgula padrão.
    """
    texto, codec = decodificar(caminho.read_bytes(), cfg)
    amostra = texto[:4096]
    try:
        dialeto = csv.Sniffer().sniff(amostra, delimiters=",;\t|")
    except csv.Error:
        dialeto = csv.excel

    linhas = [l for l in csv.reader(io.StringIO(texto), dialeto)
              if any(c.strip() for c in l)]
    trechos = _linhas_para_trechos(linhas, caminho, "csv", "", 0, cfg)
    for t in trechos:
        t.codificacao = codec
    return trechos


# ---------------------------------------------------------------------------
# Despacho por extensão
# ---------------------------------------------------------------------------
LEITORES = {
    ".txt": ler_txt,
    ".md": ler_txt,
    ".pdf": ler_pdf,
    ".docx": ler_docx,
    ".xlsx": ler_xlsx,
    ".xlsm": ler_xlsx,
    ".csv": ler_csv,
}


def ler_documento(caminho: Path, cfg=CFG) -> list[Trecho]:
    """Lê qualquer formato suportado. Levanta ErroDeLeitura em caso de falha.

    Repare que o despacho é por EXTENSÃO, não por conteúdo. É o suficiente para
    a aula, mas em produção a extensão mente com frequência (um .txt que é um
    HTML, um .xlsx que é um .csv renomeado). A correção é olhar os primeiros
    bytes — a "assinatura mágica" do arquivo. Veja o exercício 3.2.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise ErroDeLeitura("arquivo não encontrado")

    tamanho_mb = caminho.stat().st_size / (1024 * 1024)
    if tamanho_mb > cfg.tamanho_maximo_mb:
        raise ErroDeLeitura(
            f"arquivo de {tamanho_mb:.1f} MB acima do limite de "
            f"{cfg.tamanho_maximo_mb} MB (ajuste em config.py)")

    leitor = LEITORES.get(caminho.suffix.lower())
    if leitor is None:
        raise ErroDeLeitura(f"formato não suportado: {caminho.suffix or '(sem extensão)'}")
    return leitor(caminho, cfg)


def listar_arquivos(pasta: Path = BRUTOS_DIR, cfg=CFG) -> list[Path]:
    """Varre a pasta recursivamente e devolve os arquivos em ordem estável.

    `sorted` não é detalhe: sem ele a ordem depende do sistema de arquivos, e
    duas execuções da mesma ingestão produziriam corpora em ordens diferentes —
    o que atrapalha o `diff` entre execuções e a reprodutibilidade.
    """
    if not pasta.exists():
        return []
    return sorted(p for p in pasta.rglob("*")
                  if p.is_file() and p.suffix.lower() in cfg.extensoes)


if __name__ == "__main__":
    configurar_terminal()
    # Rode:  python src/readers.py
    # Serve para ver O QUE cada leitor extraiu, ANTES de limpar e filtrar.
    # Olhe seus dados: um pipeline alimentado com extração errada não reclama,
    # ele apenas produz um corpus errado com aparência perfeita.
    p = argparse.ArgumentParser(description="Inspeção dos leitores de arquivo")
    p.add_argument("arquivos", nargs="*", type=Path,
                   help="arquivos a inspecionar (padrão: tudo em data/brutos)")
    p.add_argument("--texto", action="store_true", help="despeja o texto extraído")
    args = p.parse_args()

    alvos = args.arquivos or listar_arquivos()
    if not alvos:
        print(f"Nenhum arquivo em {BRUTOS_DIR}. Rode antes: python src/make_samples.py")
        raise SystemExit(1)

    print(f"{'arquivo':<34}{'formato':>8}{'tamanho':>10}{'trechos':>9}{'palavras':>10}")
    print("-" * 71)
    for caminho in alvos:
        try:
            trechos = ler_documento(caminho)
        except ErroDeLeitura as e:
            print(f"{caminho.name:<34}{'—':>8}{'—':>10}   [erro] {e}")
            continue

        n_palavras = sum(contar_palavras(t.texto) for t in trechos)
        vazios = sum(1 for t in trechos if not t.texto.strip())
        marca = f"  ({vazios} vazio(s)!)" if vazios else ""
        print(f"{caminho.name:<34}{trechos[0].formato if trechos else '—':>8}"
              f"{formatar_bytes(caminho.stat().st_size):>10}"
              f"{len(trechos):>9}{n_palavras:>10}{marca}")

        if args.texto:
            for t in trechos:
                print(f"\n  -- {t.localizador} "
                      f"({len(t.texto)} car., {contar_palavras(t.texto)} pal."
                      f"{', ' + t.codificacao if t.codificacao else ''})")
                print("  " + (truncar(t.texto, 400) if t.texto.strip()
                              else "[VAZIO — provável PDF escaneado, precisa de OCR]"))
            print()
