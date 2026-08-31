"""
clean.py — ETAPA 2 do pipeline: normalização e limpeza
--------------------------------------------------------

Limpar texto NÃO é "tirar o que é feio". É tornar EQUIVALENTES as coisas que
deveriam ser iguais, e só isso.

Duas ocorrências da palavra "inspeção" precisam ser o mesmo símbolo para o
modelo, mesmo que uma tenha vindo de um PDF com acento pré-composto (U+00E7) e
a outra de um DOCX com acento combinante (c + U+0327). Visualmente idênticas,
bytes diferentes, tokens diferentes, vocabulário inflado.

Onde a limpeza vira destruição — a decisão mais importante desta etapa:

    Cada regra abaixo APAGA informação. Minúsculas apagam a distinção entre
    "Apple" e "apple". Remover pontuação apaga o limite da frase. Remover
    acento apaga a diferença entre "e" e "é". Nada disso volta.

Por isso o pipeline desta aula é DELIBERADAMENTE conservador: normaliza forma
(unicode, espaço, hífen de fim de linha) e não toca no conteúdo. Não passa para
minúsculas, não remove acento, não remove stopwords, não faz stemming.

Esse conselho mudou nos últimos anos e vale explicar por quê: com modelos de
saco de palavras (Naive Bayes, TF-IDF + regressão), reduzir o vocabulário era
essencial e a receita "minúscula + sem acento + sem stopword + stemming" fazia
sentido. Com modelos baseados em transformadores, o tokenizador já lida com
maiúsculas e morfologia, e essa limpeza agressiva só JOGA SINAL FORA. Limpe
para corrigir defeito de extração, não para "simplificar" a língua.

Uso:
    python src/clean.py data/brutos/relatorio_tecnico.pdf   # antes x depois
    python src/clean.py data/brutos/comunicado_legado.txt
"""

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

from config import CFG, PADROES_MOLDURA
from utils import configurar_terminal


# ---------------------------------------------------------------------------
# 1. Codificação: consertar o que já foi lido errado
# ---------------------------------------------------------------------------
# Assinaturas de mojibake: como ficam, lidos em cp1252, os bytes UTF-8 de "ç",
# "ã", "é", "ó" e do espaço não separável.
SINAIS_MOJIBAKE = re.compile(r"Ã[\x80-\xbf©£§ª±µº]|Â[\xa0-\xbf]|â€[\x93\x94\x9c\x9d™]")


def corrigir_mojibake(texto: str) -> str:
    """Desfaz o clássico "informaÃ§Ã£o" -> "informação".

    O estrago acontece assim: alguém salvou o arquivo em UTF-8, um segundo
    programa leu esses bytes como se fossem cp1252, e gravou o resultado. Cada
    caractere acentuado (2 bytes em UTF-8) virou dois caracteres latinos.

    A correção é fazer o caminho inverso: recodificar o texto de volta para
    bytes usando cp1252 e decodificá-lo como UTF-8. Só aplicamos quando o
    resultado REDUZ o número de sinais suspeitos — a verificação evita
    estragar um texto que legitimamente contenha "Ã" (nomes próprios, palavras
    em vietnamita, exemplos de aula sobre mojibake...).
    """
    if not SINAIS_MOJIBAKE.search(texto):
        return texto
    try:
        candidato = texto.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto
    if len(SINAIS_MOJIBAKE.findall(candidato)) < len(SINAIS_MOJIBAKE.findall(texto)):
        return candidato
    return texto


# ---------------------------------------------------------------------------
# 2. Unicode: uma forma canônica para cada caractere
# ---------------------------------------------------------------------------
def normalizar_unicode(texto: str) -> str:
    """Aplica a forma de normalização NFKC.

    NFC  junta base + acento combinante em um único ponto de código: a mesma
         palavra passa a ter sempre a mesma sequência de bytes.
    NFKC faz o mesmo E ainda converte "compatibilidade": as ligaduras ﬁ e ﬂ
         (que os PDFs adoram) viram "fi" e "fl", as aspas e traços tipográficos
         de largura fixa viram os equivalentes comuns, e os caracteres de
         largura dupla viram largura simples.

    Escolhemos NFKC porque as ligaduras de PDF são justamente o nosso problema.
    O preço: NFKC também transforma "²" em "2" e "½" em "1⁄2". Em um corpus
    científico onde o expoente importa, prefira NFC. É uma decisão de domínio,
    não uma verdade universal.
    """
    return unicodedata.normalize("NFKC", texto)


def remover_controle(texto: str) -> str:
    """Remove caracteres de controle e invisíveis, preservando \\n e \\t.

    Vêm de todo lado: \\x0c (quebra de página) em PDFs, \\x00 em arquivos
    truncados, U+200B (espaço de largura zero) e U+00AD (hífen opcional) em
    conteúdo copiado da web. São invisíveis para você e visíveis para o
    tokenizador — que os transforma em tokens sujos no meio das palavras.
    """
    return "".join(
        c for c in texto
        if c in "\n\t" or (unicodedata.category(c)[0] != "C" and c != "­")
    )


def normalizar_pontuacao(texto: str) -> str:
    """Unifica aspas, travessões e reticências tipográficas.

    Editores de texto trocam " por “ ” automaticamente, e três pontos por "…".
    Não há nada de errado com isso — o problema é a MISTURA: o mesmo corpus com
    quatro variantes de aspas produz quatro tokens diferentes para a mesma
    função sintática.
    """
    trocas = {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‚": "'",
        "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
        "…": "...", " ": " ", " ": " ", " ": " ",
    }
    return texto.translate(str.maketrans(trocas))


# ---------------------------------------------------------------------------
# 3. Defeitos específicos de PDF
# ---------------------------------------------------------------------------
HIFENIZACAO = re.compile(r"(\w)-\s*\n\s*(\w)", re.UNICODE)


def juntar_hifenizacao(texto: str) -> str:
    """Desfaz a hifenização de fim de linha: "trimes-\\ntre" -> "trimestre".

    O hífen ali não é parte da palavra: é um artefato da diagramação, inserido
    para justificar o texto na largura da página. Se ficar, o corpus passa a
    conter "trimes", "tre" e "trimes-tre" como se fossem palavras.

    Honestidade sobre a heurística: ela erra com palavras legitimamente
    hifenizadas que caíram no fim da linha ("guarda-\\nchuva" vira
    "guardachuva"). O erro é raro e muito menos danoso que o problema que
    resolve. A correção — consultar um léxico antes de juntar — é o
    exercício 3.3.
    """
    return HIFENIZACAO.sub(r"\1\2", texto)


FIM_DE_FRASE = re.compile(r"[.!?:;»\"')\]]\s*$")
INICIO_MINUSCULO = re.compile(r"^[a-zà-öø-ÿ]")
INICIO_DE_ITEM = re.compile(r"^\s*(?:[-•*·—]|\(?[a-z0-9]{1,3}[.)])\s")
PROPORCAO_LINHA_CHEIA = 0.9

# Separador de células usado por `readers._tabela_para_texto`. Uma linha que o
# contém é uma LINHA DE TABELA, e linha de tabela nunca pode ser juntada com a
# seguinte: ali a quebra de linha não é diagramação, é a fronteira entre dois
# registros — exatamente a informação que a tabela codifica.
SEPARADOR_TABELA = " | "


def rejuntar_paragrafos(texto: str) -> str:
    """Refaz os parágrafos que a diagramação quebrou em linhas soltas.

    Depois da extração, uma frase única aparece assim:

        A analise preliminar indica que a maior parte das
        falhas nao e detectada pela conferencia manual.

    Para o modelo, essa quebra é ruído: ela não corresponde a nada semântico,
    só à largura da coluna em que o texto foi diagramado.

    O problema é distinguir esse caso deste outro, que NÃO pode ser juntado:

        1. Contexto do projeto
        A area de inspecao visual da fabrica registrou...

    Nos dois casos a primeira linha não termina em pontuação. Duas evidências
    resolvem o impasse:

      * a linha SEGUINTE começa em minúscula -> é continuação de frase,
        praticamente sem chance de erro;
      * a linha ANTERIOR está "cheia", isto é, chega perto da maior largura do
        bloco -> ela acabou porque a margem chegou, não porque a ideia acabou.
        Título é curto por natureza; linha diagramada é cheia por natureza.

    Basta uma das duas para juntar. É a segunda que salva "…metropolitana de
    Belo / Horizonte, iniciou…", em que a continuação começa por nome próprio
    em maiúscula.

    Linha em branco, marcador de lista e LINHA DE TABELA interrompem sempre.
    A tabela merece nota: as suas linhas são todas "cheias" e nenhuma termina
    em ponto, então a regra da largura as juntaria todas em um parágrafo único
    — destruindo a fronteira entre os registros, que é a informação que a
    tabela carrega. Foi o que aconteceu na primeira versão deste arquivo, e o
    defeito só apareceu quando alguém leu um documento do corpus final. Moral:
    heurística boa em prosa pode ser destrutiva em dado estruturado, e o único
    jeito de descobrir é olhar a saída.
    """
    saida = []
    for paragrafo in texto.split("\n\n"):
        linhas = [l.strip() for l in paragrafo.split("\n")]
        # A largura de referência é a do próprio bloco: cada PDF, cada coluna e
        # cada corpo de fonte produzem uma largura diferente. Medir localmente
        # dispensa qualquer constante mágica de "80 caracteres".
        largura_cheia = max((len(l) for l in linhas), default=0) * PROPORCAO_LINHA_CHEIA

        juntadas: list[str] = []
        for linha in linhas:
            anterior = juntadas[-1] if juntadas else ""
            juntar = (
                juntadas and linha
                and not FIM_DE_FRASE.search(anterior)
                and not INICIO_DE_ITEM.match(linha)
                and SEPARADOR_TABELA not in anterior
                and SEPARADOR_TABELA not in linha
                and (INICIO_MINUSCULO.match(linha) or len(anterior) >= largura_cheia)
            )
            if juntar:
                juntadas[-1] = anterior + " " + linha
            else:
                juntadas.append(linha)
        saida.append("\n".join(juntadas))
    return "\n\n".join(saida)


# ---------------------------------------------------------------------------
# 4. Cabeçalhos e rodapés: a limpeza que precisa olhar o ARQUIVO INTEIRO
# ---------------------------------------------------------------------------
def _chave_linha(linha: str) -> str:
    return re.sub(r"\s+", " ", linha).strip()


def detectar_cabecalhos_rodapes(textos: list[str], cfg=CFG) -> set[str]:
    """Descobre as linhas que se repetem em quase toda página do MESMO arquivo.

    Nenhuma regra fixa poderia saber que "ACME Tecnologia - Relatorio Interno
    RT-2024-017" é cabeçalho: isso muda a cada documento. Mas há uma pista
    estatística que vale para todos: **cabeçalho é a linha que aparece em toda
    página**. Se uma linha ocorre em 60% ou mais das páginas, ela é moldura,
    não conteúdo.

    Duas salvaguardas:
      * exigimos um mínimo de páginas (`min_paginas_para_cabecalho`), porque
        com 2 páginas "aparecer nas duas" é coincidência, não evidência;
      * linhas muito longas são poupadas — uma frase de 200 caracteres que se
        repete é citação ou cláusula contratual, e apagá-la seria perder
        conteúdo real.

    O que este método NÃO pega: o rodapé "Página 1 de 3", que muda a cada
    página e por isso nunca acumula frequência. Esse é papel do filtro de
    boilerplate por expressão regular, na ETAPA 3. Os dois se completam.
    """
    if len(textos) < cfg.min_paginas_para_cabecalho:
        return set()

    contagem = Counter()
    for texto in textos:
        # set(): a mesma linha repetida DENTRO de uma página conta uma vez só,
        # senão uma tabela com células iguais inflaria a estatística.
        contagem.update({_chave_linha(l) for l in texto.split("\n")
                         if 0 < len(_chave_linha(l)) <= 120})

    minimo = max(2, int(len(textos) * cfg.limiar_cabecalho_rodape))
    return {linha for linha, n in contagem.items() if n >= minimo}


def remover_linhas(texto: str, linhas_a_remover: set[str]) -> str:
    if not linhas_a_remover:
        return texto
    return "\n".join(l for l in texto.split("\n")
                     if _chave_linha(l) not in linhas_a_remover)


REGEX_MOLDURA = [re.compile(p, re.IGNORECASE) for p in PADROES_MOLDURA]
MAX_LINHA_MOLDURA = 80


def remover_moldura(texto: str) -> str:
    """Remove o mobiliário de página que a detecção por frequência não pega.

    "Página 1 de 3" muda a cada página, então nunca acumula frequência — e é
    justamente por isso que precisa de um padrão fixo. Mesma coisa para
    numeração centralizada ("- 12 -") e marca d'água de conversão.

    A salvaguarda do comprimento é essencial: só removemos linhas CURTAS
    (até 80 caracteres). Uma cláusula contratual de 300 caracteres que por
    acaso contenha "©" é conteúdo, e apagá-la seria trocar um problema
    cosmético por perda de informação real. Na dúvida entre remover e manter,
    mantenha: filtrar depois é possível, recuperar não é.
    """
    return "\n".join(
        l for l in texto.split("\n")
        if not (len(l.strip()) <= MAX_LINHA_MOLDURA
                and any(r.search(l) for r in REGEX_MOLDURA))
    )


# ---------------------------------------------------------------------------
# 5. Espaços em branco
# ---------------------------------------------------------------------------
def normalizar_espacos(texto: str, cfg=CFG) -> str:
    """Colapsa espaços repetidos, remove espaço no fim da linha e limita as
    linhas em branco consecutivas.

    Espaço repetido é a assinatura da extração de PDF em coluna (o extrator
    preenche a distância horizontal com espaços). Manter isso significa gastar
    tokens — e o contexto de um modelo é pago por token — com nada.
    """
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    limite = cfg.max_linhas_em_branco + 1
    texto = re.sub(r"\n{%d,}" % (limite + 1), "\n" * limite, texto)
    return texto.strip()


# ---------------------------------------------------------------------------
# 6. Dados pessoais (LGPD)
# ---------------------------------------------------------------------------
# Guardas (?<!\d) e (?!\d): impedem que um padrão morda o MEIO de uma sequência
# numérica maior. Sem eles, o padrão de CEP casa dentro de um número de
# protocolo e mascara cinco dígitos no meio, deixando lixo dos dois lados.
PADROES_PII = [
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Duas formas: com a barra (que distingue CNPJ de CPF sem ambiguidade) e
    # os 14 dígitos crus. O CNPJ cru pode ser reconhecido com segurança porque
    # 14 é um comprimento que nenhum outro identificador brasileiro usa — ao
    # contrário dos 11 dígitos do CPF, que colidem com telefone.
    ("[CNPJ]", re.compile(
        r"(?<!\d)(?:\d{2}\.?\d{3}\.?\d{3}/\d{4}-?\d{2}|\d{14})(?!\d)")),
    ("[CPF]", re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)")),
    # O DDD é obrigatório, e os parênteses entram DENTRO do casamento. Uma
    # versão anterior deixava o "(" de fora e produzia "tel ([TELEFONE]." —
    # parêntese órfão. Quando um padrão pode começar por um delimitador, ele
    # precisa consumir o par inteiro.
    ("[TELEFONE]", re.compile(
        r"(?<!\d)(?:\+55\s*)?(?:\(\d{2}\)|\d{2})[\s.-]?9?\d{4}[-.\s]?\d{4}(?!\d)")),
    ("[CEP]", re.compile(r"(?<!\d)\d{5}-\d{3}(?!\d)")),
]


def mascarar_pii(texto: str) -> str:
    """Substitui dados pessoais identificáveis por marcadores.

    Por que isto vive no pipeline e não "depois": porque o corpus é copiado,
    versionado e usado para treinar. Um CPF que entrou no conjunto de treino
    pode ser REPRODUZIDO pelo modelo mais tarde, e aí não há como retirá-lo.
    Sob a LGPD, minimização de dados é princípio (art. 6º, III): o tratamento
    deve se limitar ao mínimo necessário para a finalidade.

    Quatro limitações que precisam ser ditas em voz alta:

      1. A ORDEM importa. CNPJ antes de CPF, senão o padrão de CPF morde o
         começo do CNPJ e deixa um resto.

      2. Só pegamos CPF FORMATADO (123.456.789-00). Uma sequência crua de 11
         dígitos é genuinamente ambígua — pode ser CPF, telefone com DDD ou
         número de protocolo — e chutar significa mascarar a coisa errada. A
         ambiguidade é do dado, não do código: nenhuma expressão regular
         resolve, porque a informação necessária não está lá.

      3. Expressão regular não reconhece NOME de pessoa, endereço nem
         matrícula. Para isso é preciso reconhecimento de entidades nomeadas
         (NER) — outra aula. Não confunda este filtro com anonimização.

      4. Mascarar é irreversível e pode destruir conteúdo legítimo (um contrato
         cujo objeto É o CNPJ da parte). Por isso o padrão é `False` em
         config.py: mascarar é uma decisão consciente, não um efeito colateral.
    """
    for marcador, padrao in PADROES_PII:
        texto = padrao.sub(marcador, texto)
    return texto


# ---------------------------------------------------------------------------
# O pipeline de limpeza
# ---------------------------------------------------------------------------
def limpar(texto: str, cfg=CFG, mascarar: bool | None = None) -> str:
    """Aplica todas as regras, NESTA ORDEM (a ordem não é arbitrária).

    1. mojibake      -> antes de tudo: normalizar bytes errados só os congela.
    2. unicode NFKC  -> uma forma canônica; desfaz ligaduras do PDF.
    3. controle      -> some com o invisível.
    4. pontuação     -> aspas e travessões uniformes.
    5. moldura       -> AINDA com uma linha por linha do original: depois do
                        rejuntar, "Página 1 de 3" pode ter grudado no parágrafo
                        anterior e o padrão não casa mais.
    6. hifenização   -> PRECISA dos "\\n" originais para achar o fim da linha.
    7. reparágrafos  -> idem; por isso vem antes de mexer nos espaços.
    8. espaços       -> só agora, quando a estrutura de linhas já cumpriu o papel.
    9. PII           -> por último, sobre o texto já normalizado, senão um
                        telefone escrito com espaço não separável escapa.

    Trocar a ordem de 5, 6 e 7 quebra o resultado de um jeito silencioso: o
    texto sai plausível, só que com o rodapé no meio do parágrafo. Sequência de
    transformações textuais é código sensível à ordem — documente sempre.
    """
    mascarar = cfg.mascarar_pii if mascarar is None else mascarar
    texto = corrigir_mojibake(texto)
    texto = normalizar_unicode(texto)
    texto = remover_controle(texto)
    texto = normalizar_pontuacao(texto)
    texto = remover_moldura(texto)
    texto = juntar_hifenizacao(texto)
    texto = rejuntar_paragrafos(texto)
    texto = normalizar_espacos(texto, cfg)
    if mascarar:
        texto = mascarar_pii(texto)
    return texto


def limpar_arquivo(trechos: list, cfg=CFG, mascarar: bool | None = None) -> list:
    """Limpa todos os trechos de UM arquivo, com a etapa que exige visão global.

    A detecção de cabeçalho/rodapé é estatística sobre as páginas do mesmo
    arquivo — logo, é impossível fazê-la trecho a trecho. Este é o motivo de o
    pipeline processar arquivo por arquivo, e não trecho por trecho: algumas
    decisões só existem no nível do documento.

    Modifica os trechos no lugar e devolve a lista, além do conjunto de linhas
    removidas (útil para o relatório).
    """
    repetidas = detectar_cabecalhos_rodapes([t.texto for t in trechos], cfg)
    for trecho in trechos:
        texto = remover_linhas(trecho.texto, repetidas)
        trecho.texto = limpar(texto, cfg, mascarar)
    return trechos, repetidas


if __name__ == "__main__":
    configurar_terminal()
    # Rode:  python src/clean.py data/brutos/relatorio_tecnico.pdf
    # Compare o ANTES e o DEPOIS. Toda regra de limpeza precisa ser inspecionada
    # com o olho humano pelo menos uma vez: é assim que você descobre que
    # apagou uma coluna inteira "sem querer".
    from readers import ler_documento

    p = argparse.ArgumentParser(description="Antes x depois da limpeza")
    p.add_argument("arquivo", type=Path)
    p.add_argument("--trecho", type=int, default=0, help="índice do trecho")
    p.add_argument("--mascarar", action="store_true", help="mascarar dados pessoais")
    args = p.parse_args()

    trechos = ler_documento(args.arquivo)
    if not trechos:
        raise SystemExit("Nenhum trecho extraído.")

    originais = [t.texto for t in trechos]
    _, repetidas = limpar_arquivo(trechos, mascarar=args.mascarar)

    if repetidas:
        print(f"Linhas de cabeçalho/rodapé detectadas ({len(repetidas)}):")
        for linha in sorted(repetidas):
            print(f"  - {linha!r}")
        print()

    i = min(args.trecho, len(trechos) - 1)
    print("=" * 74)
    print(f"ANTES — {trechos[i].localizador} ({len(originais[i])} caracteres)")
    print("=" * 74)
    print(originais[i][:1200])
    print("\n" + "=" * 74)
    print(f"DEPOIS — {trechos[i].localizador} ({len(trechos[i].texto)} caracteres)")
    print("=" * 74)
    print(trechos[i].texto[:1200])

    reducao = 1 - len(trechos[i].texto) / max(1, len(originais[i]))
    print(f"\nRedução de tamanho: {reducao:.1%}")
