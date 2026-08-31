"""
make_samples.py — gera o acervo de exemplo em data/brutos/
------------------------------------------------------------

O projeto de CNN da disciplina baixa o Fashion-MNIST pronto. Em PLN não existe
esse conforto: o "dataset" de um pipeline de aquisição é um punhado de arquivos
heterogêneos, e é justamente a heterogeneidade que se quer ensinar.

Este script cria, de forma DETERMINÍSTICA, um acervo pequeno que contém de
propósito todas as patologias que o pipeline precisa saber tratar:

    documento                        patologia embutida
    ---------------------------------------------------------------------
    relatorio_tecnico.pdf            cabeçalho/rodapé repetidos, hifenização
    manual_qualidade.pdf             prosa limpa (o "controle" do experimento)
    nota_fiscal_escaneada.pdf        SEM camada de texto -> extração vazia
    noticia.txt                      UTF-8, documento saudável
    noticia_copia.txt                duplicata EXATA de noticia.txt
    noticia_editada.txt              QUASE-duplicata (poucas palavras trocadas)
    comunicado_legado.txt            codificado em cp1252 (sistema legado)
    menu_site.txt                    navegação/boilerplate, sem conteúdo
    leia-me.txt                      curto demais
    ocr_ruim.txt                     ruído de OCR (símbolos no lugar de letras)
    abstract.txt                     inglês (idioma fora do alvo)
    procedimento_operacional.docx    parágrafos + TABELA (que o ingênuo perde)
    avaliacoes_clientes.xlsx         aba textual + aba puramente numérica
    chamados.csv                     separador ';' e cp1252

O gerador de PDF é escrito à mão, em ~60 linhas, sem nenhuma biblioteca. Além
de evitar mais uma dependência, ele deixa visível o motivo de o PDF ser um
formato tão hostil à extração de texto: o arquivo não guarda parágrafos, guarda
comandos de desenho que POSICIONAM pedaços de texto na página.

Uso:
    python src/make_samples.py
    python src/make_samples.py --forcar     # sobrescreve o que já existe
"""

import argparse
import csv
import io
from pathlib import Path

from config import BRUTOS_DIR
from utils import configurar_terminal


# ===========================================================================
# Um gerador mínimo de PDF
# ===========================================================================
def _escapar(texto: str) -> str:
    """Dentro de um PDF, `(`, `)` e `\\` são metacaracteres da string."""
    return texto.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _fluxo_de_texto(linhas: list[str], topo: int = 790, corpo: int = 11,
                    entrelinha: int = 15) -> bytes:
    """Monta o content stream: comandos de desenho de texto.

    BT/ET  : abre e fecha um bloco de texto
    /F1 Tf : escolhe a fonte e o corpo
    Td     : posiciona o cursor na página (x, y a partir do canto INFERIOR)
    TL     : define a entrelinha
    Tj     : desenha a string
    T*     : desce uma linha

    Note o que NÃO existe aqui: parágrafo, título, ordem de leitura, coluna.
    Um extrator de PDF precisa RECONSTRUIR tudo isso a partir de coordenadas —
    e é por isso que ele erra.
    """
    partes = ["BT", f"/F1 {corpo} Tf", f"50 {topo} Td", f"{entrelinha} TL"]
    for linha in linhas:
        partes.append(f"({_escapar(linha)}) Tj")
        partes.append("T*")
    partes.append("ET")
    return "\n".join(partes).encode("cp1252", errors="replace")


def _fluxo_de_imagem() -> bytes:
    """Página sem NENHUM texto: apenas um retângulo cinza.

    É o que um scanner produz — uma imagem da folha. Para o extrator, esta
    página existe, tem tamanho, e devolve string vazia. Nenhum erro é
    levantado. Este é o silêncio mais perigoso da ETAPA 1.
    """
    return (b"0.86 0.86 0.86 rg\n60 480 480 300 re f\n"
            b"0.70 0.70 0.70 rg\n90 520 200 14 re f\n90 560 380 14 re f\n"
            b"90 600 340 14 re f\n90 640 300 14 re f\n90 700 250 20 re f\n")


def escrever_pdf(caminho: Path, paginas: list[list[str]],
                 escaneado: bool = False) -> None:
    """Escreve um PDF 1.4 válido, com tabela xref correta.

    Estrutura mínima de um PDF:
        1 Catalog -> 2 Pages -> N objetos Page -> cada um com um Contents
        3 Font (Helvetica, WinAnsiEncoding)
        xref: o índice com o deslocamento em bytes de cada objeto
        trailer: aponta para a raiz (Catalog)
    """
    n = len(paginas)
    ids_pagina = [4 + 2 * i for i in range(n)]      # 4, 6, 8, ...
    ids_conteudo = [5 + 2 * i for i in range(n)]    # 5, 7, 9, ...

    objetos: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: ("<< /Type /Pages /Kids [" + " ".join(f"{i} 0 R" for i in ids_pagina)
            + f"] /Count {n} >>").encode(),
        3: (b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding >>"),
    }

    for i in range(n):
        objetos[ids_pagina[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {ids_conteudo[i]} 0 R >>").encode()
        fluxo = _fluxo_de_imagem() if escaneado else _fluxo_de_texto(paginas[i])
        objetos[ids_conteudo[i]] = (
            f"<< /Length {len(fluxo)} >>\nstream\n".encode() + fluxo + b"\nendstream")

    saida = io.BytesIO()
    saida.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")   # o comentário binário marca
    deslocamentos = {}                              # o arquivo como não-ASCII
    for num in sorted(objetos):
        deslocamentos[num] = saida.tell()
        saida.write(f"{num} 0 obj\n".encode() + objetos[num] + b"\nendobj\n")

    inicio_xref = saida.tell()
    total = max(objetos) + 1
    saida.write(f"xref\n0 {total}\n".encode())
    saida.write(b"0000000000 65535 f \n")
    for num in range(1, total):
        # Objetos inexistentes na numeração viram entradas livres.
        if num in deslocamentos:
            saida.write(f"{deslocamentos[num]:010d} 00000 n \n".encode())
        else:
            saida.write(b"0000000000 65535 f \n")
    saida.write(f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n"
                f"{inicio_xref}\n%%EOF\n".encode())

    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(saida.getvalue())


# ===========================================================================
# Conteúdo do acervo
# ===========================================================================
CABECALHO = "ACME Tecnologia  -  Relatorio Interno RT-2024-017"
RODAPE = "Documento de circulacao restrita - nao distribuir"

# Repare nas linhas terminadas em "-": é a hifenização de fim de linha, que o
# PDF preserva e que a limpeza (ETAPA 2) precisa desfazer.
PAGINAS_RELATORIO = [
    [CABECALHO, "",
     "1. Contexto do projeto",
     "",
     "A area de inspecao visual da fabrica registrou, no ultimo trimes-",
     "tre, um aumento de 18% no numero de pecas devolvidas pelo cli-",
     "ente final. A analise preliminar indica que a maior parte das fa-",
     "lhas nao e detectada pela conferencia manual, realizada por amos-",
     "tragem em apenas 5% do lote produzido. O objetivo deste relato-",
     "rio e avaliar a viabilidade de um sistema automatico de inspecao",
     "baseado em visao computacional, capaz de analisar 100% das pecas",
     "na propria linha de producao, sem reduzir a cadencia atual.",
     "",
     "O escopo considerado abrange as tres familias de produtos com",
     "maior volume, que juntas respondem por 72% do faturamento anual",
     "da unidade. Ficam fora do escopo, nesta primeira fase, as pecas",
     "sob encomenda, cuja variabilidade geometrica exigiria um conjun-",
     "to de treinamento especifico para cada pedido.",
     "", "Pagina 1 de 3", RODAPE],

    [CABECALHO, "",
     "2. Dados disponiveis e limitacoes",
     "",
     "Foram levantadas 12.400 imagens capturadas pelas cameras ja ins-",
     "taladas na linha, referentes aos ultimos oito meses de operacao.",
     "Apenas 1.850 dessas imagens possuem rotulo confiavel, atribuido",
     "pela equipe de qualidade no momento da inspecao manual. As de-",
     "mais estao sem rotulo ou com rotulo herdado do lote, o que nao",
     "garante correspondencia com a peca individual fotografada.",
     "",
     "A distribuicao das classes e fortemente desbalanceada: pecas con-",
     "formes representam 96,3% do total rotulado. Esse desbalanceamento",
     "precisa ser tratado explicitamente, sob pena de o modelo aprender",
     "a responder sempre conforme e ainda assim exibir acuracia alta.",
     "Recomenda-se avaliar o sistema por revocacao da classe defeito e",
     "por custo esperado do erro, nunca por acuracia global.",
     "", "Pagina 2 de 3", RODAPE],

    [CABECALHO, "",
     "3. Recomendacao e proximos passos",
     "",
     "Recomenda-se iniciar por uma prova de conceito de oito semanas,",
     "restrita a familia de produtos com maior volume, usando somente",
     "as imagens ja rotuladas. A meta e atingir revocacao superior a",
     "90% na classe defeito, mantendo a taxa de falsos alarmes abaixo",
     "de 5%, patamar considerado aceitavel pela equipe de producao.",
     "",
     "Em paralelo, e necessario estruturar o processo de rotulagem: sem",
     "um fluxo continuo de imagens rotuladas, o modelo perde desempenho",
     "conforme a linha muda de iluminacao, de fornecedor de materia",
     "prima ou de ferramental. A degradacao silenciosa por mudanca de",
     "distribuicao e o risco operacional mais relevante identificado",
     "nesta analise e deve ser monitorado desde o primeiro dia.",
     "", "Pagina 3 de 3", RODAPE],
]

PAGINAS_MANUAL = [
    ["Manual de Qualidade - Capitulo 4", "",
     "4. Registro de nao conformidades",
     "",
     "Toda nao conformidade identificada durante o processo produtivo",
     "deve ser registrada no sistema em ate vinte e quatro horas apos a",
     "sua deteccao. O registro precisa conter, no minimo, a descricao",
     "do desvio observado, o lote afetado, o posto de trabalho onde a",
     "ocorrencia foi detectada e a identificacao do responsavel pelo",
     "apontamento. Registros incompletos sao devolvidos ao emissor.",
     "",
     "A classificacao de severidade segue tres niveis. O nivel critico",
     "aplica-se a desvios com potencial de causar dano ao usuario final",
     "ou de violar requisito regulatorio, e exige bloqueio imediato do",
     "lote. O nivel maior aplica-se a desvios que comprometem a funcao",
     "do produto sem risco a seguranca. O nivel menor cobre desvios de",
     "acabamento e identificacao, tratados no proprio posto.",
     "",
     "A analise de causa raiz e obrigatoria para os niveis critico e",
     "maior, e deve ser concluida em ate dez dias uteis."],

    ["Manual de Qualidade - Capitulo 4", "",
     "4.1 Acoes corretivas e verificacao de eficacia",
     "",
     "Uma acao corretiva so pode ser encerrada apos a verificacao de",
     "eficacia, realizada por pessoa distinta de quem executou a acao.",
     "A verificacao consiste em comprovar, com evidencia objetiva, que",
     "a causa raiz foi eliminada e que o desvio nao voltou a ocorrer em",
     "tres lotes consecutivos apos a implementacao.",
     "",
     "Evidencia objetiva significa dado registrado: carta de controle,",
     "relatorio de inspecao, registro de calibracao ou fotografia com",
     "identificacao de lote e data. Declaracao verbal do responsavel",
     "nao constitui evidencia e nao encerra a acao corretiva.",
     "",
     "Quando a verificacao de eficacia falha, a acao retorna a etapa de",
     "analise de causa raiz, e nao a etapa de execucao. Repetir a mesma",
     "acao esperando resultado diferente e o erro mais comum observado",
     "nas auditorias internas dos ultimos dois ciclos."],
]

NOTICIA = """Fábrica de Contagem começa a testar inspeção automática de peças

A unidade industrial instalada em Contagem, na região metropolitana de Belo
Horizonte, iniciou nesta semana um projeto piloto de inspeção automática de
peças metálicas na própria linha de produção. O sistema utiliza câmeras já
existentes nos postos de trabalho e um modelo de visão computacional treinado
com imagens coletadas ao longo dos últimos oito meses de operação.

Segundo a coordenação de qualidade da unidade, a conferência manual atual
alcança cerca de cinco por cento das peças produzidas, percentual definido por
amostragem estatística. Com a inspeção automática, a expectativa é analisar a
totalidade do lote sem reduzir a cadência da linha, hoje em torno de mil e
duzentas peças por hora.

O projeto piloto tem duração prevista de oito semanas e está restrito à família
de produtos de maior volume. A equipe responsável afirma que o principal
desafio não é técnico, mas de dados: apenas uma fração das imagens disponíveis
possui rótulo confiável atribuído por um inspetor humano, o que limita o
tamanho do conjunto de treinamento e exige atenção redobrada com o
desbalanceamento entre peças conformes e peças com defeito.

A direção industrial informou que a decisão sobre a expansão do sistema para as
demais linhas será tomada apenas após a avaliação dos resultados do piloto,
prevista para o final do próximo trimestre.
"""

# Quase-duplicata: mesmo texto com o título trocado, uma cidade diferente e
# duas frases reescritas. O hash exato NÃO pega isto — só o MinHash pega.
NOTICIA_EDITADA = NOTICIA.replace(
    "Fábrica de Contagem começa a testar inspeção automática de peças",
    "Indústria mineira inicia teste de inspeção automática de peças"
).replace(
    "A unidade industrial instalada em Contagem, na região metropolitana de Belo\nHorizonte, iniciou nesta semana",
    "A unidade industrial de Betim, na Grande Belo Horizonte, iniciou na última\nsegunda-feira"
).replace("oito semanas", "dois meses")

COMUNICADO_LEGADO = """COMUNICADO INTERNO N. 042/2024

Prezados colaboradores,

Informamos que a manutenção preventiva da linha três será realizada no próximo
sábado, das seis às dezoito horas. Durante esse período, o acesso ao galpão B
ficará restrito à equipe de manutenção e aos técnicos da empresa contratada.

Solicitamos que todos os apontamentos de produção referentes à sexta-feira
sejam lançados no sistema até as dezessete horas, para que o inventário possa
ser conciliado antes da parada. Apontamentos lançados após esse horário serão
contabilizados apenas na segunda-feira seguinte, o que distorce os indicadores
diários de eficiência e atrapalha a análise do turno.

Em caso de dúvida sobre o procedimento de lançamento, procure a supervisão do
seu turno ou consulte o manual disponível na intranet. A área de manutenção
permanecerá disponível pelo ramal duzentos e trinta e quatro durante toda a
execução dos serviços programados.

Atenciosamente,
Coordenação de Manutenção Industrial
"""

MENU_SITE = """Início
Sobre nós
Produtos
Serviços
Contato
Blog
Trabalhe conosco
Política de Privacidade
Termos de Uso
Fale conosco
Leia mais
Clique aqui
Compartilhe
Voltar ao topo
Este site utiliza cookies para melhorar sua experiência.
Aceitar todos os cookies
© 2024 ACME Tecnologia. Todos os direitos reservados.
CNPJ: 12.345.678/0001-90
Página 1 de 1
"""

LEIA_ME = "Pasta de documentos do projeto piloto. Ver relatorio principal.\n"

# Saída típica de OCR de má qualidade: letras trocadas por símbolos, espaçamento
# destruído, caracteres de controle de página.
OCR_RUIM = """R3l4t0ri0 d3 |nsp3ç4o ~~ 1ote 88#2

Pç. c0nf0rm3s: 1.2O4 || Pç. n4o c0nf: 3l
0bs3rv4ç~0es: m4rc4s d3 0x1d4ç4o n0 fl4ng3 d1r31t0 ####
V3r1f1c4r c4l1br4ç~4o d0 p4qu1m3tr0 n° 7 §§§ [[[ ]]] ~~~~
D4t4: l2/O3/2O24 -- Insp3t0r: J. S1lv4 %%%%%%
@@@@ ****** ###### |||||| ~~~~~~ ^^^^^^ <<<<<< >>>>>>
0BS: p4g1n4 d1g1t4l1z4d4 c0m r3s0luç4o 1nsuf1c13nt3 (150 dp1)
R3c0m3nd4-s3 r3d1g1t4l1z4ç4o 3m 3OO dp1 p4r4 n0v4 t3nt4t1v4 d3 0CR
"""

ABSTRACT_INGLES = """Automated Visual Inspection in Discrete Manufacturing: A Case Study

This paper presents a case study on the deployment of an automated visual
inspection system in a discrete manufacturing environment. We describe the data
collection process, the labeling protocol adopted by the quality team, and the
architecture of the convolutional network used for defect detection.

The main challenge reported by the engineering team was not model accuracy but
label scarcity and class imbalance. Only fifteen percent of the collected
images carried a reliable label, and conforming parts accounted for more than
ninety six percent of the labeled set. We discuss how these constraints shaped
the evaluation protocol, which prioritizes recall on the defect class over
overall accuracy.

Results from an eight week pilot indicate that recall above ninety percent is
achievable while keeping the false alarm rate below five percent, provided that
a continuous labeling workflow is established to counteract distribution shift
caused by changes in lighting and raw material suppliers.
"""

PARAGRAFOS_DOCX = [
    ("Procedimento Operacional Padrao POP-QA-014", "titulo"),
    ("Objetivo", "subtitulo"),
    ("Estabelecer o criterio de aceitacao visual das pecas metalicas "
     "produzidas na linha tres, de modo que a decisao de aprovar ou reprovar "
     "uma peca nao dependa do julgamento individual do inspetor. Este "
     "procedimento aplica-se a todos os turnos e a todos os postos de "
     "inspecao da unidade.", "corpo"),
    ("Campo de aplicacao", "subtitulo"),
    ("Aplica-se as familias de produto A, B e C. Pecas sob encomenda seguem "
     "criterio especifico definido em contrato com o cliente, registrado na "
     "ordem de producao correspondente. Em caso de divergencia entre este "
     "procedimento e o contrato, prevalece o contrato.", "corpo"),
    ("Criterios de aceitacao", "subtitulo"),
    ("A tabela a seguir consolida os limites aceitaveis por tipo de defeito. "
     "Valores acima do limite implicam reprovacao da peca e abertura de "
     "registro de nao conformidade conforme o Manual de Qualidade.", "corpo"),
]

TABELA_DOCX = [
    ["Tipo de defeito", "Limite aceitavel", "Metodo de medicao", "Severidade"],
    ["Risco superficial", "ate 2 mm de extensao", "Gabarito visual G-04", "Menor"],
    ["Oxidacao localizada", "nao aceitavel", "Inspecao visual direta", "Critico"],
    ["Rebarba na aresta", "ate 0,5 mm de altura", "Paquimetro digital", "Maior"],
    ["Deformacao do flange", "ate 1 mm de desvio", "Relogio comparador", "Critico"],
    ["Falha de pintura", "ate 3 mm de diametro", "Gabarito visual G-07", "Menor"],
]

# Planilha: a coluna "Comentario" é texto natural; as demais são metadados.
AVALIACOES = [
    ["ID", "Data", "Nota", "Canal", "Comentario"],
    [1001, "2024-03-04", 2, "telefone",
     "A peca chegou com uma marca de oxidacao bem visivel na aresta lateral, e "
     "o lote inteiro precisou ser conferido de novo aqui na nossa recepcao. "
     "Perdemos quase um dia de producao por causa dessa conferencia extra."],
    [1002, "2024-03-05", 5, "email",
     "Atendimento rapido e prazo cumprido conforme o combinado no pedido. A "
     "documentacao veio completa e o certificado de qualidade estava anexado, "
     "o que facilitou muito a liberacao pela nossa area de recebimento."],
    [1003, "2024-03-06", 3, "formulario",
     "O produto atende ao especificado, mas a embalagem chegou amassada em "
     "dois volumes e uma das etiquetas de identificacao estava ilegivel. "
     "Sugiro revisar o processo de paletizacao antes do envio."],
    [1004, "2024-03-08", 1, "telefone",
     "Recebemos pecas com rebarba acima do limite acordado em contrato e sem o "
     "relatorio de inspecao que combinamos na ultima reuniao tecnica. "
     "Precisamos de um posicionamento formal sobre esse desvio."],
    [1005, "2024-03-11", 4, "email",
     "De modo geral o fornecimento tem sido regular e a qualidade melhorou "
     "bastante em relacao ao trimestre passado. O unico ponto de atencao "
     "continua sendo o prazo de resposta as solicitacoes de assistencia."],
    [1006, "2024-03-12", 2, "formulario",
     "A ultima remessa apresentou variacao de tonalidade na pintura entre as "
     "pecas do mesmo lote, o que fica evidente quando elas sao montadas lado a "
     "lado no conjunto final. Isso gerou reclamacao do nosso cliente."],
]

RESUMO_NUMERICO = [
    ["Mes", "Pedidos", "Devolucoes", "Taxa", "Faturamento"],
    [1, 412, 18, 0.0437, 284500.0],
    [2, 388, 15, 0.0387, 261300.0],
    [3, 455, 27, 0.0593, 312750.0],
    [4, 401, 12, 0.0299, 275900.0],
]

# Este é o único arquivo do acervo com acentuação E codificação cp1252: são os
# acentos que fazem a decodificação UTF-8 falhar e a cadeia de tentativas de
# `readers.decodificar` entrar em ação. Sem acento, cp1252 e UTF-8 coincidem e
# o problema simplesmente não aparece.
CHAMADOS = [
    ["Chamado", "Abertura", "Prioridade", "Descrição"],
    ["CH-4471", "2024-03-04", "Alta",
     "Cliente relata que o sistema de apontamento não registra as peças do "
     "terceiro turno desde a atualização de ontem; os lançamentos somem após "
     "a confirmação e não aparecem no relatório diário do supervisor."],
    ["CH-4472", "2024-03-05", "Média",
     "Solicitação de novo perfil de acesso para dois inspetores recém "
     "contratados, com permissão de leitura nos relatórios de qualidade e "
     "escrita apenas nos registros do próprio posto de trabalho."],
    ["CH-4473", "2024-03-06", "Baixa",
     "Usuário pede alteração do texto padrão do email automático de "
     "notificação, que hoje menciona um procedimento revogado e confunde os "
     "supervisores que recebem a mensagem todas as manhãs."],
    ["CH-4474", "2024-03-07", "Alta",
     "A câmera do posto sete parou de enviar imagens para o servidor durante "
     "a madrugada e a inspeção ficou sem registro fotográfico por quatro "
     "horas, comprometendo a rastreabilidade do lote produzido."],
]


# ===========================================================================
# Escrita dos arquivos
# ===========================================================================
def escrever_docx(caminho: Path) -> None:
    """DOCX com parágrafos E tabela — a tabela é o teste do leitor."""
    try:
        import docx
    except ImportError:
        print("[aviso] python-docx não instalado; pulando o .docx")
        return

    documento = docx.Document()
    for texto, estilo in PARAGRAFOS_DOCX:
        if estilo == "titulo":
            documento.add_heading(texto, level=1)
        elif estilo == "subtitulo":
            documento.add_heading(texto, level=2)
        else:
            documento.add_paragraph(texto)

    tabela = documento.add_table(rows=0, cols=len(TABELA_DOCX[0]))
    tabela.style = "Table Grid"
    for linha in TABELA_DOCX:
        celulas = tabela.add_row().cells
        for celula, valor in zip(celulas, linha):
            celula.text = valor

    documento.add_paragraph(
        "Este procedimento deve ser revisado anualmente ou sempre que houver "
        "alteracao no processo produtivo que afete os criterios acima "
        "estabelecidos para a inspecao visual das pecas.")
    documento.save(str(caminho))


def escrever_xlsx(caminho: Path) -> None:
    """Planilha com duas abas: uma textual e uma puramente numérica."""
    try:
        from openpyxl import Workbook
    except ImportError:
        print("[aviso] openpyxl não instalado; pulando o .xlsx")
        return

    livro = Workbook()
    aba = livro.active
    aba.title = "Avaliacoes"
    for linha in AVALIACOES:
        aba.append(linha)

    resumo = livro.create_sheet("Resumo")
    for linha in RESUMO_NUMERICO:
        resumo.append(linha)

    livro.save(str(caminho))


def escrever_csv(caminho: Path) -> None:
    """CSV com separador ';' e codificação cp1252 — o padrão do Excel brasileiro."""
    buffer = io.StringIO()
    escritor = csv.writer(buffer, delimiter=";", quoting=csv.QUOTE_MINIMAL,
                          lineterminator="\n")
    escritor.writerows(CHAMADOS)
    caminho.write_bytes(buffer.getvalue().encode("cp1252", errors="replace"))


def gerar(forcar: bool = False) -> None:
    BRUTOS_DIR.mkdir(parents=True, exist_ok=True)

    if any(BRUTOS_DIR.iterdir()) and not forcar:
        print(f"[!] {BRUTOS_DIR} já contém arquivos. Use --forcar para sobrescrever.")
        return

    arquivos: list[tuple[str, str]] = []

    def txt(nome: str, conteudo: str, codec: str = "utf-8", nota: str = ""):
        (BRUTOS_DIR / nome).write_bytes(conteudo.encode(codec, errors="replace"))
        arquivos.append((nome, nota or f"texto {codec}"))

    escrever_pdf(BRUTOS_DIR / "relatorio_tecnico.pdf", PAGINAS_RELATORIO)
    arquivos.append(("relatorio_tecnico.pdf", "3 páginas, cabeçalho/rodapé + hifenização"))

    escrever_pdf(BRUTOS_DIR / "manual_qualidade.pdf", PAGINAS_MANUAL)
    arquivos.append(("manual_qualidade.pdf", "2 páginas de prosa limpa"))

    escrever_pdf(BRUTOS_DIR / "nota_fiscal_escaneada.pdf", [[]], escaneado=True)
    arquivos.append(("nota_fiscal_escaneada.pdf", "SEM camada de texto (precisa de OCR)"))

    txt("noticia.txt", NOTICIA, "utf-8", "documento saudável (UTF-8)")
    txt("noticia_copia.txt", NOTICIA, "utf-8", "duplicata EXATA de noticia.txt")
    txt("noticia_editada.txt", NOTICIA_EDITADA, "utf-8", "QUASE-duplicata")
    txt("comunicado_legado.txt", COMUNICADO_LEGADO, "cp1252", "codificado em cp1252")
    txt("menu_site.txt", MENU_SITE, "utf-8", "boilerplate de navegação")
    txt("leia-me.txt", LEIA_ME, "utf-8", "curto demais")
    txt("ocr_ruim.txt", OCR_RUIM, "utf-8", "ruído de OCR")
    txt("abstract.txt", ABSTRACT_INGLES, "utf-8", "inglês (idioma fora do alvo)")

    escrever_docx(BRUTOS_DIR / "procedimento_operacional.docx")
    arquivos.append(("procedimento_operacional.docx", "parágrafos + tabela"))

    escrever_xlsx(BRUTOS_DIR / "avaliacoes_clientes.xlsx")
    arquivos.append(("avaliacoes_clientes.xlsx", "aba textual + aba numérica"))

    escrever_csv(BRUTOS_DIR / "chamados.csv")
    arquivos.append(("chamados.csv", "separador ';', cp1252"))

    print(f"Acervo gerado em {BRUTOS_DIR}\n")
    print(f"{'arquivo':<34}{'patologia embutida'}")
    print("-" * 78)
    for nome, nota in arquivos:
        existe = (BRUTOS_DIR / nome).exists()
        print(f"{nome:<34}{nota}" + ("" if existe else "   [NÃO GERADO]"))
    print(f"\n{len([a for a in arquivos if (BRUTOS_DIR / a[0]).exists()])} arquivos.")
    print("Próximo passo:  python src/readers.py --texto")


if __name__ == "__main__":
    configurar_terminal()
    p = argparse.ArgumentParser(description="Gera o acervo de exemplo")
    p.add_argument("--forcar", action="store_true",
                   help="sobrescreve arquivos existentes")
    gerar(p.parse_args().forcar)
