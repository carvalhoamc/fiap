"""
config.py — ETAPA 0: ponto único de configuração
-------------------------------------------------

Por que centralizar? Porque um pipeline de dados só é reprodutível se todos os
"botões" (limiares de filtragem, caminhos, semente aleatória) estiverem escritos
em um lugar só, versionados junto com o código.

Em pipelines de texto isso é ainda mais crítico do que em visão computacional:
os limiares abaixo DECIDEM QUAIS DOCUMENTOS ENTRAM NO CORPUS. Se eles estiverem
espalhados como números mágicos dentro dos scripts, ninguém — nem você, seis
meses depois — consegue explicar por que um documento foi descartado.

Regra da disciplina: **todo limiar aqui é um chute inicial**. Ele só vira uma
decisão defensável depois que você lê o arquivo `rejeitados.jsonl` e confirma
que o que está sendo jogado fora é realmente lixo.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos: sempre relativos à raiz do projeto, nunca ao diretório de onde
# você chamou o script. parents[1] sobe de src/ para nlp1/.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"            # arquivos de entrada
BRUTOS_DIR = DATA_DIR / "brutos"    # os pdf/txt/docx/planilhas originais
OUT_DIR = ROOT / "outputs"          # corpus, rejeitados, relatórios e gráficos


@dataclass
class Config:
    # --- ETAPA 1: aquisição ---------------------------------------------
    # Extensões que o pipeline sabe ler. Qualquer outra é registrada como
    # "formato não suportado" — e NUNCA ignorada em silêncio.
    extensoes: tuple = (".txt", ".md", ".pdf", ".docx", ".xlsx", ".xlsm", ".csv")

    # Ordem de tentativa para decodificar arquivos de texto. UTF-8 primeiro
    # porque é o padrão moderno; cp1252/latin-1 porque é o que sai de sistemas
    # legados brasileiros (e latin-1 nunca falha, então tem de ser o último).
    codificacoes: tuple = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

    tamanho_maximo_mb: float = 50.0     # arquivos maiores exigem streaming
    max_paginas_pdf: int = 0            # 0 = todas; use >0 para PDFs gigantes

    # Planilhas: uma coluna só vira texto se tiver comprimento médio decente e
    # não for majoritariamente numérica. Ver readers.detectar_colunas_textuais.
    min_comprimento_medio_coluna: float = 25.0
    max_prop_numerica_coluna: float = 0.5
    linhas_amostra_planilha: int = 50   # quantas linhas olhar para decidir

    # --- ETAPA 2: limpeza ------------------------------------------------
    max_linhas_em_branco: int = 1       # colapsa 5 linhas vazias em 1
    # Uma linha que aparece em pelo menos esta fração das páginas do MESMO
    # arquivo é cabeçalho/rodapé, não conteúdo.
    limiar_cabecalho_rodape: float = 0.6
    min_paginas_para_cabecalho: int = 3  # com 2 páginas, a estatística não vale
    mascarar_pii: bool = False          # LGPD: ver clean.mascarar_pii()

    # --- ETAPA 3: filtragem ---------------------------------------------
    # Estes dois limiares foram CALIBRADOS olhando a tabela de
    # `python src/filters.py` sobre o acervo de exemplo, e não escolhidos por
    # gosto. O valor "natural" para páginas de PDF seria bem mais alto (40
    # palavras passa tranquilo), mas uma LINHA de planilha tem entre 29 e 38
    # palavras: um limiar calibrado só para PDF apagaria as duas fontes
    # tabulares inteiras. Suba para 40 e rode de novo para ver o estrago —
    # é exatamente o exercício 2.2.
    min_caracteres: int = 150           # abaixo disso não há contexto útil
    min_palavras: int = 25
    min_prop_alfabetica: float = 0.60   # letras / total de caracteres
    max_prop_digitos: float = 0.30      # dump de planilha, tabela de números
    max_prop_simbolos: float = 0.10     # ruído típico de OCR ruim
    max_prop_maiusculas: float = 0.40   # TÍTULOS EM CAIXA ALTA, menus, banners
    min_comprimento_palavra: float = 2.5   # texto quebrado caractere a caractere
    max_comprimento_palavra: float = 12.0  # base64, hashes, URLs coladas
    min_prop_stopwords: float = 0.06    # detector de idioma do pobre (ver filters)
    max_prop_linhas_repetidas: float = 0.30  # listas, menus, boilerplate
    max_prop_boilerplate: float = 0.20  # linhas com padrões jurídicos/navegação

    # --- ETAPA 4: deduplicação -------------------------------------------
    tamanho_shingle: int = 5            # n-gramas de palavras
    n_permutacoes: int = 64             # assinaturas MinHash
    n_bandas: int = 16                  # LSH: 16 bandas x 4 linhas = 64
    # Acima deste valor, dois documentos são considerados quase-duplicatas.
    # A literatura usa a faixa 0,7-0,8. Escolhemos 0,75 depois de MEDIR: no
    # acervo de exemplo, `noticia_editada.txt` tem similaridade 0,783 com
    # `noticia.txt` — ou seja, com o limiar em 0,80 ela SOBREVIVE, e com 0,75
    # é capturada. A diferença entre pegar e não pegar essa duplicata é de
    # 0,017 no limiar. Rode `python src/dedup.py --limiar 0.85` para ver.
    limiar_jaccard: float = 0.75
    semente: int = 42

    # --- ETAPA 5: segmentação --------------------------------------------
    tamanho_bloco: int = 900            # caracteres por bloco (chunk)
    sobreposicao_bloco: int = 150       # janela de sobreposição entre blocos

    # --- Arquivos de saída ------------------------------------------------
    arq_corpus: str = "corpus.jsonl"
    arq_rejeitados: str = "rejeitados.jsonl"
    arq_duplicatas: str = "duplicatas.jsonl"
    arq_blocos: str = "blocos.jsonl"
    arq_relatorio: str = "relatorio.json"
    arq_grafico_motivos: str = "motivos_rejeicao.png"
    arq_grafico_tamanhos: str = "distribuicao_tamanhos.png"
    arq_grafico_formatos: str = "documentos_por_formato.png"
    arq_metadados: str = "metadados_corpus.json"

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()


# ---------------------------------------------------------------------------
# Stopwords do português. Usadas como detector de idioma barato: todo texto
# natural em português tem uma fração alta de "de, a, o, que, e, do, da...".
# Um PDF em inglês, uma listagem de códigos ou uma tabela de números têm
# fração próxima de zero. Não substitui um langdetect/fastText, mas é
# transparente, instantâneo e não adiciona dependência — e você entende
# exatamente por que um documento foi recusado.
# ---------------------------------------------------------------------------
STOPWORDS_PT = frozenset("""
a à às ao aos aquela aquelas aquele aqueles aquilo as até com como da das de
dela delas dele deles depois do dos e é ela elas ele eles em entre era eram
essa essas esse esses esta estas este estes eu foi fomos for foram há isso isto
já lhe lhes mais mas me mesmo meu meus minha minhas muito na não nas nem no nos
nós nossa nosso num numa o os ou para pela pelas pelo pelos por qual quando que
quem são se sem ser seu seus só sobre sua suas também te tem têm ter teu teus
tu tua tuas um uma umas uns você vocês
""".split())

# ---------------------------------------------------------------------------
# Dois conjuntos de padrões, com PAPÉIS DIFERENTES no pipeline. A distinção é
# uma das ideias centrais da aula:
#
#   MOLDURA     -> mobiliário de página: numeração, marca d'água de conversão,
#                  linha de copyright. É lixo LOCAL: some a linha, o documento
#                  continua bom. Logo, isto se RESOLVE NA LIMPEZA (ETAPA 2),
#                  removendo a linha. Rejeitar um relatório inteiro porque ele
#                  tem "Página 1 de 3" no rodapé seria absurdo.
#
#   BOILERPLATE -> evidência de que o documento É moldura: menu de navegação,
#                  aviso de cookie, rodapé jurídico, chamadas de clique. Aqui a
#                  linha isolada é inofensiva, mas um documento CHEIO delas não
#                  tem conteúdo nenhum. Logo, isto vira uma PROPORÇÃO medida na
#                  FILTRAGEM (ETAPA 3).
#
# Resumindo: a limpeza remove a moldura; a filtragem descarta o que é só
# moldura. Confundir os dois papéis é o erro que produz um pipeline que joga
# fora justamente os documentos bem formatados.
# ---------------------------------------------------------------------------
PADROES_MOLDURA = (
    r"^\s*p[áa]g(?:\.|ina)?\s*\d+\s*(?:de\s*\d+)?\s*$",
    r"^\s*-\s*\d+\s*-\s*$",              # "- 12 -", numeração centralizada
    r"^\s*©\s*\d{4}.{0,80}$",
    r"gerado (?:automaticamente|por)\b",
    r"\b(?:converted by|created with|generated by)\b",
    r"^\s*voltar ao topo\s*$",
)

PADROES_BOILERPLATE = (
    r"todos os direitos reservados",
    r"all rights reserved",
    r"pol[íi]tica de privacidade",
    r"termos de uso",
    r"aceit(?:ar|e) (?:todos os )?cookies",
    r"este site utiliza cookies",
    r"clique aqui",
    r"leia mais",
    r"^\s*compartilhe\s*$",
    r"^\s*(?:in[íi]cio|home|menu|contato|sobre n[óo]s|produtos|servi[çc]os|"
    r"blog|trabalhe conosco|fale conosco)\s*$",
    r"cnpj[\s:]*\d",
    r"^\s*p[áa]gina\s+\d+\s+de\s+\d+\s*$",
)

# Rótulos legíveis dos motivos de rejeição — usados nos relatórios e gráficos.
# Se um motivo aparece aqui, ele PRECISA aparecer no rejeitados.jsonl.
MOTIVOS = {
    "vazio": "Sem texto extraível (PDF escaneado?)",
    "curto_demais": "Curto demais",
    "pouca_letra": "Poucos caracteres alfabéticos",
    "muito_numerico": "Excesso de dígitos (tabela?)",
    "ruido_simbolos": "Ruído de símbolos (OCR?)",
    "caixa_alta": "Excesso de maiúsculas",
    "palavra_curta": "Palavras curtas demais",
    "palavra_longa": "Palavras longas demais",
    "outro_idioma": "Não parece português",
    "repetitivo": "Linhas repetidas em excesso",
    "boilerplate": "Boilerplate/navegação",
    "duplicata_exata": "Duplicata exata",
    "quase_duplicata": "Quase-duplicata",
}
