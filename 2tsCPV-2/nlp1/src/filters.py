"""
filters.py — ETAPA 3 do pipeline: filtragem de qualidade
----------------------------------------------------------

Filtrar é decidir o que ENTRA no corpus. É a etapa de maior impacto e a de
maior risco do pipeline inteiro, por um motivo simples: o que você descarta
aqui, o modelo nunca vê, e nenhum ajuste posterior recupera.

A literatura de construção de corpora (C4, Gopher, RefinedWeb) converge para
um mesmo conjunto de heurísticas baratas, e é ele que está implementado aqui.
Todas medem a mesma coisa por ângulos diferentes: **isto parece prosa escrita
por uma pessoa, ou parece resíduo de extração?**

Três princípios que valem mais que os limiares:

1. NENHUM DESCARTE SILENCIOSO. Todo documento rejeitado vai para
   `rejeitados.jsonl` com o motivo e as métricas que levaram à decisão. Um
   filtro que você não consegue auditar é um bug que você não consegue ver.

2. UM MOTIVO POR DOCUMENTO, mas todos registrados. O motivo PRINCIPAL é o
   primeiro filtro que reprovou, na ordem em que estão declarados — do defeito
   mais grosseiro para o mais sutil. Os demais motivos ficam guardados, porque
   documento ruim costuma ser ruim de várias maneiras ao mesmo tempo.

3. LIMIAR É HIPÓTESE, NÃO VERDADE. Os números em `config.py` são um chute
   inicial razoável. Calibrá-los é olhar o `rejeitados.jsonl` e responder duas
   perguntas: "o que está sendo jogado fora é mesmo lixo?" (precisão do filtro)
   e "o que passou é mesmo bom?" (revocação). Um filtro nunca é avaliado pelo
   número que ele produz, e sim pelos documentos que ele move de lado.

Uso:
    python src/filters.py                  # tabela de métricas de todo o acervo
    python src/filters.py --rejeitados     # só os reprovados, com o motivo
"""

import argparse
import re
import unicodedata

from config import CFG, MOTIVOS, PADROES_BOILERPLATE, STOPWORDS_PT
from utils import configurar_terminal, palavras

# Compilar uma vez, no import — e não a cada documento. Compilar 17 expressões
# regulares por documento, em um corpus de milhões, é hora de CPU jogada fora.
REGEX_BOILERPLATE = [re.compile(p, re.IGNORECASE) for p in PADROES_BOILERPLATE]

# Pontuação que a prosa em português realmente usa. Tudo que NÃO for letra,
# dígito, espaço ou um destes conta como "símbolo".
#
# Repare no que ficou DE FORA de propósito: # * + = < > _ \ | ~ ^. Eles não
# aparecem em texto corrido, e são exatamente a assinatura do OCR ruim
# (|||, ###, ~~~, <<<) e do resíduo de marcação. Uma versão anterior deste
# arquivo os incluía como "normais" — e o resultado foi um `prop_simbolos`
# que dava 0,00 para uma página de OCR destruída. Métrica que não separa
# nada é pior que métrica nenhuma: ela dá falsa sensação de cobertura.
PONTUACAO_NORMAL = set(".,;:!?()[]{}\"'`´-–—/%&@ºª°§€$")


# ===========================================================================
# Medição: calcular uma vez, decidir depois
# ===========================================================================
def medir(texto: str, cfg=CFG) -> dict:
    """Calcula todas as métricas de qualidade de um texto, de uma só vez.

    Separar MEDIR de DECIDIR é uma escolha de projeto que vale explicar: as
    métricas vão para o relatório mesmo quando o documento é aprovado. É isso
    que permite plotar a distribuição de cada métrica e enxergar onde o limiar
    deveria estar — em vez de adivinhar. Se medir e decidir estivessem no mesmo
    laço, você só saberia o veredito, nunca a margem.
    """
    n_caracteres = len(texto)
    if n_caracteres == 0:
        return {"n_caracteres": 0, "n_palavras": 0, "n_linhas": 0,
                "prop_alfabetica": 0.0, "prop_digitos": 0.0, "prop_simbolos": 0.0,
                "prop_maiusculas": 0.0, "comprimento_palavra": 0.0,
                "prop_stopwords": 0.0, "prop_linhas_repetidas": 0.0,
                "prop_boilerplate": 0.0}

    n_alfabeticos = n_digitos = n_simbolos = n_maiusculas = 0
    for c in texto:
        if c.isalpha():
            n_alfabeticos += 1
            if c.isupper():
                n_maiusculas += 1
        elif c.isdigit():
            n_digitos += 1
        elif not c.isspace() and c not in PONTUACAO_NORMAL:
            # unicodedata pega os casos exóticos: emojis, setas, blocos de
            # desenho, caracteres de moeda estrangeira.
            if unicodedata.category(c)[0] in "SPC":
                n_simbolos += 1

    lista = palavras(texto)
    n_palavras = len(lista)
    n_stopwords = sum(1 for p in lista if p in STOPWORDS_PT)
    comprimento_medio = (sum(len(p) for p in lista) / n_palavras) if n_palavras else 0.0

    linhas = [l.strip() for l in texto.split("\n") if l.strip()]
    n_linhas = len(linhas)
    n_unicas = len(set(linhas))

    # Boilerplate medido em CARACTERES, não em linhas. O motivo é sutil e vale
    # a atenção: depois que a limpeza rejunta os parágrafos, uma página inteira
    # de PDF vira 3 ou 4 "linhas" enormes. Uma única linha de rodapé entre elas
    # dá 25% das linhas — e reprovaria o relatório inteiro. Em caracteres, essa
    # mesma linha de rodapé pesa 1,5%, que é o peso que ela de fato tem.
    # Métrica normalizada pela unidade errada é uma das formas mais comuns de
    # um filtro razoável produzir um resultado absurdo.
    n_car_boilerplate = sum(len(l) for l in linhas
                            if any(r.search(l) for r in REGEX_BOILERPLATE))
    n_car_linhas = sum(len(l) for l in linhas)

    return {
        "n_caracteres": n_caracteres,
        "n_palavras": n_palavras,
        "n_linhas": n_linhas,
        "prop_alfabetica": n_alfabeticos / n_caracteres,
        "prop_digitos": n_digitos / n_caracteres,
        "prop_simbolos": n_simbolos / n_caracteres,
        # Proporção de maiúsculas entre as LETRAS, não entre os caracteres:
        # senão um texto cheio de números pareceria ter poucas maiúsculas.
        "prop_maiusculas": (n_maiusculas / n_alfabeticos) if n_alfabeticos else 0.0,
        "comprimento_palavra": comprimento_medio,
        "prop_stopwords": (n_stopwords / n_palavras) if n_palavras else 0.0,
        "prop_linhas_repetidas": (1 - n_unicas / n_linhas) if n_linhas else 0.0,
        "prop_boilerplate": (n_car_boilerplate / n_car_linhas) if n_car_linhas else 0.0,
    }


# ===========================================================================
# Os filtros
# ===========================================================================
# Cada filtro é uma função pura das métricas: devolve True quando REPROVA.
# A ordem da tupla FILTROS é a ordem de prioridade do motivo principal.

def f_vazio(m, cfg):
    """Nada extraído. Em PDF, quase sempre significa página escaneada.

    Este é o achado mais importante que a filtragem produz, e por isso vem
    primeiro: ele não diz "o documento é ruim", diz "a ETAPA 1 falhou neste
    arquivo". A resposta certa não é descartar e seguir — é rodar OCR e
    reingerir. Um relatório que mostra 30% de páginas vazias está dizendo que
    o seu acervo precisa de um extrator diferente.
    """
    return m["n_caracteres"] == 0 or m["n_palavras"] == 0


def f_curto_demais(m, cfg):
    """Texto curto não tem contexto suficiente para ser útil.

    Vale para qualquer uso a jusante: classificar, sumarizar ou recuperar. Um
    fragmento de dez palavras entra no índice, casa com qualquer consulta por
    acaso e não responde nada.

    Atenção ao efeito colateral, visível no acervo desta aula: uma LINHA de
    planilha tem naturalmente menos palavras que uma PÁGINA de PDF. Um limiar
    único calibrado para PDF apaga a planilha inteira. É o exercício 2.2.
    """
    return m["n_caracteres"] < cfg.min_caracteres or m["n_palavras"] < cfg.min_palavras


def f_ruido_simbolos(m, cfg):
    """Símbolos demais: OCR de má qualidade, marcação vazada, arte ASCII.

    Texto em português tem pouquíssimo caractere fora de letras, dígitos e
    pontuação comum. Quando a proporção sobe, quase sempre é o reconhecimento
    óptico trocando letras por símbolos parecidos (ç -> §, o -> 0, l -> |).
    """
    return m["prop_simbolos"] > cfg.max_prop_simbolos


def f_muito_numerico(m, cfg):
    """Dígitos demais: tabela, extrato, log, dump de planilha.

    Não é lixo em termos absolutos — é dado tabular, e dado tabular pertence a
    um banco de dados, não a um corpus de texto. Misturar os dois faz o modelo
    gastar capacidade decorando números que não significam nada fora da tabela.
    """
    return m["prop_digitos"] > cfg.max_prop_digitos


def f_pouca_letra(m, cfg):
    """Poucas letras no total: o texto não é predominantemente linguagem.

    Rede de segurança para o que escapou dos dois filtros anteriores — a
    combinação de muitos dígitos com muitos símbolos, cada um abaixo do seu
    próprio limiar, mas que somados não deixam quase nada de prosa.
    """
    return m["prop_alfabetica"] < cfg.min_prop_alfabetica


def f_caixa_alta(m, cfg):
    """MAIÚSCULAS DEMAIS: banner, menu, cabeçalho de formulário, aviso legal.

    Nenhuma prosa passa de ~10% de maiúsculas. Acima de 40% não é texto
    corrido: é rótulo de interface ou título solto que sobrou da extração.
    """
    return m["prop_maiusculas"] > cfg.max_prop_maiusculas


def f_palavra_curta(m, cfg):
    """Palavras curtas demais: o texto foi quebrado onde não devia.

    O sintoma clássico de extração espaçada caractere a caractere
    ("R E L A T Ó R I O"), que algumas fontes de PDF produzem. O comprimento
    médio de palavra do português escrito fica entre 4 e 6 letras.
    """
    return m["comprimento_palavra"] < cfg.min_comprimento_palavra


def f_palavra_longa(m, cfg):
    """Palavras longas demais: espaços perdidos, hashes, base64, URLs coladas.

    O extremo oposto: quando o extrator não separa as palavras, o texto vira
    uma sequência de "megapalavras" que nenhum tokenizador reconhece.
    """
    return m["comprimento_palavra"] > cfg.max_comprimento_palavra


def f_repetitivo(m, cfg):
    """Linhas repetidas em excesso dentro do mesmo documento.

    Formulário com o mesmo campo em branco cem vezes, índice remissivo,
    listagem de produtos com a mesma frase de rodapé. Repetição é o tipo de
    ruído que mais engana um filtro de tamanho: o documento é longo e, ainda
    assim, não tem quase nenhuma informação.
    """
    return m["prop_linhas_repetidas"] > cfg.max_prop_linhas_repetidas


def f_boilerplate(m, cfg):
    """Moldura demais: navegação de site, aviso de cookie, rodapé jurídico.

    Complementa a remoção estatística de cabeçalho/rodapé da ETAPA 2, que só
    enxerga a linha CONSTANTE. Aqui pegamos a linha VARIÁVEL de forma fixa —
    "Página 1 de 3", "Página 2 de 3" — que nunca acumula frequência suficiente
    para ser detectada por repetição.
    """
    return m["prop_boilerplate"] > cfg.max_prop_boilerplate


def f_outro_idioma(m, cfg):
    """Fração de stopwords do português abaixo do mínimo.

    Detector de idioma do pobre, e assumidamente frágil — está por último
    justamente por isso. A intuição: todo texto natural em português é feito
    de "de, a, o, que, e, do, da" em proporção alta e muito estável (20% a 40%
    das palavras). Um texto em inglês, uma listagem de códigos ou uma tabela
    ficam bem abaixo.

    Onde falha: textos curtos (a estatística não se sustenta), espanhol (que
    compartilha várias stopwords) e português técnico cheio de termos em
    inglês. Para valer em produção, troque por fastText `lid.176` ou
    `langdetect`, que dão idioma E confiança. Este é o exercício 3.4.
    """
    return m["prop_stopwords"] < cfg.min_prop_stopwords


# A ordem AQUI é a ordem do motivo principal: do defeito mais grosseiro e mais
# certo (não tem texto) para o mais sutil e mais discutível (parece outro
# idioma). Reordenar esta tupla muda os rótulos do relatório sem mudar uma
# única decisão de aceitar/rejeitar — o conjunto aprovado é o mesmo.
FILTROS = (
    ("vazio", f_vazio),
    ("curto_demais", f_curto_demais),
    ("ruido_simbolos", f_ruido_simbolos),
    ("muito_numerico", f_muito_numerico),
    ("pouca_letra", f_pouca_letra),
    ("caixa_alta", f_caixa_alta),
    ("palavra_curta", f_palavra_curta),
    ("palavra_longa", f_palavra_longa),
    ("repetitivo", f_repetitivo),
    ("boilerplate", f_boilerplate),
    ("outro_idioma", f_outro_idioma),
)


def avaliar(texto: str, cfg=CFG) -> tuple[str | None, list[str], dict]:
    """Avalia um texto. Devolve (motivo_principal, todos_os_motivos, metricas).

    motivo_principal é None quando o documento é aprovado.

    Repare que TODOS os filtros são executados, mesmo depois do primeiro que
    reprova. Custa alguns microssegundos e paga caro no diagnóstico: saber que
    um documento foi reprovado por "curto_demais" E "outro_idioma" E
    "boilerplate" conta uma história bem diferente de saber só o primeiro.
    """
    metricas = medir(texto, cfg)
    motivos = [nome for nome, funcao in FILTROS if funcao(metricas, cfg)]
    return (motivos[0] if motivos else None), motivos, metricas


if __name__ == "__main__":
    configurar_terminal()
    # Rode:  python src/filters.py
    # Esta tabela é a ferramenta de CALIBRAÇÃO dos limiares. Olhe as colunas
    # dos documentos aprovados e dos reprovados: se elas se sobrepõem, o seu
    # limiar está no lugar errado — ou a métrica não separa o que você quer.
    from clean import limpar_arquivo
    from readers import ErroDeLeitura, ler_documento, listar_arquivos
    from utils import truncar

    p = argparse.ArgumentParser(description="Métricas e decisão de filtragem")
    p.add_argument("--rejeitados", action="store_true", help="mostrar só os reprovados")
    args = p.parse_args()

    arquivos = listar_arquivos()
    if not arquivos:
        raise SystemExit("Acervo vazio. Rode antes: python src/make_samples.py")

    cabecalho = (f"{'documento':<40}{'car.':>7}{'pal.':>6}{'alf':>6}{'dig':>6}"
                 f"{'sim':>6}{'MAI':>6}{'|pal|':>6}{'stop':>6}{'rep':>6}"
                 f"{'boil':>6}  decisão")
    print(cabecalho)
    print("-" * len(cabecalho))

    n_ok = n_falha = 0
    for caminho in arquivos:
        try:
            trechos = ler_documento(caminho)
        except ErroDeLeitura as e:
            print(f"{caminho.name:<40}  [erro de leitura] {e}")
            continue
        limpar_arquivo(trechos)

        for trecho in trechos:
            motivo, todos, m = avaliar(trecho.texto)
            if args.rejeitados and motivo is None:
                continue
            n_ok += motivo is None
            n_falha += motivo is not None

            rotulo = f"{caminho.name}·{trecho.localizador}"
            decisao = "aceito" if motivo is None else f"REJ: {motivo}"
            extras = f" (+{len(todos) - 1})" if len(todos) > 1 else ""
            print(f"{truncar(rotulo, 39):<40}{m['n_caracteres']:>7}{m['n_palavras']:>6}"
                  f"{m['prop_alfabetica']:>6.2f}{m['prop_digitos']:>6.2f}"
                  f"{m['prop_simbolos']:>6.2f}{m['prop_maiusculas']:>6.2f}"
                  f"{m['comprimento_palavra']:>6.1f}{m['prop_stopwords']:>6.2f}"
                  f"{m['prop_linhas_repetidas']:>6.2f}{m['prop_boilerplate']:>6.2f}"
                  f"  {decisao}{extras}")

    print(f"\n{n_ok} aceito(s), {n_falha} rejeitado(s).")
    print("\nLegenda: alf=prop. alfabética · dig=dígitos · sim=símbolos ·"
          " MAI=maiúsculas\n         |pal|=compr. médio da palavra ·"
          " stop=stopwords PT · rep=linhas repetidas · boil=boilerplate")
    print("\nMotivos possíveis: " + ", ".join(f"{k} ({v})" for k, v in MOTIVOS.items()
                                              if not k.endswith("duplicata")))
