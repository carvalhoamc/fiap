"""
dedup.py — ETAPA 4 do pipeline: deduplicação
----------------------------------------------

Por que a duplicata é um problema sério, e não uma questão de arrumação:

  * Ela DESBALANCEIA o corpus em silêncio. Um documento presente 50 vezes tem
    50 vezes mais peso no treino. O modelo passa a "acreditar" mais naquele
    conteúdo por acidente de coleta, não por evidência.
  * Ela CONTAMINA a avaliação. Se a mesma página está no treino e no teste, a
    métrica mede memorização e você comemora um número que não existe. É o
    vazamento mais comum em projetos de PLN.
  * Ela AUMENTA a memorização literal. Trechos muito repetidos são os que os
    modelos reproduzem palavra por palavra — problema de privacidade e de
    direito autoral, não só de qualidade.
  * Ela CUSTA DINHEIRO. Em um sistema de recuperação, dez cópias da mesma
    página ocupam dez lugares entre os resultados e afogam a diversidade.

Dois níveis, com custos e alcances bem diferentes:

  EXATA        Hash do texto normalizado. Custo O(n), pega cópia literal.
               Não pega quase nada no mundo real, porque basta uma data
               diferente no rodapé para dois documentos idênticos terem hashes
               completamente distintos. Hash é tudo ou nada.

  QUASE-DUPLICATA  MinHash + LSH. Estima a similaridade de Jaccard entre os
               conjuntos de n-gramas de palavras, e encontra os pares parecidos
               sem comparar todos contra todos.

Uso:
    python src/dedup.py                    # duplicatas no acervo de exemplo
    python src/dedup.py --limiar 0.6
"""

import argparse
import random
from collections import defaultdict

from config import CFG
from utils import (configurar_terminal, hash_curto,
                   normalizar_para_comparacao, palavras)

# Primo de Mersenne 2^61 - 1: grande o bastante para as colisões serem
# desprezíveis e pequeno o bastante para a aritmética caber em inteiros
# rápidos. É a escolha padrão em implementações de MinHash.
PRIMO = (1 << 61) - 1
MAXIMO = PRIMO - 1


# ===========================================================================
# Nível 1 — duplicata exata
# ===========================================================================
def chave_exata(texto: str) -> str:
    """Hash do texto AGRESSIVAMENTE normalizado.

    Normalizamos antes de hashear (minúsculas, sem acento, sem pontuação, sem
    espaço redundante) para que duas cópias que diferem só por formatação
    caiam no mesmo balde. Sem isso, um único espaço a mais gera outro hash e a
    "deduplicação exata" não pega nem as cópias literais.

    Note que esta forma reduzida serve SÓ como chave de comparação. O texto
    guardado no corpus continua sendo o original limpo — jamais esta versão.
    """
    return hash_curto(normalizar_para_comparacao(texto), n=16)


# ===========================================================================
# Nível 2 — quase-duplicata (MinHash + LSH)
# ===========================================================================
def shingles(texto: str, n: int = CFG.tamanho_shingle) -> set[int]:
    """Conjunto de n-gramas de palavras, já convertidos em hash.

    Um "shingle" é uma janela deslizante de n palavras consecutivas. Para
    n = 5 e o texto "o gato subiu no telhado alto", os shingles são
    {"o gato subiu no telhado", "gato subiu no telhado alto"}.

    Por que n-gramas e não palavras soltas? Porque o conjunto de PALAVRAS de
    dois textos diferentes sobre o mesmo assunto é muito parecido — os dois
    falam de "inspeção", "peça", "lote". Já a sequência exata de 5 palavras só
    coincide quando um texto foi copiado do outro. O n-grama captura ordem, e
    é a ordem que distingue plágio de assunto em comum.

    n pequeno (2-3) -> sensível demais, aponta similaridade onde há só tema comum.
    n grande (8-10) -> rígido demais, qualquer edição quebra todos os shingles.
    n = 5 é o valor usual na literatura, e o padrão em config.py.
    """
    lista = palavras(texto)
    if len(lista) < n:
        # Texto curto demais para janelar: vira um shingle único. Evita o
        # conjunto vazio, que faria a similaridade ser 0 ou indefinida.
        return {hash(" ".join(lista))} if lista else set()
    return {hash(" ".join(lista[i:i + n])) for i in range(len(lista) - n + 1)}


def _coeficientes(n_permutacoes: int, semente: int) -> list[tuple[int, int]]:
    """Gera os pares (a, b) das funções de hash h(x) = (a*x + b) mod primo.

    Cada par é uma "permutação" aleatória do universo de shingles. A semente
    fixa é obrigatória: sem ela, duas execuções gerariam assinaturas
    incomparáveis, e um corpus processado em dois lotes teria deduplicação
    inútil entre os lotes.
    """
    rng = random.Random(semente)
    return [(rng.randrange(1, PRIMO), rng.randrange(0, PRIMO))
            for _ in range(n_permutacoes)]


def assinatura_minhash(conjunto: set[int], coeficientes: list[tuple[int, int]]) -> tuple:
    """Reduz um conjunto de milhares de shingles a k inteiros.

    A ideia, que é uma das mais elegantes da computação aplicada:

      Embaralhe o universo de todos os shingles possíveis com uma permutação
      aleatória e anote qual shingle do documento ficou em PRIMEIRO lugar.
      A probabilidade de dois documentos terem o mesmo primeiro colocado é
      EXATAMENTE a similaridade de Jaccard entre eles.

      Repita com k permutações independentes: a fração de posições em que as
      duas assinaturas coincidem estima a similaridade de Jaccard, com erro da
      ordem de 1/raiz(k).

    Com k = 64, o erro típico fica em torno de 12% — suficiente para separar
    "quase idêntico" (0,8) de "só fala do mesmo assunto" (0,2). Quer mais
    precisão? Aumente n_permutacoes e pague em tempo e memória.

    O ganho: um documento de 20.000 shingles vira 64 inteiros. Comparar dois
    documentos deixa de custar uma interseção de conjuntos gigantes e passa a
    custar 64 comparações de inteiro.
    """
    if not conjunto:
        return ()
    return tuple(
        min(((a * (h & MAXIMO) + b) % PRIMO) for h in conjunto)
        for a, b in coeficientes
    )


def jaccard_estimado(assinatura_a: tuple, assinatura_b: tuple) -> float:
    """Fração de posições coincidentes entre duas assinaturas."""
    if not assinatura_a or not assinatura_b:
        return 0.0
    iguais = sum(1 for x, y in zip(assinatura_a, assinatura_b) if x == y)
    return iguais / len(assinatura_a)


def jaccard_exato(a: set[int], b: set[int]) -> float:
    """|A ∩ B| / |A ∪ B| — a similaridade verdadeira, para conferir a estimativa."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _bandas(assinatura: tuple, n_bandas: int) -> list[tuple]:
    """Fatia a assinatura em bandas, para o LSH.

    LSH (Locality-Sensitive Hashing) resolve o problema que torna a
    deduplicação cara: comparar todos contra todos é O(n²). Com 100 mil
    documentos são 5 bilhões de comparações — inviável.

    O truque: quebre a assinatura de k posições em b bandas de r posições cada
    (k = b * r). Dois documentos viram CANDIDATOS se coincidirem inteiramente
    em pelo menos UMA banda. Como coincidir em r posições seguidas é
    improvável por acaso, quase só os pares realmente parecidos viram
    candidatos — e a comparação cara é feita só neles.

    A probabilidade de um par com similaridade s virar candidato é
    1 - (1 - s^r)^b: uma curva em S, cujo ponto de virada fica aproximadamente
    em (1/b)^(1/r). Com b = 16 e r = 4, a virada fica em torno de 0,4 — abaixo
    disso, quase nada passa; acima, quase tudo passa. Ajustar b e r é ajustar
    onde essa curva vira.
    """
    r = len(assinatura) // n_bandas
    return [assinatura[i * r:(i + 1) * r] for i in range(n_bandas)]


# ===========================================================================
# O deduplicador
# ===========================================================================
def deduplicar(documentos: list[dict], cfg=CFG) -> tuple[list[dict], list[dict]]:
    """Separa (únicos, duplicatas). Cada duplicata registra de quem é cópia.

    Estratégia em duas passadas, da mais barata para a mais cara:

      1. hash exato  -> elimina as cópias literais sem nenhuma matemática;
      2. MinHash+LSH -> pega o resto, comparando só os pares candidatos.

    Política de qual cópia sobrevive: **a primeira que aparece**. Como
    `readers.listar_arquivos` ordena os arquivos por nome, isso é determinístico
    e reproduzível — reprocessar o corpus mantém o mesmo sobrevivente. Em
    produção você provavelmente prefere outro critério (o mais longo, o mais
    recente, o de fonte mais confiável); o importante é que o critério seja
    EXPLÍCITO, e não um efeito colateral da ordem do sistema de arquivos.

    Cuidado com o efeito de transitividade: se A é parecido com B, e B com C,
    mas A não com C, o resultado depende da ordem de chegada. Deduplicação
    por vizinhança é intrinsecamente aproximada — não existe resposta única.
    """
    coeficientes = _coeficientes(cfg.n_permutacoes, cfg.semente)

    vistos_exatos: dict[str, str] = {}          # hash -> id do primeiro
    baldes: dict[tuple, list[int]] = defaultdict(list)   # banda -> índices
    guardados: list[tuple[str, set[int], tuple]] = []    # (id, shingles, assinatura)

    unicos, duplicatas = [], []

    for documento in documentos:
        texto = documento["texto"]

        # --- passada 1: exata --------------------------------------------
        chave = chave_exata(texto)
        if chave in vistos_exatos:
            duplicatas.append({**documento, "status": "duplicata",
                               "motivo": "duplicata_exata",
                               "duplicata_de": vistos_exatos[chave],
                               "similaridade": 1.0})
            continue

        # --- passada 2: quase-duplicata ----------------------------------
        conjunto = shingles(texto, cfg.tamanho_shingle)
        assinatura = assinatura_minhash(conjunto, coeficientes)

        candidatos = set()
        for banda in _bandas(assinatura, cfg.n_bandas):
            candidatos.update(baldes[banda])

        melhor_id, melhor_similaridade = None, 0.0
        for indice in candidatos:
            id_outro, shingles_outro, _ = guardados[indice]
            # Nos candidatos — que são poucos — vale pagar a similaridade
            # EXATA em vez da estimada: é mais precisa e o custo já está
            # contido pelo LSH.
            similaridade = jaccard_exato(conjunto, shingles_outro)
            if similaridade > melhor_similaridade:
                melhor_id, melhor_similaridade = id_outro, similaridade

        if melhor_similaridade >= cfg.limiar_jaccard:
            duplicatas.append({**documento, "status": "duplicata",
                               "motivo": "quase_duplicata",
                               "duplicata_de": melhor_id,
                               "similaridade": round(melhor_similaridade, 4)})
            continue

        # --- sobreviveu: entra no índice ---------------------------------
        vistos_exatos[chave] = documento["id"]
        indice = len(guardados)
        guardados.append((documento["id"], conjunto, assinatura))
        for banda in _bandas(assinatura, cfg.n_bandas):
            baldes[banda].append(indice)
        unicos.append(documento)

    return unicos, duplicatas


if __name__ == "__main__":
    configurar_terminal()
    # Rode:  python src/dedup.py
    # Mostra a diferença prática entre os dois níveis: a cópia literal cai no
    # hash exato; a versão editada só cai no MinHash.
    from clean import limpar_arquivo
    from readers import ErroDeLeitura, ler_documento, listar_arquivos
    from utils import truncar

    p = argparse.ArgumentParser(description="Deduplicação do acervo")
    p.add_argument("--limiar", type=float, default=CFG.limiar_jaccard)
    p.add_argument("--shingle", type=int, default=CFG.tamanho_shingle)
    args = p.parse_args()

    CFG.limiar_jaccard, CFG.tamanho_shingle = args.limiar, args.shingle

    documentos = []
    for caminho in listar_arquivos():
        try:
            trechos = ler_documento(caminho)
        except ErroDeLeitura:
            continue
        limpar_arquivo(trechos)
        documentos += [t.to_dict() for t in trechos if t.texto.strip()]

    unicos, duplicatas = deduplicar(documentos)

    print(f"Entraram {len(documentos)} documentos "
          f"(shingle={args.shingle}, limiar={args.limiar})")
    print(f"Únicos: {len(unicos)}   |   Duplicatas: {len(duplicatas)}\n")

    if duplicatas:
        indice = {d["id"]: d for d in documentos}
        for d in duplicatas:
            original = indice.get(d["duplicata_de"], {})
            print(f"[{d['motivo']}] similaridade {d['similaridade']:.3f}")
            print(f"   cópia   : {d['arquivo']} · {d['localizador']}")
            print(f"   original: {original.get('arquivo', '?')} · "
                  f"{original.get('localizador', '?')}")
            print(f"   trecho  : {truncar(d['texto'], 88)}\n")
    else:
        print("Nenhuma duplicata encontrada com este limiar.")

    # A demonstração mais útil da aula: a matriz de similaridade dos três
    # documentos que vieram da mesma notícia.
    familia = [d for d in documentos if d["arquivo"].endswith(
        ("noticia.txt", "noticia_copia.txt", "noticia_editada.txt"))]
    if len(familia) >= 2:
        print("Similaridade de Jaccard exata entre as versões da notícia:")
        conjuntos = {d["arquivo"].split("/")[-1]: shingles(d["texto"], args.shingle)
                     for d in familia}
        nomes = sorted(conjuntos)
        print(f"{'':<22}" + "".join(f"{n[:18]:>20}" for n in nomes))
        for a in nomes:
            linha = "".join(f"{jaccard_exato(conjuntos[a], conjuntos[b]):>20.3f}"
                            for b in nomes)
            print(f"{a:<22}{linha}")
