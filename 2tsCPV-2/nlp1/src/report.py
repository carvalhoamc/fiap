"""
report.py — ETAPA 6 do pipeline: relatório e auditoria do corpus
------------------------------------------------------------------

O `pipeline.py` já imprime os totais. Este script existe para a pergunta
seguinte, que é a que importa:

    **O que eu joguei fora, e eu deveria ter jogado?**

Nenhum número agregado responde isso. Só a leitura dos documentos rejeitados
responde — e é por isso que a saída principal deste script não é um gráfico, e
sim uma AMOSTRA DE TEXTO dos descartes, com o motivo ao lado.

A regra prática da disciplina: leia dez rejeitados de cada motivo, na mão,
antes de considerar um pipeline pronto. Se você concordar com os dez, o limiar
está bom. Se discordar de dois, o limiar está errado — e você acabou de
descobrir isso pelo preço de cinco minutos de leitura, em vez de descobrir
depois de treinar um modelo por três dias.

Gera também três gráficos, que respondem a perguntas diferentes:

    motivos_rejeicao.png       o que mais derruba documento?
    distribuicao_tamanhos.png  meu limiar de tamanho está na cauda ou no meio?
    documentos_por_formato.png de onde veio o corpus, afinal?

Uso:
    python src/report.py
    python src/report.py --amostras 5
    python src/report.py --motivo curto_demais
"""

import argparse
from collections import Counter, defaultdict

from config import CFG, MOTIVOS, OUT_DIR
from utils import (carregar_json, configurar_terminal, ler_jsonl, plotar_barras,
                   plotar_histograma, plotar_pizza, truncar)


def _linha(titulo: str, largura: int = 78) -> None:
    print("\n" + "=" * largura)
    print(titulo)
    print("=" * largura)


def relatar(n_amostras: int = 3, motivo_alvo: str | None = None) -> None:
    caminho_relatorio = OUT_DIR / CFG.arq_relatorio
    if not caminho_relatorio.exists():
        raise SystemExit("Relatório não encontrado. Rode antes: python src/pipeline.py")

    relatorio = carregar_json(caminho_relatorio)
    corpus = list(ler_jsonl(OUT_DIR / CFG.arq_corpus))
    rejeitados = list(ler_jsonl(OUT_DIR / CFG.arq_rejeitados))
    duplicatas = list(ler_jsonl(OUT_DIR / CFG.arq_duplicatas))

    # ---------------------------------------------------------------------
    # 1. O funil
    # ---------------------------------------------------------------------
    _linha("O FUNIL — quanto sobrou de cada etapa")
    extraidos = relatorio["n_trechos_extraidos"]
    etapas = [
        ("1-2. Trechos extraídos e limpos", extraidos),
        ("3.   Sobreviveram à filtragem", relatorio["n_aceitos_na_filtragem"]),
        ("4.   Sobreviveram à deduplicação", relatorio["n_documentos_finais"]),
    ]
    for nome, n in etapas:
        proporcao = n / extraidos if extraidos else 0
        barra = "#" * int(proporcao * 40)
        print(f"  {nome:<34}{n:>5}  {barra} {proporcao:.0%}")
    print(f"\n  Corpus final: {relatorio['n_documentos_finais']} documentos · "
          f"{relatorio['n_palavras']:,} palavras · "
          f"{relatorio['n_blocos']} blocos".replace(",", "."))

    # Um aproveitamento muito alto merece tanta desconfiança quanto um muito
    # baixo: filtro que nunca reprova nada normalmente está desligado por um
    # limiar frouxo demais, não porque o acervo é impecável.
    taxa = relatorio["taxa_aproveitamento"]
    if taxa > 0.95:
        print("\n  [?] Aproveitamento acima de 95%. Ou o acervo é limpo, ou os")
        print("      limiares estão frouxos. Confira uma amostra do que passou.")
    elif taxa < 0.3:
        print("\n  [?] Aproveitamento abaixo de 30%. Antes de aceitar isso, leia")
        print("      os rejeitados: filtro agressivo demais custa dado bom.")

    # ---------------------------------------------------------------------
    # 2. Composição do corpus
    # ---------------------------------------------------------------------
    _linha("COMPOSIÇÃO DO CORPUS")
    palavras_por_formato = relatorio["palavras_por_formato"]
    print(f"{'formato':<12}{'documentos':>12}{'palavras':>12}{'% palavras':>12}")
    print("-" * 48)
    total_palavras = max(1, relatorio["n_palavras"])
    for formato, n in relatorio["documentos_por_formato"].items():
        p = palavras_por_formato.get(formato, 0)
        print(f"{formato:<12}{n:>12}{p:>12,}{p / total_palavras:>11.1%}"
              .replace(",", "."))

    # Quem contribui com o corpus: útil para detectar que 80% do texto veio de
    # um único arquivo — o que torna o corpus enviesado para aquele assunto.
    por_arquivo = defaultdict(int)
    for d in corpus:
        por_arquivo[d["arquivo"]] += d["n_palavras"]
    if por_arquivo:
        print(f"\n{'arquivos que mais contribuíram':<52}{'palavras':>10}{'%':>8}")
        print("-" * 70)
        for arquivo, n in sorted(por_arquivo.items(), key=lambda kv: -kv[1])[:8]:
            print(f"{truncar(arquivo, 51):<52}{n:>10}{n / total_palavras:>8.1%}")

    # ---------------------------------------------------------------------
    # 3. Descartes
    # ---------------------------------------------------------------------
    _linha("DESCARTES — o que foi removido e por quê")
    motivos = Counter(relatorio["motivos"])
    if not motivos:
        print("  Nenhum documento descartado.")
    else:
        print(f"{'motivo':<20}{'n':>6}{'% do extraído':>16}   descrição")
        print("-" * 78)
        for motivo, n in motivos.most_common():
            print(f"{motivo:<20}{n:>6}{n / extraidos:>15.1%}   "
                  f"{MOTIVOS.get(motivo, '')}")

    # ---------------------------------------------------------------------
    # 4. A parte que importa: LER os descartes
    # ---------------------------------------------------------------------
    _linha(f"AUDITORIA — amostra do que foi descartado "
           f"({n_amostras} por motivo)")
    print("Leia. É aqui que você descobre que o limiar está errado.\n")

    por_motivo = defaultdict(list)
    for r in rejeitados:
        por_motivo[r["motivo"]].append(r)
    for d in duplicatas:
        por_motivo[d["motivo"]].append(d)

    alvos = [motivo_alvo] if motivo_alvo else sorted(por_motivo)
    for motivo in alvos:
        itens = por_motivo.get(motivo, [])
        if not itens:
            print(f"-- {motivo}: nenhum documento.\n")
            continue
        print(f"-- {motivo} ({len(itens)} documento(s)) — {MOTIVOS.get(motivo, '')}")
        for item in itens[:n_amostras]:
            m = item.get("metricas", {})
            print(f"   {item['arquivo']} · {item['localizador']}")
            if motivo in ("duplicata_exata", "quase_duplicata"):
                print(f"   cópia de {item['duplicata_de']} "
                      f"(similaridade {item['similaridade']:.3f})")
            else:
                print(f"   {m.get('n_caracteres', 0)} car., "
                      f"{m.get('n_palavras', 0)} pal., "
                      f"alf={m.get('prop_alfabetica', 0):.2f} "
                      f"sim={m.get('prop_simbolos', 0):.2f} "
                      f"stop={m.get('prop_stopwords', 0):.2f} "
                      f"boil={m.get('prop_boilerplate', 0):.2f}")
                if len(item.get("motivos", [])) > 1:
                    print(f"   também reprovou em: "
                          f"{', '.join(item['motivos'][1:])}")
            texto = item["texto"].strip()
            print(f"   > {truncar(texto, 150) if texto else '[VAZIO]'}\n")

    # ---------------------------------------------------------------------
    # 5. Gráficos
    # ---------------------------------------------------------------------
    _linha("GRÁFICOS")
    if motivos:
        plotar_barras({f"{MOTIVOS.get(k, k)}": v for k, v in motivos.items()},
                      "Motivos de descarte", "documentos descartados",
                      OUT_DIR / CFG.arq_grafico_motivos, cor="#dc2626")

    # O histograma junta aceitos e rejeitados de propósito: o que interessa é
    # ver ONDE o limiar caiu em relação à distribuição inteira.
    tamanhos = ([d["n_palavras"] for d in corpus]
                + [r["metricas"]["n_palavras"] for r in rejeitados
                   if "metricas" in r])
    plotar_histograma(tamanhos, "Distribuição de tamanho dos trechos extraídos",
                      "palavras por trecho",
                      OUT_DIR / CFG.arq_grafico_tamanhos,
                      limiar=CFG.min_palavras)

    if relatorio["documentos_por_formato"]:
        plotar_pizza(relatorio["documentos_por_formato"],
                     "Documentos do corpus por formato de origem",
                     OUT_DIR / CFG.arq_grafico_formatos)

    print(f"\nArtefatos completos em {OUT_DIR}")
    print("Para inspecionar um documento específico:")
    print(f"  python -c \"from utils import ler_jsonl;"
          f"[print(d['texto'][:500]) for d in list(ler_jsonl('outputs/corpus.jsonl'))[:1]]\"")


if __name__ == "__main__":
    configurar_terminal()
    p = argparse.ArgumentParser(description="Relatório e auditoria do corpus")
    p.add_argument("--amostras", type=int, default=3,
                   help="quantos descartes mostrar por motivo")
    p.add_argument("--motivo", default=None,
                   help="auditar apenas um motivo (ex.: curto_demais)")
    args = p.parse_args()
    relatar(args.amostras, args.motivo)
