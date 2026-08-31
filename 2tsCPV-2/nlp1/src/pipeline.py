"""
pipeline.py — orquestra as ETAPAS 1 a 5 e produz o corpus
-----------------------------------------------------------

Este é o script que você roda de verdade. Ele encadeia:

    1. aquisição      readers.ler_documento   -> trechos com procedência
    2. limpeza        clean.limpar_arquivo    -> texto normalizado
    3. filtragem      filters.avaliar         -> aceito / rejeitado + motivo
    4. deduplicação   dedup.deduplicar        -> únicos / duplicatas
    5. segmentação    segmentar_em_blocos     -> blocos prontos para uso

E grava CINCO artefatos em outputs/ — não um só:

    corpus.jsonl            o que passou
    rejeitados.jsonl        o que não passou, COM o motivo e as métricas
    duplicatas.jsonl        o que foi removido por repetição, com o original
    blocos.jsonl            o corpus segmentado
    relatorio.json          números agregados de todas as etapas
    metadados_corpus.json   a "ficha técnica" do corpus

Gravar os descartes é a decisão de projeto mais importante do arquivo. Um
pipeline que só produz o corpus final é uma caixa preta: quando alguém
perguntar "cadê o contrato da Souza Ltda.?", a única resposta possível será dar
de ombros. Com `rejeitados.jsonl`, a resposta é "foi descartado na filtragem
por `curto_demais`, tinha 31 caracteres, aqui está o texto".

Uso:
    python src/pipeline.py
    python src/pipeline.py --entrada /caminho/para/meus_documentos
    python src/pipeline.py --mascarar-pii
    python src/pipeline.py --sem-dedup --limite 5
"""

import argparse
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from clean import limpar_arquivo
from config import CFG, MOTIVOS, OUT_DIR, BRUTOS_DIR
from dedup import deduplicar
from filters import avaliar
from readers import ErroDeLeitura, ler_documento, listar_arquivos
from utils import configurar_terminal, salvar_json, salvar_jsonl


# ---------------------------------------------------------------------------
# ETAPA 5 — segmentação
# ---------------------------------------------------------------------------
def segmentar_em_blocos(texto: str, cfg=CFG) -> list[str]:
    """Corta o texto em blocos com sobreposição, respeitando os parágrafos.

    Por que segmentar: tudo que consome corpus tem limite de tamanho — a janela
    de contexto de um modelo, o campo de um índice vetorial, a entrada de um
    classificador. Cortar por número fixo de caracteres é fácil e ruim: parte
    frases no meio e produz blocos que começam em "…ção do lote, o que".

    Duas decisões que fazem a diferença:

      * Cortar na FRONTEIRA DE PARÁGRAFO sempre que possível. O parágrafo já é
        a unidade de sentido escolhida por quem escreveu — aproveite.
      * SOBREPOR os blocos (`sobreposicao_bloco`). Se a resposta a uma pergunta
        estiver bem na emenda entre dois blocos, sem sobreposição ela fica
        partida e nenhum dos dois blocos responde. O preço é redundância: ~15%
        de texto repetido, com o parâmetro padrão.

    Parágrafo maior que o bloco é cortado à força, no limite de caracteres —
    caso raro, mas precisa ser tratado, senão um documento sem quebras de linha
    produziria um bloco gigante que estoura o limite lá na frente.
    """
    if len(texto) <= cfg.tamanho_bloco:
        return [texto]

    blocos: list[str] = []
    atual = ""
    for paragrafo in texto.split("\n"):
        while len(paragrafo) > cfg.tamanho_bloco:
            if atual:
                blocos.append(atual.strip())
                atual = ""
            blocos.append(paragrafo[:cfg.tamanho_bloco].strip())
            paragrafo = paragrafo[cfg.tamanho_bloco:]

        if len(atual) + len(paragrafo) + 1 <= cfg.tamanho_bloco:
            atual = f"{atual}\n{paragrafo}" if atual else paragrafo
        else:
            blocos.append(atual.strip())
            # A cauda do bloco anterior abre o próximo: é a sobreposição.
            cauda = atual[-cfg.sobreposicao_bloco:] if cfg.sobreposicao_bloco else ""
            atual = f"{cauda}\n{paragrafo}".strip() if cauda else paragrafo

    if atual.strip():
        blocos.append(atual.strip())
    return [b for b in blocos if b]


# ---------------------------------------------------------------------------
# ETAPAS 1 a 3, arquivo por arquivo
# ---------------------------------------------------------------------------
def processar_arquivo(caminho: Path, cfg=CFG, mascarar: bool | None = None) -> dict:
    """Lê, limpa e filtra UM arquivo. Nunca levanta exceção.

    Devolve um dicionário com os documentos e o diagnóstico do arquivo. O
    contrato de "nunca levanta exceção" é o que permite processar um acervo
    real: um PDF corrompido no meio de 4.000 arquivos não pode custar as três
    horas de processamento que já rodaram.
    """
    diagnostico = {"arquivo": caminho.name, "formato": caminho.suffix.lstrip("."),
                   "erro": None, "n_trechos": 0, "n_aceitos": 0,
                   "n_rejeitados": 0, "linhas_moldura": 0}

    try:
        trechos = ler_documento(caminho, cfg)
    except ErroDeLeitura as e:
        diagnostico["erro"] = str(e)
        return {"aceitos": [], "rejeitados": [], "diagnostico": diagnostico}

    trechos, repetidas = limpar_arquivo(trechos, cfg, mascarar)
    diagnostico["n_trechos"] = len(trechos)
    diagnostico["linhas_moldura"] = len(repetidas)

    aceitos, rejeitados = [], []
    for trecho in trechos:
        motivo, motivos, metricas = avaliar(trecho.texto, cfg)
        registro = trecho.to_dict()
        registro["metricas"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in metricas.items()}
        if motivo is None:
            registro["status"] = "aceito"
            aceitos.append(registro)
        else:
            registro["status"] = "rejeitado"
            registro["motivo"] = motivo
            registro["motivos"] = motivos
            rejeitados.append(registro)

    diagnostico["n_aceitos"] = len(aceitos)
    diagnostico["n_rejeitados"] = len(rejeitados)
    return {"aceitos": aceitos, "rejeitados": rejeitados, "diagnostico": diagnostico}


# ---------------------------------------------------------------------------
# O pipeline completo
# ---------------------------------------------------------------------------
def executar(entrada: Path = BRUTOS_DIR, cfg=CFG, mascarar: bool | None = None,
             sem_dedup: bool = False, limite: int = 0) -> dict:
    t0 = time.time()

    arquivos = listar_arquivos(entrada, cfg)
    if limite:
        arquivos = arquivos[:limite]
    if not arquivos:
        raise SystemExit(
            f"Nenhum arquivo suportado em {entrada}.\n"
            f"Extensões aceitas: {', '.join(cfg.extensoes)}\n"
            f"Para gerar o acervo de exemplo: python src/make_samples.py")

    print(f"Entrada : {entrada}")
    print(f"Arquivos: {len(arquivos)}\n")

    # --- ETAPAS 1 a 3 -----------------------------------------------------
    aceitos, rejeitados, diagnosticos = [], [], []
    print(f"{'arquivo':<34}{'trechos':>8}{'aceitos':>9}{'rejeit.':>9}  observação")
    print("-" * 78)
    for caminho in arquivos:
        resultado = processar_arquivo(caminho, cfg, mascarar)
        aceitos += resultado["aceitos"]
        rejeitados += resultado["rejeitados"]
        d = resultado["diagnostico"]
        diagnosticos.append(d)

        if d["erro"]:
            print(f"{caminho.name:<34}{'—':>8}{'—':>9}{'—':>9}  [erro] {d['erro']}")
            continue
        nota = ""
        if d["linhas_moldura"]:
            nota = f"{d['linhas_moldura']} linha(s) de cabeçalho/rodapé removida(s)"
        if d["n_trechos"] == 0:
            nota = "nada extraído"
        print(f"{caminho.name:<34}{d['n_trechos']:>8}{d['n_aceitos']:>9}"
              f"{d['n_rejeitados']:>9}  {nota}")

    # --- ETAPA 4 ----------------------------------------------------------
    if sem_dedup:
        unicos, duplicatas = aceitos, []
    else:
        unicos, duplicatas = deduplicar(aceitos, cfg)

    # --- ETAPA 5 ----------------------------------------------------------
    blocos = []
    for documento in unicos:
        for i, texto in enumerate(segmentar_em_blocos(documento["texto"], cfg)):
            blocos.append({
                "id": f"{documento['id']}#{i}",
                "id_documento": documento["id"],
                "arquivo": documento["arquivo"],
                "formato": documento["formato"],
                "localizador": documento["localizador"],
                "bloco": i,
                "texto": texto,
                "n_caracteres": len(texto),
            })

    # --- Gravação ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    salvar_jsonl(unicos, OUT_DIR / cfg.arq_corpus)
    salvar_jsonl(rejeitados, OUT_DIR / cfg.arq_rejeitados)
    salvar_jsonl(duplicatas, OUT_DIR / cfg.arq_duplicatas)
    salvar_jsonl(blocos, OUT_DIR / cfg.arq_blocos)

    # --- Números agregados ------------------------------------------------
    motivos = Counter(r["motivo"] for r in rejeitados)
    motivos.update(d["motivo"] for d in duplicatas)
    por_formato = Counter(d["formato"] for d in unicos)
    palavras_por_formato = defaultdict(int)
    for d in unicos:
        palavras_por_formato[d["formato"]] += d["n_palavras"]

    n_entrada = len(aceitos) + len(rejeitados)
    relatorio = {
        "entrada": str(entrada),
        "n_arquivos": len(arquivos),
        "n_arquivos_com_erro": sum(1 for d in diagnosticos if d["erro"]),
        "n_trechos_extraidos": n_entrada,
        "n_aceitos_na_filtragem": len(aceitos),
        "n_rejeitados": len(rejeitados),
        "n_duplicatas": len(duplicatas),
        "n_documentos_finais": len(unicos),
        "n_blocos": len(blocos),
        "n_palavras": sum(d["n_palavras"] for d in unicos),
        "n_caracteres": sum(d["n_caracteres"] for d in unicos),
        "taxa_aproveitamento": round(len(unicos) / n_entrada, 4) if n_entrada else 0.0,
        "motivos": dict(motivos.most_common()),
        "documentos_por_formato": dict(por_formato.most_common()),
        "palavras_por_formato": dict(palavras_por_formato),
        "diagnostico_por_arquivo": diagnosticos,
        "config": cfg.to_dict(),
        "segundos": round(time.time() - t0, 2),
    }
    salvar_json(relatorio, OUT_DIR / cfg.arq_relatorio)

    # Ficha técnica do corpus (datasheet). Prática recomendada por Gebru et al.
    # (2018): todo conjunto de dados deve vir acompanhado de um documento que
    # diga de onde ele veio, o que foi removido e com que critério. Sem isso,
    # daqui a seis meses ninguém consegue responder por que o corpus é assim.
    salvar_json({
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origem": str(entrada),
        "n_documentos": len(unicos),
        "n_palavras": relatorio["n_palavras"],
        "formatos": dict(por_formato),
        "idioma_alvo": "pt-BR",
        "pii_mascarada": cfg.mascarar_pii if mascarar is None else mascarar,
        "criterios_de_exclusao": {k: MOTIVOS.get(k, k) for k in motivos},
        "limiares": {k: v for k, v in cfg.to_dict().items()
                     if k.startswith(("min_", "max_", "limiar_"))},
        "arquivos_de_origem": [d["arquivo"] for d in diagnosticos],
    }, OUT_DIR / cfg.arq_metadados)

    # --- Resumo no terminal -----------------------------------------------
    print("\n" + "=" * 78)
    print("RESUMO DO PIPELINE")
    print("=" * 78)
    print(f"  1-2. Extraídos e limpos     {n_entrada:>6} trechos "
          f"de {len(arquivos)} arquivos")
    print(f"  3.   Rejeitados na filtragem{len(rejeitados):>6}")
    print(f"  4.   Removidos por duplicata{len(duplicatas):>6}")
    # Separador de milhar no padrão brasileiro: formata com vírgula e troca.
    # A troca é feita SÓ no número — aplicá-la na frase inteira comeria também
    # a vírgula da pontuação.
    n_palavras_br = f"{relatorio['n_palavras']:,}".replace(",", ".")
    print(f"  5.   Corpus final           {len(unicos):>6} documentos, "
          f"{n_palavras_br} palavras")
    print(f"       Segmentado em          {len(blocos):>6} blocos")
    print(f"\n  Aproveitamento: {relatorio['taxa_aproveitamento']:.1%} "
          f"dos trechos extraídos   ({relatorio['segundos']}s)")

    if motivos:
        print("\n  Motivos de descarte:")
        for motivo, n in motivos.most_common():
            print(f"    {n:>4}  {motivo:<18} {MOTIVOS.get(motivo, '')}")

    erros = [d for d in diagnosticos if d["erro"]]
    if erros:
        print(f"\n  [!] {len(erros)} arquivo(s) não puderam ser lidos:")
        for d in erros:
            print(f"      {d['arquivo']}: {d['erro']}")

    # O aviso mais útil do relatório: extração vazia não é documento ruim, é
    # ETAPA 1 falhando. Merece destaque próprio, não uma linha na tabela.
    n_vazios = sum(1 for r in rejeitados if r["motivo"] == "vazio")
    if n_vazios:
        print(f"\n  [!] {n_vazios} trecho(s) sem texto extraível. Provável PDF")
        print("      escaneado: o arquivo tem imagem da página, não texto.")
        print("      A solução é OCR (ocrmypdf/tesseract), não ajustar filtro.")

    print(f"\n  Artefatos em {OUT_DIR}:")
    for arq in (cfg.arq_corpus, cfg.arq_rejeitados, cfg.arq_duplicatas,
                cfg.arq_blocos, cfg.arq_relatorio, cfg.arq_metadados):
        caminho = OUT_DIR / arq
        if caminho.exists():
            print(f"    {arq:<24} {caminho.stat().st_size / 1024:>7.1f} KB")

    print("\n  Próximo passo:  python src/report.py")
    return relatorio


if __name__ == "__main__":
    configurar_terminal()
    p = argparse.ArgumentParser(description="Pipeline de aquisição e filtragem de textos")
    p.add_argument("--entrada", type=Path, default=BRUTOS_DIR,
                   help="pasta com os documentos (padrão: data/brutos)")
    p.add_argument("--mascarar-pii", action="store_true",
                   help="substitui e-mail, CPF, CNPJ e telefone por marcadores")
    p.add_argument("--sem-dedup", action="store_true",
                   help="pula a ETAPA 4 (para comparar o antes e o depois)")
    p.add_argument("--limite", type=int, default=0,
                   help="processa apenas os N primeiros arquivos")
    args = p.parse_args()

    executar(entrada=args.entrada, mascarar=args.mascarar_pii or None,
             sem_dedup=args.sem_dedup, limite=args.limite)
