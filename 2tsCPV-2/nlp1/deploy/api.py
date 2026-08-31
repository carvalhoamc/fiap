"""
api.py — ETAPA 7: servir o pipeline como uma API web
------------------------------------------------------

O pipeline em lote (`pipeline.py`) processa uma pasta. Este serviço faz o
mesmo trabalho para UM arquivo enviado por HTTP, e é o que permite plugar a
aquisição de textos em qualquer outro sistema: um formulário de upload, um
robô que monitora uma caixa de e-mail, uma esteira que recebe documentos de um
portal.

    POST /extrair   arquivo (pdf/txt/docx/xlsx/csv) -> trechos limpos + decisão
    POST /avaliar   texto puro -> métricas de qualidade + decisão
    GET  /saude     health check

UMA DECISÃO DE ENGENHARIA QUE CONTRARIA A AULA DE CNN — e vale entender o porquê

Na aula de redes convolucionais, o serviço NÃO importava nada de `src/`: ele
dependia só de dois artefatos (`modelo_scriptado.pt` e `classes.json`), e o
pré-processamento era reimplementado no servidor. Aqui fazemos o OPOSTO: este
arquivo importa `readers`, `clean` e `filters` diretamente.

Isso não é incoerência, é a mesma regra aplicada a um caso diferente. A regra
verdadeira nunca foi "não importe do src". A regra é:

    NUNCA DEIXE DUAS CÓPIAS DA MESMA DECISÃO EXISTIREM SEPARADAS.

Na CNN, o "artefato" que atravessa a fronteira é o modelo serializado: ele
carrega os pesos E a arquitetura, então o servidor não precisa do código. Já em
um pipeline de texto, não existe artefato equivalente — as regras de limpeza e
os limiares de filtragem SÃO o produto. Se reimplementássemos `limpar()` aqui,
teríamos duas versões da hifenização, elas divergiriam na primeira correção de
bug, e o texto servido pela API deixaria de ser igual ao texto do corpus, sem
nenhum erro aparecer. É exatamente o mesmo desastre da divergência de
pré-processamento da aula de CNN — só que pela via oposta.

O contrato entre lote e serviço é o `config.py`: os dois leem os mesmos
limiares. O endpoint `/saude` devolve esses limiares justamente para que você
possa conferir, em produção, com que configuração o serviço está rodando.

Como rodar (a partir da pasta nlp1/):
    pip install fastapi "uvicorn[standard]" python-multipart
    python -m uvicorn deploy.api:app --reload --port 8000

Depois abra http://127.0.0.1:8000        (formulário de teste)
             http://127.0.0.1:8000/docs   (documentação interativa Swagger)
"""

import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from clean import limpar, limpar_arquivo          # noqa: E402
from config import CFG, MOTIVOS                   # noqa: E402
from filters import avaliar                       # noqa: E402
from readers import ErroDeLeitura, LEITORES, ler_documento   # noqa: E402

# 25 MB. Limite explícito porque um upload sem teto é uma negação de serviço
# esperando acontecer: basta alguém mandar um PDF de 2 GB.
TAMANHO_MAXIMO = 25 * 1024 * 1024


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """O que roda na subida e na descida do servidor.

    Aqui não há modelo pesado para carregar, mas há uma verificação que vale
    ouro: conferir, no startup, que as bibliotecas de leitura estão instaladas.
    Descobrir que falta `pypdf` quando o primeiro PDF chega, às três da manhã,
    é bem pior do que descobrir na subida.
    """
    faltando = []
    for modulo, formato in (("pypdf", "pdf"), ("docx", "docx"), ("openpyxl", "xlsx")):
        try:
            __import__(modulo)
        except ImportError:
            faltando.append(f"{formato} ({modulo})")
    if faltando:
        print(f"[startup] AVISO: formatos indisponíveis: {', '.join(faltando)}")
    print(f"[startup] formatos suportados: {', '.join(sorted(LEITORES))}")
    print(f"[startup] limiares: min_palavras={CFG.min_palavras}, "
          f"min_caracteres={CFG.min_caracteres}")
    yield
    print("[shutdown] encerrando serviço")


app = FastAPI(
    title="Aquisição e Filtragem de Textos",
    description="Extrai, limpa e filtra texto de PDF, TXT, DOCX e planilhas.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


class Texto(BaseModel):
    texto: str
    mascarar_pii: bool = False


@app.get("/saude")
def saude():
    """Health check — todo serviço em produção precisa de um.

    É este endpoint que o orquestrador (Docker, Kubernetes, balanceador)
    consulta para decidir se a instância pode receber tráfego. Devolvemos
    junto os limiares em uso: em produção, "com que configuração este serviço
    está rodando?" é a primeira pergunta de qualquer investigação.
    """
    return {
        "status": "ok",
        "formatos": sorted(LEITORES),
        "limiares": {k: v for k, v in CFG.to_dict().items()
                     if k.startswith(("min_", "max_", "limiar_"))},
    }


@app.post("/extrair")
async def extrair(arquivo: UploadFile = File(...), mascarar_pii: bool = False):
    """Recebe um arquivo, devolve os trechos limpos com a decisão de cada um.

    Repare que a resposta inclui os trechos REJEITADOS, com o motivo — e não
    só os aprovados. É a mesma regra do `rejeitados.jsonl` do modo em lote:
    quem chama o serviço precisa poder saber por que um pedaço do documento
    não veio, sob pena de achar que o arquivo estava vazio.
    """
    sufixo = Path(arquivo.filename or "").suffix.lower()
    if sufixo not in LEITORES:
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado: {sufixo or '(sem extensão)'}. "
                   f"Aceitos: {', '.join(sorted(LEITORES))}")

    conteudo = await arquivo.read()
    if len(conteudo) > TAMANHO_MAXIMO:
        raise HTTPException(status_code=413, detail="Arquivo acima de 25 MB.")
    if not conteudo:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    t0 = time.perf_counter()
    # Os leitores trabalham com caminho de arquivo (o pypdf e o openpyxl
    # esperam um arquivo em disco), então gravamos o upload em um temporário.
    # delete=False + finally é o padrão que funciona no Windows, onde não se
    # pode reabrir um arquivo temporário ainda aberto.
    temporario = tempfile.NamedTemporaryFile(suffix=sufixo, delete=False)
    try:
        temporario.write(conteudo)
        temporario.close()
        try:
            trechos = ler_documento(Path(temporario.name), CFG)
        except ErroDeLeitura as e:
            raise HTTPException(status_code=422, detail=f"Falha na leitura: {e}")

        trechos, repetidas = limpar_arquivo(trechos, CFG, mascarar_pii)
    finally:
        Path(temporario.name).unlink(missing_ok=True)

    resultado = []
    for trecho in trechos:
        motivo, motivos, metricas = avaliar(trecho.texto, CFG)
        resultado.append({
            "localizador": trecho.localizador,
            "status": "aceito" if motivo is None else "rejeitado",
            "motivo": motivo,
            "explicacao": MOTIVOS.get(motivo) if motivo else None,
            "motivos_secundarios": motivos[1:],
            "n_caracteres": len(trecho.texto),
            "n_palavras": metricas["n_palavras"],
            "metricas": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in metricas.items()},
            "texto": trecho.texto,
        })

    aceitos = [t for t in resultado if t["status"] == "aceito"]
    return {
        "arquivo": arquivo.filename,
        "formato": sufixo.lstrip("."),
        "codificacao": trechos[0].codificacao if trechos else None,
        "n_trechos": len(resultado),
        "n_aceitos": len(aceitos),
        "n_rejeitados": len(resultado) - len(aceitos),
        "linhas_moldura_removidas": sorted(repetidas),
        "n_palavras_aceitas": sum(t["n_palavras"] for t in aceitos),
        "tempo_ms": round((time.perf_counter() - t0) * 1000, 2),
        "trechos": resultado,
    }


@app.post("/avaliar")
def avaliar_texto(entrada: Texto):
    """Limpa e avalia um texto enviado direto, sem arquivo.

    Útil para duas coisas: testar limiares rapidamente (cole um texto e veja
    as métricas) e integrar a filtragem em uma esteira onde o texto já foi
    extraído por outro sistema.
    """
    if not entrada.texto.strip():
        raise HTTPException(status_code=400, detail="Texto vazio.")

    limpo = limpar(entrada.texto, CFG, entrada.mascarar_pii)
    motivo, motivos, metricas = avaliar(limpo, CFG)
    return {
        "status": "aceito" if motivo is None else "rejeitado",
        "motivo": motivo,
        "explicacao": MOTIVOS.get(motivo) if motivo else None,
        "motivos_secundarios": motivos[1:],
        "metricas": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in metricas.items()},
        "texto_limpo": limpo,
    }


@app.get("/", response_class=HTMLResponse)
def pagina_teste():
    """Página mínima para demonstração em sala, sem precisar de Postman/curl."""
    return """
<!doctype html><html lang="pt-br"><meta charset="utf-8">
<title>Aquisição e Filtragem de Textos</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:3rem auto;padding:0 1rem;line-height:1.5}
 h1{margin-bottom:.2rem} p.sub{color:#666;margin-top:0}
 fieldset{border:1px solid #ddd;border-radius:8px;margin-bottom:1.5rem;padding:1rem}
 legend{font-weight:600;padding:0 .4rem}
 textarea{width:100%;min-height:110px;font-family:ui-monospace,monospace;font-size:.85rem}
 #saida{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:8px;
        font-family:ui-monospace,monospace;font-size:.8rem;max-height:460px;overflow:auto}
 button{padding:.6rem 1.2rem;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}
 label{margin-left:.8rem;font-size:.9rem}
</style>
<h1>Aquisição e Filtragem de Textos</h1>
<p class="sub">PDF · TXT · DOCX · XLSX · CSV &nbsp;—&nbsp; extrai, limpa, mede e decide.</p>

<fieldset><legend>Enviar um arquivo</legend>
  <input type="file" id="arq" accept=".pdf,.txt,.md,.docx,.xlsx,.xlsm,.csv">
  <label><input type="checkbox" id="pii"> mascarar dados pessoais</label>
  <p><button onclick="enviarArquivo()">Extrair e filtrar</button></p>
</fieldset>

<fieldset><legend>Ou colar um texto</legend>
  <textarea id="txt" placeholder="Cole aqui um texto para ver as métricas de qualidade…"></textarea>
  <p><button onclick="enviarTexto()">Avaliar texto</button></p>
</fieldset>

<div id="saida">aguardando…</div>
<script>
const saida = document.getElementById('saida');
async function enviarArquivo(){
  const f = document.getElementById('arq').files[0];
  if(!f){ saida.textContent = 'Selecione um arquivo.'; return; }
  const fd = new FormData(); fd.append('arquivo', f);
  const pii = document.getElementById('pii').checked;
  saida.textContent = 'processando…';
  const r = await fetch('/extrair?mascarar_pii=' + pii, {method:'POST', body: fd});
  saida.textContent = JSON.stringify(await r.json(), null, 2);
}
async function enviarTexto(){
  const t = document.getElementById('txt').value;
  if(!t.trim()){ saida.textContent = 'Cole um texto.'; return; }
  saida.textContent = 'processando…';
  const r = await fetch('/avaliar', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({texto: t, mascarar_pii: document.getElementById('pii').checked})});
  saida.textContent = JSON.stringify(await r.json(), null, 2);
}
</script></html>
"""
