"""
api.py — ETAPA 7 (parte B): servir o modelo como uma API web
--------------------------------------------------------------

Colocar em produção significa transformar o modelo em um SERVIÇO que qualquer
aplicação (site, app, outro backend) consegue chamar por HTTP.

Repare em três decisões de engenharia importantes:

1. Este arquivo NÃO importa nada de src/. Ele depende apenas de dois artefatos
   produzidos pelo treino: `modelo_scriptado.pt` e `classes.json`. É assim que
   se separa o mundo da pesquisa do mundo da produção.

2. O modelo é carregado UMA VEZ, quando o servidor sobe — nunca a cada
   requisição. Carregar por requisição é o erro de desempenho mais comum em
   deploy de ML.

3. As constantes de pré-processamento estão duplicadas aqui de propósito, com
   um aviso: elas TÊM que ser idênticas às do treino. Em um projeto real você
   as salvaria em um arquivo de metadados junto ao modelo.

Como rodar (a partir da pasta cnn/):
    pip install fastapi "uvicorn[standard]" python-multipart pillow
    python -m uvicorn deploy.api:app --reload --port 8000

Depois abra http://127.0.0.1:8000  (formulário de teste)
             http://127.0.0.1:8000/docs  (documentação interativa Swagger)
"""

import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps

# --- constantes que PRECISAM bater com o treino ----------------------------
TAMANHO = 28
MEDIA = 0.2860
DESVIO = 0.3530

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_MODELO = RAIZ / "outputs" / "modelo_scriptado.pt"
CAMINHO_CLASSES = RAIZ / "outputs" / "classes.json"

modelo = None
classes: list[str] = []


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carrega o modelo uma única vez, na subida do servidor.

    O que vem antes do `yield` roda no startup; o que vem depois, no shutdown
    (fechar conexões, liberar recursos).
    """
    global modelo, classes
    if not CAMINHO_MODELO.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {CAMINHO_MODELO}.\n"
            "Rode antes:  python src/train.py  e  python src/export_model.py")
    modelo = torch.jit.load(str(CAMINHO_MODELO), map_location="cpu")
    modelo.eval()
    classes = json.loads(CAMINHO_CLASSES.read_text(encoding="utf-8"))
    print(f"[startup] modelo carregado | {len(classes)} classes")
    yield
    print("[shutdown] encerrando serviço")


app = FastAPI(
    title="Classificador de Roupas (CNN)",
    description="Serviço de inferência da CNN treinada no FashionMNIST",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


def preprocessar(bytes_imagem: bytes, inverter: bool = False) -> torch.Tensor:
    """bytes -> tensor (1,1,28,28). Mesma receita do treino, sem torchvision."""
    imagem = Image.open(io.BytesIO(bytes_imagem)).convert("L")
    if inverter:
        imagem = ImageOps.invert(imagem)
    imagem = imagem.resize((TAMANHO, TAMANHO))

    # bytearray (e não bytes) porque torch.frombuffer exige um buffer gravável
    bruto = torch.frombuffer(bytearray(imagem.tobytes()), dtype=torch.uint8)
    tensor = bruto.float().reshape(1, 1, TAMANHO, TAMANHO) / 255.0  # = ToTensor()
    return (tensor - MEDIA) / DESVIO                                # = Normalize()


@app.get("/saude")
def saude():
    """Health check — todo serviço em produção precisa de um.

    É este endpoint que o orquestrador (Docker, Kubernetes, load balancer)
    consulta para decidir se a instância está viva e pode receber tráfego.
    """
    return {"status": "ok", "modelo_carregado": modelo is not None,
            "n_classes": len(classes)}


@app.post("/prever")
async def prever(arquivo: UploadFile = File(...), inverter: bool = False):
    """Recebe uma imagem e devolve as 3 classes mais prováveis."""
    if not arquivo.content_type or not arquivo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem.")

    conteudo = await arquivo.read()
    try:
        tensor = preprocessar(conteudo, inverter)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {e}")

    t0 = time.perf_counter()
    with torch.no_grad():
        probabilidades = torch.softmax(modelo(tensor), dim=1)[0]
    ms = (time.perf_counter() - t0) * 1000

    valores, indices = probabilidades.topk(3)
    return {
        "arquivo": arquivo.filename,
        "predicao": classes[indices[0]],
        "confianca": round(valores[0].item(), 4),
        "top3": [{"classe": classes[i], "probabilidade": round(v.item(), 4)}
                 for v, i in zip(valores, indices)],
        "tempo_inferencia_ms": round(ms, 2),
    }


@app.get("/", response_class=HTMLResponse)
def pagina_teste():
    """Página mínima para demonstração em sala, sem precisar de Postman/curl."""
    return """
<!doctype html><html lang="pt-br"><meta charset="utf-8">
<title>Classificador de Roupas</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:3rem auto;padding:0 1rem}
 #saida{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:8px;margin-top:1rem}
 button{padding:.6rem 1.2rem;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}
</style>
<h1>Classificador de Roupas — CNN</h1>
<p>Envie uma imagem (idealmente uma peça de roupa clara sobre fundo escuro).</p>
<input type="file" id="arq" accept="image/*">
<label><input type="checkbox" id="inv"> inverter cores</label>
<button onclick="enviar()">Classificar</button>
<div id="saida">aguardando…</div>
<script>
async function enviar(){
  const f = document.getElementById('arq').files[0];
  if(!f){ document.getElementById('saida').textContent = 'Selecione um arquivo.'; return; }
  const fd = new FormData(); fd.append('arquivo', f);
  const inv = document.getElementById('inv').checked;
  document.getElementById('saida').textContent = 'processando…';
  const r = await fetch('/prever?inverter=' + inv, {method:'POST', body: fd});
  document.getElementById('saida').textContent = JSON.stringify(await r.json(), null, 2);
}
</script></html>
"""
