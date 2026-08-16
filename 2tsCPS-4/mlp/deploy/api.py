"""
api.py — ETAPA 7 (parte B): servir o modelo como uma API web
--------------------------------------------------------------

Colocar em produção significa transformar o modelo em um SERVIÇO que qualquer
aplicação (site, app, outro backend) consegue chamar por HTTP.

Quatro decisões de engenharia visíveis neste arquivo:

1. NADA daqui importa src/. O serviço depende apenas de três artefatos gerados
   pelo treino: modelo_scriptado.pt, preprocessador.json e metadados.json. É
   assim que se separa o mundo da pesquisa do mundo da produção.

2. O pré-processamento é REIMPLEMENTADO aqui, em numpy puro, a partir do JSON.
   Sem pandas, sem sklearn, sem pickle. Repare que as constantes (medianas,
   categorias, médias, desvios) não estão escritas no código: elas vêm do
   arquivo, porque foram APRENDIDAS no treino. Código duplicado com constantes
   copiadas à mão é o erro silencioso mais comum em deploy tabular.

3. O modelo é carregado UMA VEZ, na subida do servidor — nunca por requisição.
   Carregar por requisição é o erro de desempenho mais comum em deploy de ML.

4. A resposta devolve a PROBABILIDADE e o LIMIAR usado, não só o "sim/não".
   Quem consome precisa poder aplicar a própria política de decisão: um banco
   pode querer cortar em 0,8; uma triagem médica, em 0,2.

Como rodar (a partir da pasta mlp/):
    pip install fastapi "uvicorn[standard]"
    python -m uvicorn deploy.api:app --reload --port 8000

Depois abra http://127.0.0.1:8000       (formulário de teste)
             http://127.0.0.1:8000/docs  (documentação interativa Swagger)
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_MODELO = RAIZ / "outputs" / "modelo_scriptado.pt"
CAMINHO_PREP = RAIZ / "outputs" / "preprocessador.json"
CAMINHO_META = RAIZ / "outputs" / "metadados.json"

modelo = None
prep: dict = {}
meta: dict = {}


# ---------------------------------------------------------------------------
# Pré-processamento em numpy puro, dirigido pelo JSON do treino
# ---------------------------------------------------------------------------
def preprocessar(registros: list[dict]) -> torch.Tensor:
    """Lista de cadastros -> tensor (n, n_features), idêntico ao do treino.

    A ordem das operações tem que ser exatamente a mesma de src/data.py:
        preencher ausente -> log1p (nas colunas marcadas) -> padronizar
    e as colunas one-hot seguem a ordem da lista de categorias do JSON.
    Qualquer divergência aqui derruba a acurácia sem gerar erro nenhum.
    """
    n = len(registros)
    colunas_num = prep["colunas_numericas"]
    colunas_log = set(prep["colunas_log"])
    colunas_cat = prep["colunas_categoricas"]
    ausente = prep["categoria_ausente"]

    blocos = []

    bloco_num = np.zeros((n, len(colunas_num)), dtype=np.float32)
    for j, coluna in enumerate(colunas_num):
        for i, registro in enumerate(registros):
            valor = registro.get(coluna, None)
            try:
                valor = float(valor)
            except (TypeError, ValueError):
                valor = float("nan")
            if not np.isfinite(valor):
                valor = prep["medianas"][coluna]        # mediana DO TREINO
            if coluna in colunas_log:
                valor = float(np.log1p(max(valor, 0.0)))
            bloco_num[i, j] = (valor - prep["media"][coluna]) / prep["desvio"][coluna]
    blocos.append(bloco_num)

    for coluna in colunas_cat:
        categorias = prep["categorias"][coluna]
        posicao = {c: k for k, c in enumerate(categorias)}
        bloco = np.zeros((n, len(categorias)), dtype=np.float32)
        for i, registro in enumerate(registros):
            valor = registro.get(coluna) or ausente
            indice = posicao.get(str(valor))
            if indice is not None:
                bloco[i, indice] = 1.0
            # Categoria desconhecida (um país que não estava no treino) deixa a
            # linha zerada — mesma convenção de src/data.py.
        blocos.append(bloco)

    return torch.from_numpy(np.concatenate(blocos, axis=1))


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carrega modelo e artefatos uma única vez, na subida do servidor.

    O que vem antes do `yield` roda no startup; o que vem depois, no shutdown
    (fechar conexões, liberar recursos).
    """
    global modelo, prep, meta
    faltando = [c for c in (CAMINHO_MODELO, CAMINHO_PREP, CAMINHO_META) if not c.exists()]
    if faltando:
        raise RuntimeError(
            f"Artefatos ausentes: {[str(c) for c in faltando]}\n"
            "Rode antes:  python src/train.py  e  python src/export_model.py")

    modelo = torch.jit.load(str(CAMINHO_MODELO), map_location="cpu")
    modelo.eval()
    prep = json.loads(CAMINHO_PREP.read_text(encoding="utf-8"))
    meta = json.loads(CAMINHO_META.read_text(encoding="utf-8"))
    print(f"[startup] modelo v{meta['versao_modelo']} | "
          f"{meta['n_features']} features | limiar {meta['limiar']}")
    yield
    print("[shutdown] encerrando serviço")


app = FastAPI(
    title="Previsão de faixa de renda (MLP)",
    description="Serviço de inferência do MLP treinado no Adult/Census Income",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


class Cadastro(BaseModel):
    """Contrato de entrada da API.

    O Pydantic valida tipos e devolve HTTP 422 com uma mensagem clara quando o
    cliente manda `age: "trinta"`. Um serviço de ML sem validação de entrada
    aceita lixo e responde com uma probabilidade — que é bem pior que um erro.

    Campos com `None` são tratados como ausentes: o pré-processador aplica a
    mediana (numéricos) ou "Desconhecido" (categóricos), exatamente como no
    treino.
    """
    age: float | None = Field(None, ge=0, le=120, description="idade em anos")
    workclass: str | None = Field(None, description="ex.: Private, Self-emp-not-inc")
    education_num: float | None = Field(None, ge=1, le=16, alias="education-num")
    marital_status: str | None = Field(None, alias="marital-status")
    occupation: str | None = None
    relationship: str | None = None
    race: str | None = None
    sex: str | None = None
    capital_gain: float | None = Field(None, ge=0, alias="capital-gain")
    capital_loss: float | None = Field(None, ge=0, alias="capital-loss")
    hours_per_week: float | None = Field(None, ge=0, le=168, alias="hours-per-week")
    native_country: str | None = Field(None, alias="native-country")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"example": {
            "age": 39, "workclass": "State-gov", "education-num": 13,
            "marital-status": "Never-married", "occupation": "Adm-clerical",
            "relationship": "Not-in-family", "race": "White", "sex": "Male",
            "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
            "native-country": "United-States"}},
    }

    def para_dicionario(self) -> dict:
        # by_alias=True devolve as chaves com hífen, que são os nomes das
        # colunas usados no treino e no preprocessador.json.
        return self.model_dump(by_alias=True)


@app.get("/saude")
def saude():
    """Health check — todo serviço em produção precisa de um.

    É este endpoint que o orquestrador (Docker, Kubernetes, load balancer)
    consulta para decidir se a instância está viva e pode receber tráfego.
    """
    return {
        "status": "ok",
        "modelo_carregado": modelo is not None,
        "versao_modelo": meta.get("versao_modelo"),
        "n_features": meta.get("n_features"),
        "limiar": meta.get("limiar"),
    }


@app.post("/prever")
def prever(cadastro: Cadastro, limiar: float | None = None):
    """Recebe um cadastro em JSON e devolve a probabilidade de renda > 50k."""
    corte = meta["limiar"] if limiar is None else limiar
    if not 0.0 < corte < 1.0:
        raise HTTPException(status_code=400, detail="limiar deve estar entre 0 e 1")

    t0 = time.perf_counter()
    tensor = preprocessar([cadastro.para_dicionario()])
    with torch.no_grad():
        probabilidade = torch.sigmoid(modelo(tensor))[0].item()
    ms = (time.perf_counter() - t0) * 1000

    return {
        "probabilidade_acima_50k": round(probabilidade, 4),
        "decisao": meta["classes"][1] if probabilidade >= corte else meta["classes"][0],
        "limiar": corte,
        # Sinaliza a decisão frágil em vez de escondê-la atrás de um "sim/não".
        "confiabilidade": "baixa" if abs(probabilidade - corte) < 0.05 else "normal",
        "versao_modelo": meta["versao_modelo"],
        "tempo_inferencia_ms": round(ms, 2),
    }


@app.get("/", response_class=HTMLResponse)
def pagina_teste():
    """Página mínima para demonstração em sala, sem precisar de Postman/curl."""
    return """
<!doctype html><html lang="pt-br"><meta charset="utf-8">
<title>Previsão de faixa de renda</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2.5rem auto;padding:0 1rem}
 .grade{display:grid;grid-template-columns:1fr 1fr;gap:.6rem 1rem}
 label{display:flex;flex-direction:column;font-size:.85rem;color:#334}
 input,select{padding:.4rem;border:1px solid #cbd5e1;border-radius:6px;font-size:.95rem}
 #saida{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:8px;margin-top:1rem}
 button{margin-top:1rem;padding:.6rem 1.2rem;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer}
</style>
<h1>Previsão de faixa de renda — MLP</h1>
<p>Modelo treinado no censo dos EUA de 1994. Estime a probabilidade de a renda
anual passar de US$ 50 mil.</p>
<div class="grade">
  <label>Idade <input id="age" type="number" value="39"></label>
  <label>Anos de estudo (1-16) <input id="education-num" type="number" value="13"></label>
  <label>Horas por semana <input id="hours-per-week" type="number" value="40"></label>
  <label>Ganho de capital <input id="capital-gain" type="number" value="0"></label>
  <label>Perda de capital <input id="capital-loss" type="number" value="0"></label>
  <label>Sexo <select id="sex"><option>Male</option><option>Female</option></select></label>
  <label>Estado civil <select id="marital-status">
    <option>Never-married</option><option>Married-civ-spouse</option>
    <option>Divorced</option><option>Separated</option><option>Widowed</option></select></label>
  <label>Ocupação <select id="occupation">
    <option>Adm-clerical</option><option>Exec-managerial</option><option>Prof-specialty</option>
    <option>Craft-repair</option><option>Sales</option><option>Other-service</option>
    <option>Machine-op-inspct</option><option>Handlers-cleaners</option></select></label>
  <label>Vínculo <select id="workclass">
    <option>Private</option><option>State-gov</option><option>Self-emp-not-inc</option>
    <option>Self-emp-inc</option><option>Federal-gov</option><option>Local-gov</option></select></label>
  <label>Relação familiar <select id="relationship">
    <option>Not-in-family</option><option>Husband</option><option>Wife</option>
    <option>Own-child</option><option>Unmarried</option></select></label>
</div>
<button onclick="enviar()">Prever</button>
<div id="saida">aguardando…</div>
<script>
async function enviar(){
  const campos = ['age','education-num','hours-per-week','capital-gain','capital-loss',
                  'sex','marital-status','occupation','workclass','relationship'];
  const corpo = {};
  for(const c of campos){
    const el = document.getElementById(c);
    corpo[c] = el.type === 'number' ? Number(el.value) : el.value;
  }
  document.getElementById('saida').textContent = 'processando…';
  const r = await fetch('/prever', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(corpo)});
  document.getElementById('saida').textContent = JSON.stringify(await r.json(), null, 2);
}
</script></html>
"""
