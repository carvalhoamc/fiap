"""
api.py — ETAPA 7 (parte B): servir o modelo como uma API web
--------------------------------------------------------------

Colocar em produção significa transformar o modelo em um SERVIÇO que qualquer
aplicação (site, app, outro backend) consegue chamar por HTTP.

COMPARE COM O api.py DA AULA DE MLP
-----------------------------------
Lá, este arquivo tinha 32 linhas de código (uns 60 com os comentários) só de
pré-processamento reimplementado em numpy: preencher com a mediana, aplicar
log1p, padronizar, montar as 90 colunas one-hot na ordem exata. Duas
implementações da mesma regra sempre acabam divergindo, e a divergência não
levanta exceção nenhuma — o serviço continua respondendo probabilidades, só que
erradas.

Aqui essas 32 linhas viraram uma: `pipeline.predict_proba(df)`. O
pré-processamento veio dentro do `.pkl`, ajustado no treino, e é por definição
o mesmo.

O QUE **NÃO** DESAPARECEU, E POR QUE ISSO É A LIÇÃO DESTE ARQUIVO
-----------------------------------------------------------------
O `log1p` de `capital-gain` e `capital-loss` foi aplicado em `src/data.py`,
ANTES do `setup()`. Logo, ele NÃO está dentro do pipeline — e este servidor
precisa repeti-lo. São três linhas, contra as sessenta da aula anterior, mas o
tipo de risco é exatamente o mesmo: se alguém mudar a transformação no treino e
esquecer daqui, o serviço passa a mentir em silêncio.

Duas providências, ambas visíveis abaixo:

  1. A lista de colunas a transformar vem de `metadados.json`, não está escrita
     no código. Constante copiada à mão é constante que diverge.
  2. `testar_api.py` compara o F1 do serviço com o do `evaluate.py`. É esse
     teste — e não a leitura do código — que prova que o deploy está correto.

Quatro decisões de engenharia, as mesmas de qualquer serviço de ML:

  1. NADA aqui importa `src/`. O serviço depende de dois arquivos:
     `pipeline_renda.pkl` e `metadados.json`.
  2. O modelo é carregado UMA VEZ, na subida do servidor — nunca por
     requisição. Carregar por requisição é o erro de desempenho nº 1 em ML.
  3. A entrada é validada pelo Pydantic. Um serviço de ML sem validação aceita
     lixo e responde com uma probabilidade, que é bem pior que um erro.
  4. A resposta devolve a PROBABILIDADE e o LIMIAR, não só o "sim/não". Quem
     consome precisa poder aplicar a própria política de decisão.

Como rodar (a partir da pasta pycaret/):
    python -m uvicorn deploy.api:app --reload --port 8000

Depois abra http://127.0.0.1:8000       (formulário de teste)
             http://127.0.0.1:8000/docs  (documentação interativa Swagger)
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pycaret.classification import load_model

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_MODELO = RAIZ / "outputs" / "pipeline_renda"      # sem .pkl: load_model acrescenta
CAMINHO_META = RAIZ / "outputs" / "metadados.json"

pipeline = None
meta: dict = {}


# ---------------------------------------------------------------------------
# O pedaço de pré-processamento que ficou FORA do pipeline
# ---------------------------------------------------------------------------
def preparar(registros: list[dict]) -> pd.DataFrame:
    """Lista de cadastros -> DataFrame no formato que o pipeline espera.

    Só duas coisas acontecem aqui, e as duas são consequência de decisões
    tomadas no treino:

      * `log1p` nas colunas listadas em `colunas_log1p_externas` — a
        transformação que ficou fora do pipeline;
      * garantir que TODAS as colunas esperadas existam. Campo que não veio
        entra como nulo e é imputado lá dentro, com a mediana do treino (nos
        numéricos) ou "Desconhecido" (nos categóricos).

    Imputação, one-hot e padronização NÃO estão aqui: quem faz é o pipeline.
    """
    df = pd.DataFrame(registros)

    for coluna in meta["colunas_numericas"]:
        if coluna not in df.columns:
            df[coluna] = np.nan
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    for coluna in meta["colunas_categoricas"]:
        if coluna not in df.columns:
            df[coluna] = None
        df[coluna] = df[coluna].fillna("Desconhecido").astype(str)

    # A lista vem do arquivo, não do código — é o que impede a divergência.
    for coluna in meta.get("colunas_log1p_externas", []):
        df[coluna] = np.log1p(df[coluna].clip(lower=0))

    return df[list(meta["colunas_numericas"]) + list(meta["colunas_categoricas"])]


def probabilidades(df: pd.DataFrame) -> np.ndarray:
    """P(classe 1) para cada linha, pelo caminho rápido.

    POR QUE `pipeline.predict_proba` E NÃO `predict_model`
    ------------------------------------------------------
    Em `src/` usamos `predict_model`, que é a interface idiomática do PyCaret:
    devolve um DataFrame anotado, calcula métricas se o alvo estiver presente e
    é ótima para explorar. Em produção ela custa caro. Medido nesta máquina,
    uma linha por requisição:

        predict_model(pipeline, data=df)  ->  ~153 ms  (~168 ms via HTTP)
        pipeline.predict_proba(df)        ->  ~ 40 ms  (~ 54 ms via HTTP)

    A diferença é toda infraestrutura de conveniência: validação de argumentos,
    montagem do DataFrame de saída, checagem de colunas de rótulo. Nada disso
    serve a um endpoint HTTP.

    Os ~40 ms restantes são custo FIXO por chamada (montar o DataFrame,
    atravessar os 6 transformadores do pipeline), não custo por linha. É por
    isso que `/prever_lote` sai a 0,07 ms por registro com 500 de uma vez:
    o mesmo custo fixo, diluído. Em serviço de ML, o lote quase sempre é a
    diferença entre "funciona" e "aguenta produção".

    Há ainda um detalhe silencioso: `predict_model` ARREDONDA a probabilidade
    para 4 casas por padrão (`round=4`). Para um limiar em 0,41 isso é
    irrelevante; para uma política que corte em 0,9999 (fraude, por exemplo),
    não é. `predict_proba` devolve o número cheio.

    O `pipeline` continua sendo o mesmo objeto — o pré-processamento aplicado é
    idêntico. Muda só o invólucro.
    """
    return pipeline.predict_proba(df)[:, 1]


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carrega modelo e metadados uma única vez, na subida do servidor.

    O que vem antes do `yield` roda no startup; o que vem depois, no shutdown.
    """
    global pipeline, meta
    if not (RAIZ / "outputs" / "pipeline_renda.pkl").exists() or not CAMINHO_META.exists():
        raise RuntimeError(
            "Artefatos ausentes em outputs/.\n"
            "Rode antes:  python src/train.py  e  python src/export_model.py")

    meta = json.loads(CAMINHO_META.read_text(encoding="utf-8"))
    pipeline = load_model(str(CAMINHO_MODELO), verbose=False)
    print(f"[startup] modelo {meta['modelo']} v{meta['versao_modelo']} | "
          f"limiar {meta['limiar']} | treinado com pycaret "
          f"{meta['versoes'].get('pycaret')}")
    yield
    print("[shutdown] encerrando serviço")


app = FastAPI(
    title="Previsão de faixa de renda (PyCaret)",
    description="Serviço de inferência do pipeline treinado no Adult/Census Income",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)


class Cadastro(BaseModel):
    """Contrato de entrada da API.

    O Pydantic valida tipos e devolve HTTP 422 com mensagem clara quando o
    cliente manda `age: "trinta"`. Campos com `None` são tratados como
    ausentes: o pipeline imputa a mediana (numéricos) ou "Desconhecido"
    (categóricos), exatamente como no treino.
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
        # colunas usados no treino e registrados em metadados.json.
        return self.model_dump(by_alias=True)


@app.get("/saude")
def saude():
    """Health check — todo serviço em produção precisa de um.

    É este endpoint que o orquestrador (Docker, Kubernetes, load balancer)
    consulta para decidir se a instância está viva e pode receber tráfego.
    """
    return {
        "status": "ok",
        "modelo_carregado": pipeline is not None,
        "modelo": meta.get("modelo"),
        "versao_modelo": meta.get("versao_modelo"),
        "limiar": meta.get("limiar"),
        "versoes_treino": meta.get("versoes"),
    }


@app.post("/prever")
def prever(cadastro: Cadastro, limiar: float | None = None):
    """Recebe um cadastro em JSON e devolve a probabilidade de renda > 50k."""
    corte = meta["limiar"] if limiar is None else limiar
    if not 0.0 < corte < 1.0:
        raise HTTPException(status_code=400, detail="limiar deve estar entre 0 e 1")

    t0 = time.perf_counter()
    df = preparar([cadastro.para_dicionario()])
    probabilidade = float(probabilidades(df)[0])
    ms = (time.perf_counter() - t0) * 1000

    return {
        "probabilidade_acima_50k": round(probabilidade, 4),
        "decisao": meta["classes"][1] if probabilidade >= corte else meta["classes"][0],
        "limiar": corte,
        # Sinaliza a decisão frágil em vez de escondê-la atrás de um "sim/não".
        "confiabilidade": "baixa" if abs(probabilidade - corte) < 0.05 else "normal",
        "modelo": meta["modelo"],
        "versao_modelo": meta["versao_modelo"],
        "tempo_inferencia_ms": round(ms, 2),
    }


@app.post("/prever_lote")
def prever_lote(cadastros: list[Cadastro], limiar: float | None = None):
    """Versão em lote: uma chamada, N cadastros.

    Existe por um motivo de desempenho concreto: o custo fixo de aplicar o
    pipeline (validar colunas, montar o one-hot, chamar o modelo) é pago uma
    vez para o lote inteiro. Comparar a latência por registro deste endpoint
    com a do /prever é um exercício de cinco minutos que ensina bastante.
    """
    corte = meta["limiar"] if limiar is None else limiar
    if not 0.0 < corte < 1.0:
        raise HTTPException(status_code=400, detail="limiar deve estar entre 0 e 1")
    if not cadastros:
        raise HTTPException(status_code=400, detail="lista vazia")

    t0 = time.perf_counter()
    df = preparar([c.para_dicionario() for c in cadastros])
    prob = probabilidades(df)
    ms = (time.perf_counter() - t0) * 1000

    return {
        "n": len(cadastros),
        "limiar": corte,
        "tempo_total_ms": round(ms, 2),
        "tempo_por_registro_ms": round(ms / len(cadastros), 3),
        "resultados": [
            {"probabilidade_acima_50k": round(float(p), 4),
             "decisao": meta["classes"][1] if p >= corte else meta["classes"][0]}
            for p in prob
        ],
    }


@app.get("/", response_class=HTMLResponse)
def pagina_teste():
    """Página mínima para demonstração em sala, sem precisar de Postman/curl."""
    return """
<!doctype html><html lang="pt-br"><meta charset="utf-8">
<title>Previsão de faixa de renda — PyCaret</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2.5rem auto;padding:0 1rem}
 .grade{display:grid;grid-template-columns:1fr 1fr;gap:.6rem 1rem}
 label{display:flex;flex-direction:column;font-size:.85rem;color:#334}
 input,select{padding:.4rem;border:1px solid #cbd5e1;border-radius:6px;font-size:.95rem}
 #saida{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:8px;margin-top:1rem}
 button{margin-top:1rem;padding:.6rem 1.2rem;border:0;border-radius:6px;background:#16a34a;color:#fff;cursor:pointer}
</style>
<h1>Previsão de faixa de renda — PyCaret</h1>
<p>Pipeline treinado no censo dos EUA de 1994. Estime a probabilidade de a renda
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
