# Dockerfile — empacota o SERVICO (nao o treino) em um conteiner.
#
# Construir e rodar, a partir da pasta pycaret/:
#   docker build -t pycaret-renda -f deploy/Dockerfile .
#   docker run -p 8000:8000 pycaret-renda
#
# Observe o que entra na imagem: apenas o codigo do servico e os dois artefatos
# (pipeline + metadados). Dataset, notebooks e scripts de treino ficam de fora.
#
# ---------------------------------------------------------------------------
# O PRECO DO PICKLE — leia antes de reclamar do tamanho da imagem
# ---------------------------------------------------------------------------
# Na aula de MLP, a imagem instalava torch-cpu (~300 MB) e carregava um
# TorchScript, que e um formato autocontido: roda sem o codigo-fonte original.
#
# Aqui o artefato e um .pkl. Para desserializa-lo o Python precisa das MESMAS
# classes que o gravaram — TransformerWrapper vem do pycaret, o modelo vem do
# lightgbm. Resultado: a imagem precisa do pycaret inteiro (~1.5 GB), mesmo que
# so va rodar predict.
#
# Esse e o trade-off central da ETAPA 7 desta aula:
#
#   PyTorch/TorchScript : imagem pequena, formato aberto, MAS voce reimplementa
#                         o pre-processamento no servidor (e ele pode divergir).
#   PyCaret/pickle      : pre-processamento vem junto e nao pode divergir, MAS
#                         a imagem carrega a biblioteca inteira e as versoes
#                         viram parte do contrato.
#
# As versoes abaixo saem de outputs/metadados.json. Se elas nao baterem com as
# do treino, o .pkl pode ate carregar — e prever diferente, em silencio.

FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
      pycaret==3.3.2 \
      lightgbm==4.5.0 \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.34"

COPY deploy/api.py                  /app/deploy/api.py
COPY outputs/pipeline_renda.pkl     /app/outputs/pipeline_renda.pkl
COPY outputs/metadados.json         /app/outputs/metadados.json

EXPOSE 8000

# Health check: o Docker reinicia o conteiner se o endpoint parar de responder.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/saude')"

CMD ["python", "-m", "uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
