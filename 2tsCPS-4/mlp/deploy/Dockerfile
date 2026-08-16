# Dockerfile — empacota o SERVICO (nao o treino) em um conteiner.
#
# Construir e rodar, a partir da pasta mlp/:
#   docker build -t mlp-renda -f deploy/Dockerfile .
#   docker run -p 8000:8000 mlp-renda
#
# Observe o que entra na imagem: apenas o codigo do servico e os tres artefatos
# (modelo + preprocessador + metadados). Dataset, notebooks e scripts de treino
# ficam de fora — imagem de producao deve ser minima.

FROM python:3.12-slim

WORKDIR /app

# torch em versao CPU: a imagem cai de ~2.5 GB para ~300 MB.
# Um MLP de 8 mil parametros nao tem o que fazer com uma GPU.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install --no-cache-dir \
    fastapi==0.115.* "uvicorn[standard]==0.32.*" numpy

COPY deploy/api.py                 /app/deploy/api.py
COPY outputs/modelo_scriptado.pt   /app/outputs/modelo_scriptado.pt
COPY outputs/preprocessador.json   /app/outputs/preprocessador.json
COPY outputs/metadados.json        /app/outputs/metadados.json

EXPOSE 8000

# Health check: o Docker reinicia o conteiner se o endpoint parar de responder.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/saude')"

CMD ["python", "-m", "uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
