# Dockerfile — empacota o SERVIÇO de aquisição e filtragem em um contêiner.
#
# Construir e rodar, a partir da pasta nlp1/:
#   docker build -t nlp-aquisicao -f deploy/Dockerfile .
#   docker run -p 8000:8000 nlp-aquisicao
#
# Diferença importante em relação à imagem da aula de CNN: lá, o que entrava na
# imagem era o ARTEFATO (o modelo serializado), e o código de treino ficava de
# fora. Aqui não existe artefato — as regras de leitura, limpeza e filtragem SÃO
# o produto. Por isso `src/` entra na imagem, e o que fica de fora é o acervo
# (data/) e os resultados (outputs/), que são dados, não código.
#
# A imagem é pequena porque este pipeline não usa nenhuma biblioteca pesada:
# só leitores de formato em Python puro. ~150 MB no total.

FROM python:3.12-slim

WORKDIR /app

# Uma camada só para as dependências: enquanto o requirements não mudar, o
# cache do Docker é reaproveitado e o build seguinte leva segundos.
RUN pip install --no-cache-dir \
    pypdf==6.* python-docx==1.* openpyxl==3.* \
    fastapi==0.115.* "uvicorn[standard]==0.32.*" python-multipart pydantic==2.*

COPY src/    /app/src/
COPY deploy/api.py /app/deploy/api.py

# PYTHONUNBUFFERED: sem isso, o log do Python fica preso no buffer e você não
# vê nada em `docker logs` até o processo terminar — péssimo para diagnóstico.
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Health check: o orquestrador consulta este endpoint para decidir se a
# instância está viva e pode receber tráfego.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/saude')"

CMD ["python", "-m", "uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
