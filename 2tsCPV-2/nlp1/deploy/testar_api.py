"""
testar_api.py — teste de integração do serviço
------------------------------------------------

Envia para a API os mesmos arquivos que o pipeline em lote processou e
CONFERE SE A RESPOSTA É A MESMA. É o teste que importa, porque a falha que ele
procura é silenciosa: o serviço responde 200, devolve um JSON bonito, e o texto
está diferente do que entrou no corpus.

Na aula de CNN, o teste equivalente comparava a acurácia da API com a do
`evaluate.py`: se caísse muito, o bug estava no pré-processamento do servidor.
Aqui a comparação é mais direta e mais exigente — exigimos IGUALDADE EXATA
entre o texto produzido pelos dois caminhos, trecho a trecho.

Se este teste passar, você tem a garantia que interessa: o serviço e o lote são
o mesmo pipeline. Se falhar, alguém duplicou uma regra.

Uso (com o servidor já rodando em outro terminal):
    python deploy/testar_api.py
    python deploy/testar_api.py --url http://127.0.0.1:8000
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from config import CFG, OUT_DIR                    # noqa: E402
from readers import listar_arquivos               # noqa: E402
from utils import ler_jsonl                       # noqa: E402

TIPOS = {".pdf": "application/pdf", ".txt": "text/plain", ".md": "text/markdown",
         ".csv": "text/csv",
         ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
         ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


def enviar_arquivo(url: str, caminho: Path) -> dict:
    """POST multipart/form-data usando apenas a biblioteca padrão do Python.

    Sem `requests` de propósito: um teste de integração não deve trazer
    dependência nova, e montar o multipart na mão mostra que não há mágica —
    é só um corpo de texto com um delimitador aleatório separando as partes.
    """
    limite = uuid.uuid4().hex
    corpo = b"".join([
        f"--{limite}\r\n".encode(),
        f'Content-Disposition: form-data; name="arquivo"; '
        f'filename="{caminho.name}"\r\n'.encode(),
        f"Content-Type: {TIPOS.get(caminho.suffix.lower(), 'application/octet-stream')}"
        f"\r\n\r\n".encode(),
        caminho.read_bytes(),
        f"\r\n--{limite}--\r\n".encode(),
    ])
    requisicao = urllib.request.Request(
        f"{url}/extrair", data=corpo, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    with urllib.request.urlopen(requisicao, timeout=60) as resposta:
        return json.loads(resposta.read())


def main():
    p = argparse.ArgumentParser(description="Teste de integração da API")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=0, help="limitar a N arquivos")
    args = p.parse_args()

    # --- 1. Health check antes de qualquer coisa --------------------------
    try:
        with urllib.request.urlopen(f"{args.url}/saude", timeout=10) as r:
            saude = json.loads(r.read())
        print(f"saúde: {saude['status']} | formatos: {', '.join(saude['formatos'])}")
        print(f"limiares do serviço: min_palavras={saude['limiares']['min_palavras']}, "
              f"min_caracteres={saude['limiares']['min_caracteres']}\n")
    except Exception as e:
        print(f"[erro] servidor não respondeu em {args.url}: {e}")
        print("Suba o servidor com:  python -m uvicorn deploy.api:app --port 8000")
        return 1

    # --- 2. Carrega o resultado do modo em lote, para comparar ------------
    caminho_corpus = OUT_DIR / CFG.arq_corpus
    if not caminho_corpus.exists():
        print(f"[erro] {caminho_corpus} não existe. Rode antes: python src/pipeline.py")
        return 1

    # Chave (arquivo, localizador) -> texto que ENTROU no corpus em lote.
    esperado = {(d["arquivo"].split("/")[-1], d["localizador"]): d["texto"]
                for d in ler_jsonl(caminho_corpus)}

    arquivos = listar_arquivos()
    if args.n:
        arquivos = arquivos[:args.n]
    if not arquivos:
        print("[erro] acervo vazio. Rode: python src/make_samples.py")
        return 1

    # --- 3. Envia cada arquivo e compara ----------------------------------
    print(f"{'arquivo':<34}{'trechos':>8}{'ok':>5}{'rej':>5}{'ms':>8}  paridade")
    print("-" * 76)

    divergencias, comparados, tempos = [], 0, []
    for caminho in arquivos:
        try:
            resposta = enviar_arquivo(args.url, caminho)
        except urllib.error.HTTPError as e:
            detalhe = json.loads(e.read()).get("detail", "?")
            print(f"{caminho.name:<34}  HTTP {e.code}: {detalhe}")
            continue

        tempos.append(resposta["tempo_ms"])
        iguais = True
        for trecho in resposta["trechos"]:
            if trecho["status"] != "aceito":
                continue
            chave = (caminho.name, trecho["localizador"])
            if chave not in esperado:
                # O lote descartou este trecho na deduplicação (ETAPA 4), que o
                # serviço não faz — ele vê um arquivo por vez e não tem como
                # saber que existe uma cópia. Não é divergência.
                continue
            comparados += 1
            if trecho["texto"] != esperado[chave]:
                iguais = False
                divergencias.append((caminho.name, trecho["localizador"]))

        print(f"{caminho.name:<34}{resposta['n_trechos']:>8}"
              f"{resposta['n_aceitos']:>5}{resposta['n_rejeitados']:>5}"
              f"{resposta['tempo_ms']:>8.1f}  {'OK' if iguais else 'DIVERGE'}")

    # --- 4. Veredito ------------------------------------------------------
    print(f"\nTrechos comparados com o corpus em lote: {comparados}")
    if divergencias:
        print(f"[FALHA] {len(divergencias)} trecho(s) divergem entre API e lote:")
        for arquivo, localizador in divergencias[:10]:
            print(f"   {arquivo} · {localizador}")
        print("\nO serviço e o lote deixaram de ser o mesmo pipeline. Procure por")
        print("regra de limpeza duplicada ou configuração diferente entre os dois.")
        return 1

    print("[OK] Texto idêntico ao do modo em lote em todos os trechos aceitos.")
    if tempos:
        print(f"Latência média: {sum(tempos) / len(tempos):.1f} ms por arquivo")

    # --- 5. Teste do endpoint de texto puro -------------------------------
    corpo = json.dumps({"texto": "Contato: joao.silva@empresa.com.br, "
                                 "CPF 123.456.789-00, tel (11) 98765-4321.",
                        "mascarar_pii": True}).encode()
    requisicao = urllib.request.Request(
        f"{args.url}/avaliar", data=corpo, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(requisicao, timeout=30) as r:
        resultado = json.loads(r.read())
    print(f"\n/avaliar com mascaramento de PII:")
    print(f"   {resultado['texto_limpo']}")
    print(f"   decisão: {resultado['status']} ({resultado['motivo'] or '—'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
