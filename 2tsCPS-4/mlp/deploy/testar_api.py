"""
testar_api.py — teste de integração do serviço
-----------------------------------------------

Pega linhas reais do conjunto de TESTE, envia para a API como JSON e compara a
resposta com o rótulo verdadeiro.

Por que este script existe: o pré-processamento do servidor foi REESCRITO em
numpy (deploy/api.py), separado do código de treino (src/data.py). Duas
implementações da mesma regra sempre acabam divergindo — uma coluna fora de
ordem, um log1p esquecido, uma média copiada errada. Nada disso levanta
exceção: o serviço continua respondendo probabilidades, só que erradas.

Se as métricas aqui ficarem muito abaixo das de `evaluate.py`, o problema está
no DEPLOY, não no modelo. Este é o erro silencioso mais comum em produção de
aprendizado de máquina tabular.

Uso (com o servidor já rodando em outro terminal):
    python deploy/testar_api.py
    python deploy/testar_api.py --n 1000 --url http://127.0.0.1:8000

Use poucas amostras (--n 50) e a diferença para o resultado offline cresce só
por ruído amostral: com 500 linhas ela cai para ~0,003.
"""

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import pandas as pd                                                # noqa: E402
from config import (CFG, CLASSE_POSITIVA, COLUNAS, COLUNAS_DESCARTADAS,   # noqa: E402
                    DATA_DIR, OUT_DIR)


def carregar_amostras(n: int) -> list[dict]:
    """Lê n linhas do adult.test — dados que o modelo nunca viu."""
    caminho_zip = DATA_DIR / CFG.arq_dados
    if not caminho_zip.exists():
        raise FileNotFoundError(
            f"{caminho_zip} não encontrado. Rode antes: python src/data.py")

    with zipfile.ZipFile(caminho_zip) as z:
        bruto = z.read("adult.test")
    df = pd.read_csv(io.BytesIO(bruto), names=COLUNAS, sep=",", skiprows=1,
                     skipinitialspace=True, na_values="?")
    df = df.sample(n=min(n, len(df)), random_state=CFG.semente)

    amostras = []
    for _, linha in df.iterrows():
        verdadeiro = int(str(linha["income"]).rstrip(".") == CLASSE_POSITIVA)
        registro = linha.drop(labels=["income"] + COLUNAS_DESCARTADAS).to_dict()
        # NaN não é JSON válido: vira null, que a API trata como ausente.
        registro = {k: (None if pd.isna(v) else v) for k, v in registro.items()}
        amostras.append({"registro": registro, "y": verdadeiro})
    return amostras


def chamar(url: str, registro: dict) -> dict:
    """POST application/json usando apenas a biblioteca padrão do Python."""
    corpo = json.dumps(registro).encode("utf-8")
    requisicao = urllib.request.Request(
        f"{url}/prever", data=corpo, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(requisicao, timeout=30) as resposta:
        return json.loads(resposta.read())


def main():
    p = argparse.ArgumentParser(description="Teste de integração da API")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--mostrar", type=int, default=8, help="quantas linhas detalhar")
    args = p.parse_args()

    # 1) Health check antes de qualquer coisa
    try:
        with urllib.request.urlopen(f"{args.url}/saude", timeout=10) as r:
            saude = json.loads(r.read())
        print("saúde:", saude, "\n")
    except Exception as e:
        print(f"[erro] servidor não respondeu em {args.url}: {e}")
        print("Suba o servidor com:  python -m uvicorn deploy.api:app --port 8000")
        return

    # 2) Envia as amostras uma a uma e confere contra o rótulo verdadeiro
    amostras = carregar_amostras(args.n)
    vp = vn = fp = fn = 0
    tempos = []

    for i, amostra in enumerate(amostras):
        try:
            resposta = chamar(args.url, amostra["registro"])
        except urllib.error.HTTPError as e:
            print(f"[erro] HTTP {e.code}: {e.read().decode()[:300]}")
            return

        previsto = int(resposta["probabilidade_acima_50k"] >= resposta["limiar"])
        y = amostra["y"]
        vp += previsto == 1 and y == 1
        vn += previsto == 0 and y == 0
        fp += previsto == 1 and y == 0
        fn += previsto == 0 and y == 1
        tempos.append(resposta["tempo_inferencia_ms"])

        if i < args.mostrar:
            r = amostra["registro"]
            print(f"[{'OK  ' if previsto == y else 'ERRO'}] "
                  f"{str(r['age']):>3} anos, {str(r['occupation'])[:18]:<18} | "
                  f"P(>50K)={resposta['probabilidade_acima_50k']:.3f} -> "
                  f"{resposta['decisao']:<6} | real: {'>50K' if y else '<=50K'} | "
                  f"{resposta['tempo_inferencia_ms']:.1f} ms")

    total = len(amostras)
    acuracia = (vp + vn) / total
    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0

    print(f"\n{'-' * 62}")
    print(f"Amostras     : {total}")
    print(f"Acurácia     : {acuracia:.2%}   (VN={vn} FP={fp} FN={fn} VP={vp})")
    print(f"Precisão     : {precisao:.3f}")
    print(f"Revocação    : {revocacao:.3f}")
    print(f"F1           : {f1:.3f}")
    print(f"Latência média: {sum(tempos) / len(tempos):.2f} ms por requisição")

    # 3) Compara com o resultado offline: é ISSO que valida o deploy
    caminho_metricas = OUT_DIR / CFG.arq_metricas
    if caminho_metricas.exists():
        offline = json.loads(caminho_metricas.read_text(encoding="utf-8"))["metricas_teste"]
        diferenca = abs(offline["f1"] - f1)
        print(f"\nF1 offline (evaluate.py): {offline['f1']:.3f}")
        print(f"F1 pela API            : {f1:.3f}   | diferença {diferenca:.3f}")
        if diferenca > 0.05:
            print("\n[ALERTA] Divergência grande entre a API e a avaliação offline.\n"
                  "         Procure o bug no PRÉ-PROCESSAMENTO do servidor:\n"
                  "         ordem das colunas, log1p, médias/desvios, one-hot.")
        else:
            print("\n[OK] API e avaliação offline concordam — pré-processamento "
                  "replicado corretamente.\n     (A diferença residual é só a "
                  "amostragem: aqui usamos um subconjunto do teste.)")
    else:
        print("\n[aviso] rode `python src/evaluate.py` para ter a referência offline.")


if __name__ == "__main__":
    main()
