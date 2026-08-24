"""
export_model.py — ETAPA 7 (parte A): empacotar o modelo para produção
----------------------------------------------------------------------

Na aula de MLP, esta etapa produzia TRÊS artefatos que precisavam viajar
juntos — `modelo_scriptado.pt`, `preprocessador.json` e `metadados.json` — e o
servidor reimplementava o pré-processamento inteiro em numpy. Publicar só o
primeiro era o erro nº 1 de deploy tabular.

Com o PyCaret, um único `.pkl` guarda o `Pipeline` inteiro:
imputação, one-hot, padronização e modelo, todos já ajustados. O erro de
"esquecer o pré-processador" some por construção. Esse é o ganho real da
abordagem, e é grande.

O QUE VOCÊ PAGA POR ESSE GANHO
------------------------------
O `.pkl` é um objeto Python serializado com pickle. Para abri-lo é preciso ter,
no ambiente de produção:

    * o PyCaret instalado (as classes `TransformerWrapper` vêm de lá);
    * o scikit-learn e o LightGBM em versões compatíveis com as do treino;
    * confiança na origem do arquivo — pickle EXECUTA código ao ser lido.

Compare com o TorchScript da aula anterior, que roda sem o código-fonte
original e até em C++. Não existe almoço grátis: trocamos "pré-processamento
que pode divergir" por "dependência de versões e formato que executa código".
Por isso `metadados.json` registra as versões: é o que permite reconstruir o
ambiente daqui a seis meses, quando o modelo der um número estranho.

E A REGRA DE OURO DO DEPLOY CONTINUA VALENDO
--------------------------------------------
Nunca confie numa exportação sem comparar as saídas. Este script recarrega o
arquivo salvo e exige que as probabilidades batam com as do modelo em memória
até 1e-9. É uma verificação de dez linhas que já evitou muito domingo perdido.

Uso:
    python src/export_model.py
"""

import argparse
import platform

import joblib
import numpy as np
from pycaret.classification import load_model

from config import (ALVO_BINARIO, CFG, CLASSES, COLUNAS_CATEGORICAS,
                    COLUNAS_LOG, COLUNAS_NUMERICAS, OUT_DIR)
from data import obter_particoes
from evaluate import carregar_modelo
from utils import carregar_json, salvar_json, silenciar_bibliotecas


def versoes() -> dict:
    """Registra as versões que produziram o artefato.

    Um `.pkl` sem esta informação é uma bomba-relógio: daqui a seis meses,
    quando ele falhar ao carregar ou (pior) carregar e prever diferente,
    ninguém vai saber com o que ele foi feito.
    """
    import sklearn
    import pycaret
    info = {
        "python": platform.python_version(),
        "pycaret": pycaret.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
    }
    for nome in ("lightgbm", "xgboost", "catboost"):
        try:
            info[nome] = __import__(nome).__version__
        except Exception:
            pass
    return info


def exportar(rapido: bool = False) -> dict:
    silenciar_bibliotecas()

    pipeline = carregar_modelo()
    historico = carregar_json(OUT_DIR / CFG.arq_historico)
    limiar = historico.get("limiar", CFG.limiar_padrao)

    # --- 1) o artefato de produção ----------------------------------------
    # `save_model()` do PyCaret exige um experimento ativo (ele guarda o
    # contexto do setup junto). Aqui estamos apenas REEMPACOTANDO um pipeline
    # que já foi treinado e salvo, então usamos o joblib direto — que é
    # exatamente o que o save_model faz por baixo. O `load_model` lê os dois
    # da mesma forma.
    caminho = OUT_DIR / CFG.arq_pipeline
    arquivo = OUT_DIR / f"{CFG.arq_pipeline}.pkl"
    joblib.dump(pipeline, arquivo)
    tamanho = arquivo.stat().st_size / 1024

    # --- 2) os metadados ---------------------------------------------------
    # Tudo o que o SERVIÇO precisa saber e que não está dentro do pipeline:
    # o limiar (que é decisão de negócio), os nomes das classes, quais colunas
    # ele espera receber e — atenção — quais transformações ficaram FORA do
    # pipeline e portanto são responsabilidade de quem chama.
    metadados = {
        "versao_modelo": "1.0.0",
        "descricao": "Classificador de faixa de renda anual (Adult/UCI 1994), "
                     "treinado com PyCaret",
        "modelo": historico.get("modelo"),
        "modelo_base": historico.get("modelo_base"),
        "classes": CLASSES,
        "limiar": limiar,
        "alvo": ALVO_BINARIO,
        "auc_validacao": historico.get("auc_validacao"),
        # As 12 colunas do cadastro, na forma em que o pipeline as espera.
        "colunas_numericas": list(COLUNAS_NUMERICAS),
        "colunas_categoricas": list(COLUNAS_CATEGORICAS),
        # ESTA é a pegadinha desta abordagem: log1p foi aplicado em data.py,
        # ANTES do setup(), e por isso NÃO está dentro do pipeline. Quem manda
        # dados para o modelo precisa aplicá-lo também. Registramos a lista
        # aqui para que o servidor leia do arquivo em vez de repetir a
        # constante no código — constante copiada à mão é constante que
        # diverge. Ver a discussão em deploy/api.py.
        "colunas_log1p_externas": list(COLUNAS_LOG),
        "versoes": versoes(),
    }
    salvar_json(metadados, OUT_DIR / CFG.arq_metadados)

    print(f"[ok] Pipeline salvo em {OUT_DIR / (CFG.arq_pipeline + '.pkl')}")
    print(f"[ok] Metadados salvos em {OUT_DIR / CFG.arq_metadados}")
    print(f"     modelo publicado : {metadados['modelo']}")
    print(f"     limiar publicado : {limiar:.2f}")
    print(f"     tamanho do artefato: {tamanho:.1f} KB")

    # --- 3) verificação obrigatória ---------------------------------------
    # Compara as probabilidades do objeto em memória com as do arquivo
    # recarregado. Diferença acima de 1e-9 significa que algo não sobreviveu à
    # serialização — e é melhor descobrir agora que em produção.
    #
    # Usamos `predict_proba` NOS DOIS LADOS de propósito. Se um lado passar por
    # `predict_model`, a comparação mede o arredondamento dele (4 casas por
    # padrão) e não a serialização: aparece uma "divergência" de 5e-05 que não
    # existe. Verificação que mistura dois caminhos de acesso não verifica
    # nada — ela mede a diferença entre os caminhos.
    _, df_val, _ = obter_particoes(CFG, rapido=rapido)
    amostra = df_val.head(500).drop(columns=[ALVO_BINARIO])

    recarregado = load_model(str(caminho), verbose=False)
    prob_original = pipeline.predict_proba(amostra)[:, 1]
    prob_recarregado = recarregado.predict_proba(amostra)[:, 1]

    diferenca = float(np.abs(prob_original - prob_recarregado).max())
    print(f"\n     diferença máxima original vs. recarregado: {diferenca:.2e} "
          f"({'OK' if diferenca < 1e-9 else 'ATENÇÃO: divergência!'})")

    print("\n     Ambiente que produziu este artefato (grave junto com ele):")
    for pacote, versao in metadados["versoes"].items():
        print(f"       {pacote:<14} {versao}")

    print("\n     Passos do pipeline que foram serializados:")
    for nome_passo, passo in pipeline.steps:
        print(f"       {nome_passo:<26} {type(passo).__name__}")
    print("\n     Repare que o pré-processamento inteiro está AQUI DENTRO — é o\n"
          "     que torna impossível servir o modelo com a transformação errada.\n"
          f"     Exceção: o log1p de {COLUNAS_LOG}, aplicado antes do setup().")

    metadados["verificacao_diferenca_maxima"] = diferenca
    return metadados


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exportação do modelo para produção")
    p.add_argument("--rapido", action="store_true")
    args = p.parse_args()
    exportar(args.rapido)
