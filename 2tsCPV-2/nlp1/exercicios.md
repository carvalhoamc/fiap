# Atividades — Aquisição e filtragem de textos

> Antes de começar, execute o roteiro completo do [README](README.md) pelo menos
> uma vez e **guarde os resultados de referência**: o `relatorio.json`, a tabela
> de `python src/filters.py` e a taxa de aproveitamento. Todo exercício abaixo é
> comparado com **essa linha de base**. Um experimento sem baseline não prova
> nada.
>
> Linha de base desta implementação: **14 arquivos → 27 trechos → 20 documentos
> (74,1% de aproveitamento), 1.553 palavras, 23 blocos.**

Regra permanente: **mexa em uma variável por vez** e registre o resultado em uma
tabela. Mudar três limiares e ver o aproveitamento subir não diz qual dos três
funcionou.

E uma regra específica desta aula, que vale mais que todas: **um número de
aproveitamento nunca é resposta suficiente.** Sempre que um limiar mudar, abra
`rejeitados.jsonl` e leia o que entrou e o que saiu. O objetivo não é maximizar
o aproveitamento — é acertar quais documentos ficam.

---

## Nível 1 — Compreensão

**1.1 Codificação.** A palavra `inspeção` tem 8 caracteres.

- (a) Quantos **bytes** ela ocupa em UTF-8? E em cp1252? Explique a diferença.
- (b) O que acontece, exatamente, ao ler os bytes UTF-8 dessa palavra usando
  cp1252? Escreva o resultado.
- (c) Por que `latin-1` é a **última** codificação da cadeia em
  `config.codificacoes`, e não a primeira? (Dica: existe algum byte que
  `latin-1` recuse?)
- (d) Confirme tudo com:
  `python -c "print('inspeção'.encode('utf-8').decode('cp1252'))"`

**1.2 O silêncio do PDF escaneado.** O arquivo `nota_fiscal_escaneada.pdf`
produz um trecho com texto vazio.

- (a) Por que `pypdf` **não levanta exceção** nesse caso?
- (b) Imagine um pipeline que descartasse trechos vazios em silêncio, sem
  registrá-los. Rodando sobre um acervo de 4.000 PDFs em que 30% são
  escaneados, o que o relatório final diria? O que estaria errado nessa
  afirmação?
- (c) Por que a correção certa é OCR, e **não** ajustar um limiar de filtragem?

**1.3 Limpar ou destruir.** Para cada operação abaixo, diga **o que se perde** e
dê um exemplo concreto em que essa perda importa:

| Operação | O que se perde | Exemplo em que importa |
|---|---|---|
| passar tudo para minúsculas | ? | ? |
| remover acentos | ? | ? |
| remover pontuação | ? | ? |
| remover stopwords | ? | ? |

Depois explique, em duas frases, por que essa receita fazia sentido com TF-IDF e
deixou de fazer com modelos baseados em transformadores.

**1.4 Jaccard na mão.** Considere:

```
A = "o gato subiu no telhado"
B = "o gato desceu do telhado"
```

- (a) Liste os shingles de **2 palavras** de cada um.
- (b) Calcule $J(A,B) = |A \cap B| / |A \cup B|$.
- (c) Refaça com shingles de **4 palavras**. O que aconteceu com a similaridade?
- (d) Generalizando: por que shingle pequeno demais aponta similaridade onde há
  apenas assunto em comum, e shingle grande demais não acha cópia nenhuma?
- (e) Confira com
  `python -c "import sys;sys.path.insert(0,'src');from dedup import shingles,jaccard_exato as j;print(j(shingles('o gato subiu no telhado',2),shingles('o gato desceu do telhado',2)))"`

**1.5 A ordem das operações.** Em `clean.limpar()`, a remoção de moldura (passo
5) vem **antes** do rejuntar de parágrafos (passo 7).

- (a) O que acontece com a linha `Pagina 1 de 3` se essa ordem for invertida?
- (b) Por que o resultado seria um erro **silencioso** — isto é, sem exceção e
  com aparência de texto correto?
- (c) Inverta a ordem no código, rode
  `python src/clean.py data/brutos/relatorio_tecnico.pdf` e mostre a saída.
  Depois desfaça.

---

## Nível 2 — Experimentação controlada

Preencha esta tabela para **cada** item, reexecutando `python src/pipeline.py` e
`python src/report.py`:

| Experimento | Trechos | Aceitos | Duplicatas | Corpus | Aproveit. | O que mudou de errado/certo |
|---|---|---|---|---|---|---|
| Linha de base | 27 | 22 | 2 | 20 | 74,1% | — |
| ... | | | | | | |

**2.1 O limiar que mata a planilha.** Em `config.py`, mude `min_palavras` de 25
para 40 (o valor "natural" para páginas de PDF).

- (a) Quantos documentos o corpus perde? **Quais fontes** desaparecem por
  completo?
- (b) Rode `python src/report.py --motivo curto_demais` e leia os descartes.
  Você concorda com todos eles?
- (c) Por que uma linha de planilha tem naturalmente menos palavras que uma
  página de PDF? Isso significa que ela vale menos?
- (d) Proponha uma correção que preserve as duas fontes. (A implementação é o
  exercício 3.6.)

**2.2 Sensibilidade do limiar de duplicata.** Rode `python src/dedup.py` com
`--limiar` em 0,60, 0,75, 0,80 e 0,90.

- (a) Monte a tabela limiar × nº de duplicatas encontradas.
- (b) Em que valor exato `noticia_editada.txt` deixa de ser capturada? Explique
  usando a similaridade medida (0,783).
- (c) Um limiar baixo demais causa qual tipo de erro? E um alto demais? Qual dos
  dois é pior para um sistema de busca? E para treinar um modelo?

**2.3 Tamanho do shingle.** Rode `python src/dedup.py --shingle 2`, `5` e `10`,
mantendo o limiar em 0,75.

- (a) Como a similaridade entre `noticia.txt` e `noticia_editada.txt` muda?
- (b) Com shingle 2, algum documento **não relacionado** passa a ser marcado como
  duplicata? Se sim, qual e por quê?

**2.4 O detector de idioma do pobre.** Varie `min_prop_stopwords` entre 0,02 e
0,20.

- (a) Em que valor `abstract.txt` (inglês) passa a ser aceito?
- (b) Em que valor algum documento **em português** começa a ser recusado? Qual
  é o primeiro a cair, e por quê ele é o mais frágil? (Olhe a coluna `stop` da
  tabela de `filters.py`.)
- (c) A distância entre esses dois valores é a **margem de segurança** do filtro.
  Ela é confortável? O que aconteceria com um corpus que misturasse português e
  espanhol?

**2.5 O custo de não deduplicar.** Rode `python src/pipeline.py --sem-dedup`.

- (a) Quantas palavras a mais o corpus tem? Que fração é redundante?
- (b) Suponha que esse corpus fosse dividido em treino e teste ao acaso. Qual é
  a chance de `noticia.txt` cair no treino e `noticia_copia.txt` no teste? O que
  isso faz com a métrica de avaliação?

**2.6 Uma métrica que não media nada.** Em `filters.py`, acrescente de volta os
caracteres `# * + = < > _ \ | ~ ^` ao conjunto `PONTUACAO_NORMAL` — era assim
que o código estava numa versão anterior.

- (a) Qual passa a ser o valor de `prop_simbolos` para `ocr_ruim.txt`?
- (b) O documento continua sendo rejeitado? Por qual motivo agora?
- (c) Formule o princípio geral: qual é o problema de uma métrica que dá o mesmo
  valor para todos os documentos?

**2.7 Normalizar pela unidade errada.** Em `filters.medir()`, troque o cálculo
de `prop_boilerplate` de caracteres de volta para **linhas**
(`n_linhas_boilerplate / n_linhas`).

- (a) Quais páginas do `relatorio_tecnico.pdf` passam a ser rejeitadas?
- (b) Explique por que isso acontece **depois** de o rejuntar de parágrafos
  transformar uma página inteira em 3 ou 4 "linhas".
- (c) Este é um bug de lógica ou de escolha de denominador? Qual é a lição
  geral sobre proporções?

---

## Nível 3 — Implementação

**3.1 Colunas declaradas em vez de adivinhadas.** `detectar_colunas_textuais`
usa uma heurística (comprimento médio e proporção numérica). Em produção isso é
frágil: uma coluna "Observação" quase sempre vazia seria ignorada.

Implemente um arquivo `colunas.json` que declare, por planilha e por aba, quais
colunas são textuais — e faça o leitor usá-lo quando existir, caindo na
heurística quando não existir. Justifique por que a declaração explícita ganha
da heurística quando ela está disponível.

**3.2 Confiar no conteúdo, não na extensão.** `ler_documento` despacha pela
extensão, e a extensão mente com frequência (um `.txt` que é HTML, um `.xlsx`
que é um CSV renomeado).

Implemente `detectar_formato(caminho)` que leia os primeiros bytes e reconheça a
**assinatura mágica**: `%PDF` para PDF, `PK\x03\x04` para os formatos ZIP (DOCX
e XLSX — como distingui-los?), e ausência de assinatura para texto puro. Faça o
pipeline avisar quando extensão e conteúdo divergirem.

**3.3 Hifenização com dicionário.** `juntar_hifenizacao` transforma
`guarda-\nchuva` em `guardachuva` — um erro raro, mas real.

Implemente uma verificação: antes de remover o hífen, consulte uma lista de
palavras do português; se a forma **com** hífen existir no léxico e a forma
colada não, preserve o hífen. Meça em quantos casos do acervo isso muda o
resultado. Vale a complexidade adicional?

**3.4 Detecção de idioma de verdade.** Substitua `f_outro_idioma` por
`langdetect` ou pelo modelo `lid.176` do fastText.

- Compare, documento a documento, a decisão do detector real com a da heurística
  de stopwords.
- Onde elas discordam? Qual está certa?
- Meça o custo: quantos milissegundos por documento cada uma gasta? Em um corpus
  de 1 milhão de documentos, quanto isso representa?

**3.5 OCR para as páginas cegas.** Faça o pipeline, ao encontrar um trecho de
PDF com motivo `vazio`, tentar OCR automaticamente (via `ocrmypdf` ou
`pytesseract`) e reingerir o resultado.

Registre no relatório quantos trechos foram recuperados por OCR e com que
qualidade — passe o texto recuperado pelos mesmos filtros e mostre quantos
sobrevivem. **Cuidado com a conclusão fácil:** OCR ruim produz exatamente o
`ocr_ruim.txt` da aula. Recuperar texto não é o mesmo que recuperar informação.

**3.6 Limiares por formato.** Como o exercício 2.1 mostrou, um limiar único é
errado por construção. Implemente limiares por formato em `config.py` (por
exemplo, `min_palavras` de 40 para PDF e 20 para planilha), com um valor padrão
para os formatos não declarados.

Rode e compare com a linha de base: o corpus melhorou? Como você **defende** que
melhorou, se o número de documentos aumentou?

**3.7 Ingestão incremental.** Hoje o pipeline reprocessa tudo a cada execução.
Implemente um cache: guarde o hash do conteúdo de cada arquivo e pule os que não
mudaram desde a última execução.

Atenção ao ponto sutil: se `config.py` mudar, **todo o cache precisa ser
invalidado**, porque o resultado depende dos limiares. Como você detecta isso?

---

## Nível 4 — Desafio final (grupo de 2 alunos — prazo definido pelo professor)

Escolha **um** dos caminhos.

**Caminho A — Corpus real de um domínio.**
Monte um corpus a partir de documentos públicos reais (relatórios de agência
reguladora, atas de câmara municipal, editais, artigos científicos abertos —
mínimo de 50 arquivos, com pelo menos três formatos diferentes). Calibre os
limiares para o seu domínio, documentando cada mudança em relação ao padrão e a
evidência que a motivou. Entregue o `metadados_corpus.json` como ficha técnica.
Documente também o que você **não** conseguiu processar e por quê.

**Caminho B — Recuperação de acervo difícil.**
Trabalhe com um acervo dominado por PDFs escaneados e documentos legados.
Integre OCR (3.5), detecção real de idioma (3.4) e detecção de formato por
assinatura (3.2). Meça a taxa de recuperação em cada etapa e mostre, com
exemplos lado a lado, onde o OCR ajudou e onde ele produziu ruído pior que a
ausência de texto.

**Caminho C — Serviço completo.**
Estenda a API com: (i) endpoint de lote (`POST /extrair_lote`) aceitando vários
arquivos; (ii) persistência dos resultados em JSONL no servidor, com
deduplicação **entre requisições** — o que exige manter o índice LSH vivo entre
chamadas; (iii) endpoint `/metricas` com contagem de requisições, latência média
e distribuição dos motivos de rejeição; (iv) imagem Docker funcionando; (v) o
teste de paridade de `testar_api.py` continuando a passar.

**Entregáveis (todos os caminhos):**

1. Código no repositório, organizado e comentado.
2. Relatório de 3 a 5 páginas: o acervo, o pipeline, a **tabela de calibração**
   (limiar testado × efeito medido), o funil de aproveitamento, a análise dos
   descartes e a ficha técnica do corpus.
3. **Uma seção obrigatória: "o que eu joguei fora indevidamente".** Analise
   pelo menos 20 documentos rejeitados e classifique-os em descarte correto e
   descarte indevido. Relate a proporção honestamente.
4. Demonstração de 5 minutos com o serviço rodando ao vivo.

---

## Rubrica de avaliação (100 pontos)

| Critério | Pts | Excelente | Suficiente | Insuficiente |
|---|---|---|---|---|
| **Aquisição** | 15 | Todos os formatos tratados com a sua patologia específica; procedência preservada; falhas de leitura registradas sem derrubar o processo | Formatos lidos, procedência parcial | Extensão confundida com formato; falhas silenciosas; PDF escaneado não percebido |
| **Limpeza** | 15 | Ordem das operações justificada; normalização conservadora defendida; cabeçalho/rodapé tratados | Limpeza funcional, escolhas não justificadas | Limpeza destrutiva (minúsculas, sem acento) sem justificativa; ordem quebrada |
| **Filtragem e calibração** | 20 | Limiares calibrados com **evidência medida**; distribuição das métricas analisada; margem de segurança discutida | Limiares alterados com justificativa superficial | Limiares padrão sem análise, ou ajustados até "o número ficar bonito" |
| **Deduplicação** | 15 | Exata e aproximada implementadas e comparadas; sensibilidade do limiar medida; política de sobrevivência explícita | Deduplicação funciona, sem análise de sensibilidade | Só hash exato, ou nenhuma; duplicatas óbvias no corpus final |
| **Auditoria e honestidade** | 20 | Descartes lidos e classificados; erros do próprio pipeline relatados; ficha técnica completa | Descartes contabilizados, pouco analisados | Nenhuma auditoria; corpus apresentado como se fosse obviamente correto |
| **Deploy** | 10 | Serviço funcionando, paridade com o lote verificada por teste, health check | API funciona, sem teste de paridade | Não roda, ou regras duplicadas entre serviço e lote |
| **Comunicação** | 5 | Relatório claro, tabelas de experimento, conclusões honestas | Completo mas confuso | Incompleto ou sem evidências |

**Observação sobre honestidade experimental.** Nesta aula, relatar que o seu
filtro descartou 8 documentos bons em 20 analisados — com a explicação do porquê
— vale **mais** do que apresentar um corpus impecável sem auditoria nenhuma. Um
pipeline de dados não é avaliado pelo número que ele produz, e sim pela
capacidade de quem o construiu de responder à pergunta: *"o que ficou de fora, e
por quê?"*

Apresentar uma taxa de aproveitamento alta obtida afrouxando limiares até tudo
passar zera o critério "Filtragem e calibração".
