# %% [markdown]
# # Relatório do Trabalho Final de Tópicos em IA
#
# - **Disciplina:** DC/CCN072, Tópicos em Inteligência Artificial (UFPI, 2026.1)
# - **Professor:** Raimundo Santos Moura
# - **Grupo 2:** Francisco Cosme Monteiro Xavier, Heitor Andrade Moura, Isaac Augusto Santana Brito e João Pedro Monteiro da Silva Barros
# - **Livro base:** *Build a Large Language Model (From Scratch)* (Sebastian Raschka, 2024)
#
# Este notebook consolida os **sete notebooks** do trabalho final: a fundação de avaliação (`01`),
# as seis questões (`02` a `06`), com objetivo, método e resultado (antes/depois) de cada uma, e a
# avaliação comparativa final (`07`). A análise crítica aprofundada, gráfico a gráfico, está em
# [07_final_eval.ipynb](07_final_eval.ipynb).
#
# ## Objetivo do trabalho
#
# - Aplicar, sobre um mesmo modelo e um mesmo domínio, **seis técnicas distintas** de adaptação e
#   avaliação de LLMs vistas na disciplina (pré-treino contínuo, SFT/LoRA, destilação, RAG, guardrails).
# - Medir cada técnica com **métricas próprias** e mostrar, com números, **o que cada uma
#   resolve**, em vez de tratar "melhorar o modelo" como um objetivo único.
#
# ## Configuração comum
#
# - **Modelo:** Qwen3-4B na rodada oficial (professor da destilação; aluno Qwen3-0.6B; sonda de
#   esquecimento em Qwen3-1.7B). Troca de `MODEL_ID` numa linha.
# - **Dados:** **DOM-PI**, o Diário Oficial dos Municípios do Piauí (`gutoportelaa/dom-pi-corpus-2025`)
#   e **docentesDC** (`vickminari/docentesDC`, 13.762 docs rotulados por professor), ambos no Hugging Face.
# - **Infra:** Google Colab com **GPU T4** grátis (QLoRA 4-bit para caber em 16 GB; treino em float32
#   por compatibilidade do GradScaler na T4).
# - **Benchmarks próprios:** 30 perguntas abertas + 20 MCQ (DOM-PI) e 100 MCQ (destilação: ~80
#   conceituais + 20 factuais). Distratores das MCQ são valores/nomes reais de outros atos do DOM-PI.

# %% [markdown]
# ## Baseline (fundação comum: Qwen3-4B cru)
#
# Medido em [01_eval_baseline.ipynb](01_eval_baseline.ipynb) antes de qualquer técnica:
#
# | Métrica | Valor | Leitura |
# |---|---|---|
# | Perplexidade no domínio | 10,33 | ponto de partida da adaptação (Q1) |
# | Similaridade nas abertas | 0,831 | alta mesmo errando: **satura** (ver Limitações) |
# | **Acerto factual nas abertas** | **0/30** | o modelo **não tem os fatos locais**, prova de ausência de contaminação |
#
# > O acerto factual zero do modelo cru é a evidência-chave do trabalho: todo ganho factual mostrado
# > adiante vem das técnicas (RAG, SFT), não de conhecimento decorado no pré-treino do Qwen.

# %% [markdown]
# ## Mapa dos notebooks (1 a 7) por questão
#
# | Notebook | Papel | Questão | Status |
# |---|---|---|---|
# | [01_eval_baseline.ipynb](01_eval_baseline.ipynb) | Fundação: harness de avaliação + o "antes" comum | todas | executado (4B) |
# | [02_rag.ipynb](02_rag.ipynb) | RAG: recuperação + geração (e5 + FAISS) | **Q5** | executado (4B) |
# | [03_qlora_sft.ipynb](03_qlora_sft.ipynb) | SFT com QLoRA 4-bit via TRL (atribuição de autoria) | **Q2 e Q3** | executado (4B) |
# | [04_continued_pretrain.ipynb](04_continued_pretrain.ipynb) | Pré-treino contínuo com LoRA (próximo token) | **Q1** | executado (4B) |
# | [05_distillation.ipynb](05_distillation.ipynb) | Destilação: professor 4B para aluno 0.6B (dados sintéticos) | **Q4** | executado |
# | [06_guardrails.ipynb](06_guardrails.ipynb) | Guardrails: filtros + máscara de PII (sem treino) | **Q6** | executado |
# | [07_final_eval.ipynb](07_final_eval.ipynb) | Avaliação comparativa final: IC 95%, LLM-juiz, sonda de esquecimento | todas | executado |
#
# > As seções a seguir percorrem os resultados por questão (Q1 a Q6, cada uma no seu notebook) e fecham
# > com a **avaliação comparativa do `07`**, que cruza todos os modelos nos mesmos benchmarks.

# %% [markdown]
# ## Q1: Pré-treino contínuo (DOM-PI)
#
# - **Objetivo:** adaptar o modelo ao domínio dos Diários Oficiais e medir o ganho de fluência.
# - **Método:** treino contínuo (previsão do próximo token) com LoRA sobre o corpus DOM-PI, em blocos
#   de 1024 tokens (*packing*); métricas ANTES e DEPOIS em conjunto *held-out* (docs-fonte do benchmark
#   **excluídos** do treino, como controle de contaminação).
# - **Notebook:** [04_continued_pretrain.ipynb](04_continued_pretrain.ipynb).
#
# **Resultado (rodada oficial, Qwen3-4B, 800 docs):**
#
# | Métrica | Antes | Depois | Leitura |
# |---|---|---|---|
# | Perplexidade (held-out) | 7,05 | **3,77** (queda de 47%) | adaptação forte ao domínio |
# | Acurácia de token | 0,612 | **0,695** | idem |
# | Acurácia MCQ | 0,333 | 0,500 | fluência ajuda um pouco, mas fato local continua sendo papel do RAG |
#
# - **Sonda de esquecimento** (no `07`, em 1.7B): a perplexidade **fora do domínio não sobe** após o
#   pré-treino: adaptação **sem esquecimento catastrófico**.

# %% [markdown]
# ## Q2 / Q3: Fine-tuning supervisionado com QLoRA
#
# - **Objetivo:** ensinar o modelo a responder no formato instrução-resposta sobre o `docentesDC`.
#   Como o dataset é um corpus (documento + `nome_professor`), a tarefa é **atribuição de autoria**
#   (do documento para o professor; 19 professores distintos no conjunto usado).
# - **Método:** SFT com **QLoRA 4-bit** via TRL: apenas **1,48%** dos parâmetros treinados (33M de
#   2,24B), o que permite ajustar um 4B numa T4 grátis. 1000 pares de treino com **deduplicação**
#   treino/avaliação (controle de contaminação); held-out de 25 documentos.
# - **Q2 e Q3:** mesmo pipeline; o Q3 destaca o eixo LoRA/QLoRA (memória versus qualidade).
# - **Notebook:** [03_qlora_sft.ipynb](03_qlora_sft.ipynb).
#
# **Resultado (rodada oficial, Qwen3-4B):**
#
# | Métrica | Antes | Depois | Referências |
# |---|---|---|---|
# | **Atribuição (geração livre)** | 0,00 | **0,56** | acaso 1/19 = 0,053; baseline de classe 0,24 |
# | Perplexidade da resposta correta (no `07`) | 2796 | **7,05** | queda de cerca de 400 vezes: o SFT "gravou" a tarefa |
# | Cloze por loss média (secundária) | 0,04 | 0,04 | **subestima**: viés de comprimento em nomes compostos |
#
# - **Lição metodológica (resultado em si):** a primeira avaliação (cloze por loss média) dava ~0,04 e
#   sugeria "tarefa não-aprendível". Era **artefato da métrica**. A geração livre + match normalizado,
#   confirmada por inspeção item-a-item (`respostas_q3.json`), mostra o aprendizado real: **a escolha
#   da métrica pode inverter a conclusão**.

# %% [markdown]
# ## Q4: Destilação de conhecimento
#
# - **Objetivo:** transferir a capacidade de um modelo grande (professor) para um menor (aluno) via
#   dados sintéticos (destilação *black-box*).
# - **Método:** o professor **Qwen3-4B** responde prompts conceituais; o aluno **Qwen3-0.6B** faz SFT
#   nessas respostas; por fim compara-se aluno antes/depois e contra o professor. Treino e avaliação usam
#   **metades disjuntas** do benchmark conceitual (controle de vazamento); 20 factuais como controle.
# - **Notebook:** [05_distillation.ipynb](05_distillation.ipynb).
#
# **Resultado (rodada oficial):**
#
# | MCQ | Professor 4B | Aluno 0.6B antes | Aluno 0.6B depois |
# |---|---|---|---|
# | Conceitual (disjunto) | 0,732 | 0,659 | 0,659 |
# | Factual (controle) | 0,250 | 0,200 | 0,300 |
#
# - **Leitura:** o aluno de 0.6B já captura **~90% da competência conceitual** do professor de 4B a
#   **1/7 do tamanho**: o valor da destilação aqui é *compressão*. O factual fica em torno do acaso
#   (0,25) para todos: destilação transfere **como responder**, não **fato local** (papel do RAG, Q5).

# %% [markdown]
# ## Q5: RAG (Retrieval-Augmented Generation)
#
# - **Objetivo:** melhorar respostas factuais fornecendo ao modelo trechos recuperados dos Diários.
# - **Método:** *chunk* do corpus (619 docs: fontes do benchmark + distratores), embeddings
#   (`multilingual-e5-small`), índice **FAISS**, recuperação top-4 e geração com o contexto. Avaliação
#   declarada **open-book** (o índice contém os docs-fonte por construção).
# - **Notebook:** [02_rag.ipynb](02_rag.ipynb); avaliação estendida no `07` (benchmark MCQ maior + LLM-juiz).
#
# **Resultado (rodada oficial, Qwen3-4B, 30 abertas):**
#
# | Métrica | Sem-RAG | Com-RAG | Leitura |
# |---|---|---|---|
# | **Acerto factual (abertas)** | **0,000** | **0,433** | o ganho real: fato vem da recuperação |
# | Acurácia MCQ (8 itens, nb 02) | 0,250 (= acaso) | **0,875** | idem |
# | Acurácia MCQ (20 itens, `07`) | 0,250 | **0,500** | benchmark maior/mais difícil (conjuntos distintos) |
# | Nota do LLM-juiz de 0 a 5 (`07`) | 1,83 | **3,30** | terceira métrica independente concordando |
# | Similaridade (e5) | 0,831 | 0,856 | quase não move: **satura** (métrica secundária) |
#
# - A distribuição das notas do juiz é **bimodal com RAG** (0 ou 5): quando o *retriever* acha o
#   documento certo a resposta é perfeita; quando falha, zera. O gargalo passa a ser o **recall da
#   recuperação**, não o modelo.

# %% [markdown]
# ## Q6: Guardrails (segurança e confiabilidade)
#
# - **Objetivo:** camada de controle que bloqueia pedidos perigosos, detecta injeção de prompt, recusa
#   fora de escopo e mascara PII, **sem treinar** o modelo (moderação multi-camada: entrada, filtros, LLM e máscara na saída, nessa ordem).
# - **Notebook:** [06_guardrails.ipynb](06_guardrails.ipynb).
#
# **Resultado (41 casos de teste):**
#
# | Filtro | Acertos | Taxa |
# |---|---|---|
# | Bloqueio de conteúdo perigoso | 18/20 | 0,90 |
# | Detecção de injeção/jailbreak | 7/7 | 1,00 |
# | Escopo (on-topic e fora) | 8/8 | 1,00 |
# | Máscara de PII (CPF, e-mail, telefone) | 6/6 | 1,00 |
# | **Proteção geral** | **39/41** | **0,951** |
#
# - Os 2 casos que passaram são frases perigosas **parafraseadas** sem as palavras-chave dos filtros,
#   limite conhecido de regras/keywords (ver Limitações).

# %% [markdown]
# ## Avaliação comparativa final ([07_final_eval.ipynb](07_final_eval.ipynb))
#
# O `07` cruza **os 6 modelos** (bases + adapters publicados no HF) nos mesmos benchmarks, com
# **IC 95% de Wilson**, e adiciona as análises transversais: sonda de esquecimento, perplexidade da
# resposta, LLM-juiz e a crítica das métricas. É o notebook-relatório avaliado, com um gráfico por
# célula e análise crítica de cada um.
#
# **MCQ conceitual por modelo (IC 95%; acaso = 0,25):**
#
# | Modelo | Conceitual | IC 95% |
# |---|---|---|
# | 0.6B base | 0,659 | [0,51 a 0,78] |
# | 0.6B + destilação | 0,683 | [0,53 a 0,80] |
# | 1.7B base | 0,683 | [0,53 a 0,80] |
# | 1.7B + pré-treino | 0,732 | [0,58 a 0,84] |
# | 4B base | 0,732 | [0,58 a 0,84] |
# | 4B + SFT | 0,610 | [0,46 a 0,74] |
#
# **Leituras principais:**
#
# - Todos **bem acima do acaso**; os **IC se sobrepõem**: diferenças pequenas (ex.: 4B+SFT abaixo do
#   4B base, sugerindo custo de especialização) são **tendência, não fato estatístico** com este *n*.
# - **Sonda de esquecimento (Q1):** do 1.7B base para o 1.7B + pré-treino, a perplexidade no domínio vai de 10,46 para 8,64 e
#   **fora do domínio de 7,94 para 6,64** (não sobe): adaptação sem esquecimento.
# - **Perplexidade da resposta (Q2/Q3):** de 2796 para 7,05 após o SFT, o sinal mais forte do trabalho.
# - **LLM-juiz (Q5):** nota média de 1,83 para 3,30 com RAG; distribuição **bimodal** com RAG (0 ou 5)
#   localiza o gargalo no *recall* do retriever.
# - **Crítica de métrica:** o gráfico de saturação mostra respostas factualmente erradas com
#   similaridade ~0,85, o que justifica a escolha de factual/MCQ/juiz como métricas principais.

# %% [markdown]
# ## Tabela consolidada de resultados (rodada oficial)
#
# | Q | Técnica | Métrica principal | Antes | Depois |
# |---|---|---|---|---|
# | Q1 | Pré-treino contínuo | Perplexidade no domínio | 7,05 | **3,77** (queda de 47%) |
# | Q2/Q3 | SFT (QLoRA, 1,48% params) | Atribuição de autoria (geração livre) | 0,00 | **0,56** (acaso 0,05; baseline 0,24) |
# | Q2/Q3 | SFT (QLoRA) | Perplexidade da resposta correta | 2796 | **7,05** |
# | Q4 | Destilação 4B para 0.6B | MCQ conceitual do aluno | 0,659 | 0,659 (cerca de 90% do professor 0,732) |
# | Q5 | RAG | Acerto factual (abertas) | 0,000 | **0,433** |
# | Q5 | RAG | Nota do LLM-juiz (0 a 5) | 1,83 | **3,30** |
# | Q6 | Guardrails | Proteção geral (41 casos) | - | **0,951** |
#
# > Todos os números são de execução real na T4 (notebooks com outputs preservados em `notebooks/`;
# > JSONs por questão em `results/`). Acurácias no `07` acompanham **IC 95% de Wilson**.

# %% [Cell 0] Setup: montar o Google Drive e entrar na pasta (persiste os resultados)
# Necessário p/ ler os results_qN.json do Drive. Fora do Colab, pule (a célula seguinte também procura
# em results/).
from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/MyDrive/ufpi/topics-in-ai/final-project

# %% [Cell 1] Tabela consolidada automática (lê os results_qN.json, se presentes)
# Procura na pasta atual e em results/ (estrutura do repositório). O que faltar aparece como "pendente".
import json, os

def _load(q):
    for p in (f"results_{q}.json", f"results/results_{q}.json", f"../results/results_{q}.json"):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None

linhas = []   # (Q, técnica, métrica, antes, depois)

r = _load("q1")
if r:
    linhas += [("Q1", "Pré-treino", "Perplexidade (held-out)", r.get("ppl_antes"), r.get("ppl_depois")),
               ("Q1", "Pré-treino", "Acurácia MCQ", r.get("mcq_antes"), r.get("mcq_depois"))]
else:
    linhas.append(("Q1", "Pré-treino", "Perplexidade / MCQ", "pendente", "pendente"))

r = _load("q3")
if r:
    linhas.append(("Q2/Q3", "SFT (QLoRA)", "Atribuição (geração livre)", r.get("atrib_antes"), r.get("atrib_depois")))
    if r.get("cloze_antes") is not None:
        linhas.append(("Q2/Q3", "SFT (QLoRA)", "Cloze (secundária, enviesada)", r.get("cloze_antes"), r.get("cloze_depois")))
else:
    linhas.append(("Q2/Q3", "SFT (QLoRA)", "Atribuição (geração livre)", "pendente", "pendente"))

r = _load("q4")
if r:
    a, d = r.get("aluno_antes", {}), r.get("aluno_depois", {})
    linhas += [("Q4", "Destilação", "MCQ conceitual (aluno)", a.get("conceitual"), d.get("conceitual")),
               ("Q4", "Destilação", "MCQ factual (controle)", a.get("factual"), d.get("factual"))]
else:
    linhas.append(("Q4", "Destilação", "MCQ conceitual / factual", "pendente", "pendente"))

r = _load("q5")
if r:
    linhas += [("Q5", "RAG", "Acerto factual", r.get("fac_sem"), r.get("fac_com")),
               ("Q5", "RAG", "Acurácia MCQ", r.get("mcq_sem"), r.get("mcq_com")),
               ("Q5", "RAG", "Similaridade (satura)", r.get("sim_sem"), r.get("sim_com"))]
else:
    linhas.append(("Q5", "RAG", "Factual / MCQ / Sim", "pendente", "pendente"))

r = _load("q6")
if r:
    linhas += [("Q6", "Guardrails", "Proteção geral", "-", r.get("protecao_geral")),
               ("Q6", "Guardrails", "PII protegida", "-", r.get("pii_taxa"))]
else:
    linhas.append(("Q6", "Guardrails", "Proteção", "-", "pendente"))

def _f(x):
    return "-" if x is None else str(x)
tabela = "| Q | Técnica | Métrica | Antes | Depois |\n|---|---|---|---|---|\n" + \
         "\n".join(f"| {q} | {t} | {m} | {_f(a)} | {_f(d)} |" for q, t, m, a, d in linhas)

achados = [q for q in ("q1", "q3", "q4", "q5", "q6") if _load(q)]
print(f"results_qN.json encontrados: {achados or 'nenhum (rode os notebooks primeiro)'}\n")
try:
    from IPython.display import Markdown, display
    display(Markdown(tabela))
except Exception:
    print(tabela)

# %% [markdown]
# ## Conclusões
#
# **Cada técnica resolve uma coisa diferente.** O ponto central do trabalho não é "qual técnica é a
# melhor", e sim *o que cada uma muda*:
#
# - **Pré-treino contínuo (Q1)** dá **fluência no domínio**: perplexidade com queda de 47% (de 7,05 para 3,77), **sem
#   esquecimento catastrófico** (a perplexidade fora do domínio não sobe).
# - **SFT/QLoRA (Q2/Q3)** ensina uma **tarefa específica**: atribuição de autoria de 0,00 para 0,56
#   treinando só 1,48% dos parâmetros. E deixa a lição metodológica: **a métrica errada (cloze por loss
#   média) invertia a conclusão**; geração livre + inspeção item-a-item é a leitura correta.
# - **Destilação (Q4)** faz **compressão**: o aluno 0.6B retém ~90% da competência conceitual do
#   professor 4B a 1/7 do tamanho. Não transfere fato local (controle no acaso), nem deveria.
# - **RAG (Q5)** resolve **fato pontual**: acerto factual de 0 para 0,43 e MCQ de 0,25 para 0,875 sem tocar
#   nos pesos. Três métricas independentes (factual, MCQ, LLM-juiz) concordam.
# - **Guardrails (Q6)** dão **segurança sem treino**: 95% de proteção com filtros determinísticos.
#
# Em conjunto: **fluência e formato vêm de adaptar o modelo (pré-treino/SFT); fatos verificáveis vêm de
# recuperar o documento certo (RAG); segurança vem de camadas externas (guardrails)**. Otimizar uma
# métrica única esconde essa divisão de trabalho.

# %% [markdown]
# ## Limitações e trabalhos futuros
#
# **Infraestrutura (Colab T4 grátis):**
#
# - Treinos enxutos (poucos passos/época única) e modelos em **4-bit**, única forma de caber um 4B em
#   16 GB. Com mais GPU, mais passos e dados aprofundariam os ganhos.
#
# **Métricas e benchmarks:**
#
# - A **similaridade semântica satura** (resposta errada no mesmo tema pontua ~0,85); por isso as
#   conclusões se apoiam em **acerto factual, MCQ e LLM-juiz**; a similaridade fica como diagnóstico.
# - O **cloze por loss média subestimou** a atribuição (viés de comprimento), corrigido para geração
#   livre + match normalizado, com inspeção item-a-item persistida (`respostas_q3.json`).
# - Benchmarks pequenos (dezenas de itens) resultam em **IC 95% largos** no `07`; diferenças pequenas entre
#   modelos não são estatisticamente significativas e são reportadas como tendência.
# - O **LLM-juiz é da mesma família** do gerador (Qwen3-4B); um juiz externo/maior seria mais isento.
#
# **Por questão:**
#
# - **Q4:** o aluno 0.6B já chega competente no benchmark conceitual (0,659), pouca margem para o
#   ganho da destilação aparecer; um benchmark conceitual mais difícil evidenciaria a transferência.
# - **Q5:** os erros remanescentes com RAG são **falhas de recall do retriever** (distribuição bimodal
#   das notas do juiz); melhorar embedding/top-k/re-ranking é a evolução natural.
# - **Q6:** filtros por regra/keyword são frágeis a paráfrase (2/20 escaparam); em produção, um
#   **classificador** (ex.: Llama Guard) substituiria as regras.
#
# **Trabalhos futuros:** benchmark curado maior; juiz externo (8B+); re-ranking no RAG; classificador
# de segurança no Q6; destilação com benchmark conceitual mais difícil e dados sintéticos via RAG.

# %% [markdown]
# ## Reprodutibilidade
#
# **Como rodar (Google Colab):**
#
# - Abra o notebook da questão no Colab e selecione **Runtime > GPU T4**.
# - Ajuste o seletor `MODE` no topo: `"smoke"` (rápido, valida o pipeline em 1.7B) ou `"oficial"`
#   (resultado de entrega em 4B, 4-bit). É a única linha a trocar entre protótipo e rodada final.
# - **Run all.** Os notebooks que treinam pedem login no Hugging Face (token *write*) e fazem `push_to_hub`.
#
# **Reaproveitamento (evita re-treino):** os notebooks 03/04/05 têm um interruptor `RETRAIN`. Com
# `RETRAIN=False` (padrão), se o adapter já existe no HF eles **carregam** e **pulam o treino**, o que permite
# re-rodar só as células de avaliação/gráficos sem esperar. `RETRAIN=True` treina do zero e publica.
#
# **Avaliação consolidada:** o `07_final_eval` carrega todos os modelos do HF (não treina), avalia com
# intervalos de confiança e gera os gráficos + análise crítica. Ele **tolera adapter ausente** (pula o
# que faltar e segue).
#
# **Ordem sugerida:** `01_eval_baseline` (mede o "antes"), depois as técnicas (`02` a `06`) e por fim `07_final_eval` (consolida).
#
# **Modelos publicados (Hugging Face, usuário `heitor-am`):**
#
# - `heitor-am/qwen3-1.7b-dompi-pretrain`: adapter LoRA do pré-treino contínuo (Q1; sonda 1.7B).
# - `heitor-am/qwen3-4b-dompi-sft`: adapter QLoRA do SFT (Q2/Q3).
# - `heitor-am/qwen3-0.6b-dompi-distill`: aluno destilado (Q4).
# - Carregar de volta: `PeftModel.from_pretrained(base, "heitor-am/qwen3-4b-dompi-sft")`.
#
# **Artefatos no repositório:**
#
# - `notebooks/`: os 8 notebooks **executados** (outputs preservados como prova de execução).
# - `results/`: `results_qN.json` por questão, `results_final.json` (consolidado do `07`),
#   respostas por item para verificação humana (`respostas_*.json`, `baseline.json`) e figuras `fig_*.png`.
# - `src/`: fontes `.py` dos notebooks + conversor `nbgen.py`.
# - `docs/`: resultados consolidados (`RESULTS.md`), resumo de apresentação (`STATUS.md`) e PDFs.
