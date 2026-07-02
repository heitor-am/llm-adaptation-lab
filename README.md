# LLM Adaptation Lab

Seis técnicas de adaptação e avaliação de LLMs aplicadas sobre o mesmo modelo (Qwen3-4B) e o mesmo
domínio (Diário Oficial dos Municípios do Piauí), todas executadas em Google Colab com GPU T4 grátis.

Trabalho final da disciplina DC/CCN072, Tópicos em Inteligência Artificial (UFPI, 2026.1), do
professor Raimundo Santos Moura. Grupo 2: Francisco Cosme Monteiro Xavier, Heitor Andrade Moura,
Isaac Augusto Santana Brito e João Pedro Monteiro da Silva Barros.

Documentos de entrega: [RELATORIO.pdf](docs/RELATORIO.pdf) (relatório consolidado) e
[07_final_eval.ipynb](notebooks/07_final_eval.ipynb) (avaliação comparativa com análise crítica por
gráfico). Enunciado em [trabalho-final.pdf](docs/trabalho-final.pdf).

## Resultados-chave (rodada oficial, execução real na T4)

| Questão | Técnica | Métrica principal | Antes | Depois |
|---|---|---|---|---|
| Q1 | Pré-treino contínuo (LoRA) | Perplexidade no domínio | 7,05 | 3,77 (queda de 47%) |
| Q2/Q3 | SFT com QLoRA (1,48% dos parâmetros) | Atribuição de autoria (geração livre) | 0,00 | 0,56 (acaso 0,05; baseline 0,24) |
| Q2/Q3 | SFT com QLoRA | Perplexidade da resposta correta | 2796 | 7,05 |
| Q4 | Destilação 4B para 0.6B | MCQ conceitual do aluno | 0,659 | 0,659 (cerca de 90% do professor, a 1/7 do tamanho) |
| Q5 | RAG (e5 + FAISS) | Acerto factual nas abertas | 0,000 | 0,433 |
| Q5 | RAG | Nota do LLM-juiz (0 a 5) | 1,83 | 3,30 |
| Q6 | Guardrails (filtros + máscara de PII) | Proteção geral (41 casos) | - | 0,951 |

Tese central: cada técnica resolve uma coisa diferente. Pré-treino dá fluência no domínio sem
esquecer o resto; SFT ensina uma tarefa específica; destilação comprime competência; fatos
verificáveis vêm do RAG; segurança vem de camadas externas. O modelo cru tirou 0/30 no acerto
factual, prova de que os fatos locais não estavam decorados no pré-treino (ausência de contaminação).

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `notebooks/` | Os 8 notebooks executados na T4, com outputs preservados como prova de execução |
| `src/` | Fonte `.py` de cada notebook + conversor `nbgen.py` (edita-se o `.py` e regenera-se o `.ipynb`) |
| `results/` | Saídas da execução: `results_q*.json` por questão, `results_final.json` (consolidado), respostas por item para verificação humana (`respostas_*.json`, `baseline.json`) e figuras `fig_*.png` |
| `docs/` | Relatório final em PDF e enunciado do trabalho |
| raiz | Benchmarks próprios (`benchmark_*.json`), lidos pelos notebooks por caminho relativo |

## Notebooks

| Notebook | Papel | Questão |
|---|---|---|
| `01_eval_baseline` | Fundação: harness de avaliação (perplexidade, similaridade, acerto factual, MCQ por log-prob) e o "antes" comum | todas |
| `02_rag` | RAG: embeddings multilingual-e5-small + índice FAISS, comparação sem/com recuperação | Q5 |
| `03_qlora_sft` | SFT com QLoRA 4-bit via TRL: atribuição de autoria no docentesDC | Q2 e Q3 |
| `04_continued_pretrain` | Pré-treino contínuo com LoRA (previsão do próximo token, packing de 1024 tokens) | Q1 |
| `05_distillation` | Destilação black-box: professor 4B gera dados sintéticos, aluno 0.6B faz SFT | Q4 |
| `06_guardrails` | Moderação multi-camada: bloqueio, detecção de injeção, escopo e máscara de PII | Q6 |
| `07_final_eval` | Avaliação comparativa dos 6 modelos com IC 95% (Wilson), LLM-juiz, sonda de esquecimento e análise crítica por gráfico | todas |
| `RELATORIO` | Consolidação executiva dos sete notebooks | todas |

## Como rodar (Google Colab)

1. Abra o notebook no Colab e selecione Runtime, depois GPU T4.
2. Ajuste o seletor `MODE` no topo: `"smoke"` valida o pipeline em Qwen3-1.7B; `"oficial"` roda a
   versão de entrega em Qwen3-4B (4-bit).
3. Execute tudo. Os notebooks que treinam pedem login no Hugging Face (token write) e publicam o
   adapter com `push_to_hub`.

Reaproveitamento: os notebooks 03, 04 e 05 têm um interruptor `RETRAIN`. Com `RETRAIN=False`
(padrão), se o adapter já existe no Hugging Face eles carregam o treino persistido e pulam direto
para a avaliação. O `07_final_eval` não treina nada: carrega todos os modelos publicados, avalia e
gera os gráficos, tolerando adapters ausentes.

Ordem sugerida: `01` (mede o antes), depois `02` a `06` (técnicas) e por fim `07` (consolida).

## Modelos publicados (Hugging Face)

| Adapter | Técnica |
|---|---|
| [heitor-am/qwen3-1.7b-dompi-pretrain](https://huggingface.co/heitor-am/qwen3-1.7b-dompi-pretrain) | Pré-treino contínuo (Q1) |
| [heitor-am/qwen3-4b-dompi-sft](https://huggingface.co/heitor-am/qwen3-4b-dompi-sft) | SFT com QLoRA (Q2/Q3) |
| [heitor-am/qwen3-0.6b-dompi-distill](https://huggingface.co/heitor-am/qwen3-0.6b-dompi-distill) | Aluno destilado (Q4) |

Para carregar: `PeftModel.from_pretrained(base, "heitor-am/qwen3-4b-dompi-sft")`.

## Dados e benchmarks

| Dataset | Uso |
|---|---|
| [gutoportelaa/dom-pi-corpus-2025](https://huggingface.co/datasets/gutoportelaa/dom-pi-corpus-2025) (config `curated`) | Corpus DOM-PI para pré-treino (Q1) e RAG (Q5) |
| [vickminari/docentesDC](https://huggingface.co/datasets/vickminari/docentesDC) | 13.762 documentos rotulados por professor, base do SFT (Q2/Q3) |

Benchmarks próprios, com distratores reais extraídos de outros atos do DOM-PI:

- `benchmark_dompi.json`: 30 perguntas abertas verificáveis no corpus.
- `benchmark_dompi_mcq.json`: 20 questões de múltipla escolha (acaso igual a 25%).
- `benchmark_distill_mcq.json`: 100 questões para a destilação (cerca de 80 conceituais e 20 factuais).

## Controles de rigor

- Q1: os documentos-fonte do benchmark são excluídos do corpus de treino.
- Q2/Q3: deduplicação de quase-duplicatas entre treino e avaliação; baseline de classe majoritária
  reportado; avaliação por geração livre com inspeção item-a-item persistida (`respostas_q3.json`).
- Q4: treino e avaliação em metades disjuntas do benchmark conceitual.
- Q5: avaliação declarada open-book (o índice garante o documento-fonte de cada pergunta).
- Estatística: acurácias acompanham intervalo de confiança 95% (Wilson); a similaridade semântica é
  tratada como métrica secundária por saturar (resposta errada no mesmo tema pontua alto).

## Reproduzir os notebooks a partir do fonte

```bash
cd notebooks
python ../src/nbgen.py ../src/02_rag.py        # gera 02_rag.ipynb
python ../src/nbgen.py ../src/relatorio.py     # gera RELATORIO.ipynb
```

Convenção de células no `.py`: `# %% [markdown]` para células de texto e `# %% [Cell N] Título`
para células de código.
