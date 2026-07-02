#!/usr/bin/env python3
"""Conversor simples .py -> .ipynb seguindo a convenção de células do projeto.

Marcadores reconhecidos (uma célula começa em cada marcador):
  # %% [markdown]            -> célula markdown (linhas seguintes são "des-comentadas")
  # %% [Cell N] Título       -> célula markdown "### Título" + célula de código
  # %% [Cell] Título         -> idem (N opcional)

Em células de código, linhas do tipo "# !pip ..." ou "# %magic" viram "!pip ..." / "%magic"
(comando de shell/magic comentado no .py vira executável no notebook).

Uso (a partir da pasta do projeto, com a fonte em src/):
  python src/nbgen.py src/02_rag.py            -> gera 02_rag.ipynb no diretório atual
  python src/nbgen.py src/relatorio.py         -> gera RELATORIO.ipynb (caso especial)
  python src/nbgen.py src/02_rag.py saida.ipynb -> caminho de saída explícito
"""
import json
import os
import re
import sys

MARK = re.compile(r"^# %% \[(markdown|Cell[^\]]*)\](.*)$")
MAGIC = re.compile(r"^#\s+([!%].*)$")


def _decomment_md(lines):
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln == "#":
            out.append("")
        else:
            out.append(ln)
    # remove linhas em branco nas pontas
    while out and out[0].strip() == "":
        out.pop(0)
    while out and out[-1].strip() == "":
        out.pop()
    return out


def _clean_code(lines):
    out = []
    for ln in lines:
        m = MAGIC.match(ln)
        out.append(m.group(1) if m else ln)
    while out and out[0].strip() == "":
        out.pop(0)
    while out and out[-1].strip() == "":
        out.pop()
    return out


def _src(lines):
    """Converte lista de linhas no formato 'source' do nbformat (\\n no fim, menos a última)."""
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


def convert(py_path, ipynb_path):
    with open(py_path, encoding="utf-8") as f:
        raw = f.read().splitlines()

    cells = []
    i = 0
    n = len(raw)
    while i < n:
        m = MARK.match(raw[i])
        if not m:
            i += 1
            continue
        kind, rest = m.group(1), m.group(2).strip()
        i += 1
        body = []
        while i < n and not MARK.match(raw[i]):
            body.append(raw[i])
            i += 1

        if kind == "markdown":
            md = _decomment_md(body)
            if md:
                cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(md)})
        else:  # Cell N
            title = rest
            if title:
                cells.append({"cell_type": "markdown", "metadata": {},
                              "source": _src(["### " + title])})
            code = _clean_code(body)
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": _src(code)})

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    # valida
    with open(ipynb_path, encoding="utf-8") as f:
        json.load(f)
    print(f"OK: {ipynb_path} ({len(cells)} células)")


if __name__ == "__main__":
    src = sys.argv[1]
    if len(sys.argv) > 2:
        out = sys.argv[2]
    else:
        stem = os.path.basename(src).rsplit(".", 1)[0]        # grava no CWD, não ao lado da fonte
        out = "RELATORIO.ipynb" if stem == "relatorio" else stem + ".ipynb"
    convert(src, out)
