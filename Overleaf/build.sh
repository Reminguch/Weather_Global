#!/usr/bin/env bash
set -euo pipefail

# Include the extra LaTeX packages and fonts missing from Della's TeX Live.
paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$paper_dir"
export TEXMFHOME="$paper_dir/texmf"

pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
bibtex main_v2
pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
pdflatex -interaction=nonstopmode -halt-on-error main_v2.tex
