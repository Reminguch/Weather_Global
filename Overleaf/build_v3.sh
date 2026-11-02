#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$paper_dir/build/main_v3"
mkdir -p "$build_dir"
cd "$paper_dir"
export TEXMFHOME="$paper_dir/texmf"

pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main_v3.tex > "$build_dir/pass1.stdout" 2>&1
(
    cd "$build_dir"
    BIBINPUTS="$paper_dir:" BSTINPUTS="$paper_dir:" bibtex main_v3 > bibtex.stdout 2>&1
)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main_v3.tex > "$build_dir/pass2.stdout" 2>&1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main_v3.tex > "$build_dir/pass3.stdout" 2>&1
cp "$build_dir/main_v3.pdf" "$paper_dir/main_v3.pdf"
printf 'Built %s\n' "$paper_dir/main_v3.pdf"
