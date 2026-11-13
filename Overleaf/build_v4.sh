#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-$(mktemp -d /tmp/weather-v4.XXXXXX)}"
mkdir -p "$build_dir"
build_dir="$(cd -- "$build_dir" && pwd)"
if [[ "$build_dir" == "$paper_dir" || "$build_dir" == "$paper_dir/"* ]]; then
    printf 'Choose a build directory outside the Overleaf project.\n' >&2
    exit 1
fi

cd "$paper_dir"
trap 'printf "Build failed; inspect logs in %s\n" "$build_dir" >&2' ERR
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" v4.tex > "$build_dir/pass1.stdout" 2>&1
(
    cd "$build_dir"
    BIBINPUTS="$paper_dir:" BSTINPUTS="$paper_dir:" bibtex v4 > bibtex.stdout 2>&1
)
for pass in 2 3 4; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" v4.tex > "$build_dir/pass${pass}.stdout" 2>&1
done
printf 'Built %s/v4.pdf\n' "$build_dir"
