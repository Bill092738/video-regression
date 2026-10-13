#!/usr/bin/env bash
# Build the paper and package an arXiv-ready source tarball (arxiv.tar.gz) in this folder.
# arXiv compiles the sources itself but does not run BibTeX reliably, so the generated main.bbl is
# included. Figures referenced from ../img/ are copied into img/ inside the bundle.
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error -recorder main.tex
rm -rf arxiv && mkdir -p arxiv/sections arxiv/img
cp main.tex main.bbl arxiv/
cp sections/*.tex arxiv/sections/
# every image file pdflatex actually read, according to the recorder file
grep '^INPUT ../img/' main.fls | sed 's|^INPUT ../img/||' | sort -u | while read -r f; do
  mkdir -p "arxiv/img/$(dirname "$f")"
  cp "../img/$f" "arxiv/img/$f"
done
sed -i 's|\\graphicspath{{../img/}}|\\graphicspath{{img/}}|' arxiv/main.tex
# check that the bundle compiles on its own before uploading
(cd arxiv && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null) \
  || { echo "bundle does not compile; see arxiv/main.log"; exit 1; }
rm -f arxiv/*.aux arxiv/*.log arxiv/*.out arxiv/main.pdf
tar -czf arxiv.tar.gz -C arxiv .
echo "wrote $(pwd)/arxiv.tar.gz"
