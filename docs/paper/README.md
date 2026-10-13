# Paper draft

LaTeX sources for the paper on this project. The draft is kept on its own branch so it can be
edited while experiments run elsewhere. Numbers are copied from `PROGRESS.md`; the experiments
themselves live in the code and scripts of the main repository.

```
main.tex            preamble, title, macros, section order
sections/*.tex      one file per section (abstract, introduction, related, problem, method,
                    setup, results, discussion, statements, appendix)
references.bib      bibliography, sorted by key, one STATUS line per entry
make_arxiv.sh       builds the PDF and packages arxiv.tar.gz for upload
```

Figures are read from `../img/` (the same files the README uses), so nothing is duplicated.

## Build

```bash
cd docs/paper
latexmk -pdf main.tex        # main.pdf; latexmk runs pdflatex and bibtex as often as needed
latexmk -c                   # remove build files (keeps the PDF)
```

You need a TeX distribution with `latexmk` (TeX Live or MiKTeX on Windows, MacTeX on macOS). On
Debian/Ubuntu: `texlive-latex-recommended texlive-latex-extra texlive-science
texlive-fonts-recommended lmodern latexmk`. Build outputs are git-ignored; commit sources only.

**Overleaf:** run `./make_arxiv.sh`, then upload `arxiv.tar.gz` as a new project. The bundle has the
figures inside it. Edits made on Overleaf have to be copied back into this folder by hand.

## Editing conventions

* `\todo{...}` (red): text or a decision that is still missing.
* `\pending{...}` (orange): a result that depends on an experiment that has not run yet (P1 E20–E23,
  P2).
* **Every number in `results.tex` and `appendix.tex` must come from `PROGRESS.md`**, which is
  generated from the run outputs (`scripts/p0_tables.py`, `scripts/p1_tables.py`). If a number
  changes there, change it here. Do not round or "improve" numbers by hand.
* Claims about other papers (`related.tex`) must be checked against the paper itself, not against a
  summary.
* Use `\cref{...}` for references to sections, tables and figures, and `~\cite{...}` for citations.

## References: how to verify

`references.bib` is sorted alphabetically by key (`firstauthorYEARfirstword`). Every entry has a
comment line above it:

```
% STATUS: unverified                 assembled from search results; not yet checked
% STATUS: unverified, CHECK pages    as above, and the named field is known to be uncertain
% STATUS: verified 2026-10-20        you checked it against the primary source
```

**All entries currently start as `unverified`**: they were assembled from web search results, and
arXiv pages could not be opened when they were written. To verify an entry:

1. Open the primary source: the DOI link, the publisher's page or the arXiv abstract page. Do not
   rely on Google Scholar's BibTeX export, which often has wrong venues or truncated author lists.
2. Check the full author list and its order, the exact title, the venue (journal or conference
   name), the year, and the volume and pages for journal articles.
3. For an arXiv preprint that has since been published, cite the published version and keep the
   arXiv number in `note`.
4. Check that the sentence in the paper that cites it says what the cited paper actually says.
5. Edit the entry, change its STATUS line to `verified <date>`, and keep the alphabetical order.

Never put the at-sign in a comment line: BibTeX reads it as the start of an entry. Before
submission, `grep -c "STATUS: unverified" references.bib` should print 0.

## Submitting to arXiv (first time)

**Before you start:**
1. **Create an arXiv account** with your university email; an institutional address helps with
   the next step.
2. **Endorsement.** First-time submitters in a category often need an endorsement from an
   established author in that category. arXiv tells you during submission whether you need one and
   gives you a code to send to an endorser. Ask your course supervisor or a lab member who has
   posted in the category. Do this early: it can take days.
3. **Category.** Primary `eess.IV` (Image and Video Processing); cross-list `cs.MM` (Multimedia),
   and `cs.LG` if the learning part is substantial.
4. **License.** arXiv asks you to pick one. CC BY 4.0 is the common choice for open research. The
   default arXiv non-exclusive license is the most compatible with later journal submission. Check
   the target venue's policy if you plan to submit there too.

**The source package:**
1. In `main.tex` set `\finaltrue`. The build then stops with an error if any `\todo` or `\pending`
   remains.
2. Every STATUS line should say `verified`, and the author block and the statements section should
   be filled in.
3. Run `./make_arxiv.sh`. It builds the PDF, collects exactly the figures used, includes `main.bbl`
   (arXiv does not reliably run BibTeX, so the `.bbl` has to be in the upload), checks that the
   bundle compiles on its own, and writes `arxiv.tar.gz`.
4. Upload `arxiv.tar.gz` in the arXiv submission form. arXiv compiles it again, and **you must look
   at the PDF arXiv produces** before you confirm.

**The metadata form:**
* **Title.** The same as in the PDF.
* **Abstract.** Plain text, at most 1920 characters, no LaTeX commands and no citations. Use the
  abstract from `sections/abstract.tex` without the `\pending` parts.
* **Comments**, for example: `16 pages, 2 figures. Code: <repository URL>`.
* **Authors**: real people only. AI tools cannot be authors. Disclose AI assistance in the paper
  (`statements.tex`) according to arXiv's current policy.

**After submission:**
* Papers are announced on a fixed schedule (Sunday to Thursday evenings, US Eastern time). A
  submission can be replaced by a new version (v2, v3, ...) at any time, but it cannot be deleted
  once announced; old versions stay visible.
* If you also submit to a conference or journal, check its policy on preprints first. Most
  computer-science venues allow arXiv preprints.
* For the course: the arXiv version can be the course deliverable, or the course report can be a
  frozen PDF of an earlier version. Ask the course staff which they prefer.

## Data licenses in the paper

Third-party clips are not redistributed. Figures must use only clips whose license allows
publication: SVT (EBU/SVT conditions), CC BY (Blender) or public domain. **Research-only sources
(NTIA/ITS, HDgreetings) must not appear in figures.** The clip manifest, `docs/p1/clips.csv` on the
experiment branch, records each clip's license.
