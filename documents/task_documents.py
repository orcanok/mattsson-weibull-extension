import subprocess
from pathlib import Path

import pytask

from flexible_moral_hazard.config import DOCUMENTS, ROOT

PAPER_FIGURES: list[Path] = []
GENERATED_RESULTS: Path = DOCUMENTS / "generated" / "results.tex"


@pytask.task(id="latex-paper")
def task_compile_latex_paper(
    tex: Path = DOCUMENTS / "paper.tex",
    results_tex: Path = GENERATED_RESULTS,
    figures: list[Path] = PAPER_FIGURES,
    produces: Path = ROOT / "paper.pdf",
) -> None:
    """Compile LaTeX paper to PDF (with BibTeX)."""
    # 1) pdflatex -> .aux
    subprocess.run(
        ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name),
        check=True,
        cwd=DOCUMENTS,
    )

    # 2) bibtex -> .bbl
    subprocess.run(
        ("bibtex", tex.with_suffix("").name),  # "paper"
        check=True,
        cwd=DOCUMENTS,
    )

    # 3) pdflatex -> include .bbl
    subprocess.run(
        ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name),
        check=True,
        cwd=DOCUMENTS,
    )

    # 4) pdflatex -> resolve refs
    subprocess.run(
        ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name),
        check=True,
        cwd=DOCUMENTS,
    )

    built = DOCUMENTS / tex.with_suffix(".pdf").name
    produces.write_bytes(built.read_bytes())