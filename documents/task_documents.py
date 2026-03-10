# tasks/task_documents.py
import subprocess
from pathlib import Path

import pytask

from flexible_moral_hazard.config import DOCUMENTS, ROOT

PAPER_FIGURES: list[Path] = []

# This is the file produced by task_results_tex.
GENERATED_RESULTS: Path = DOCUMENTS / "generated" / "results.tex"


@pytask.task(id="latex-paper")
def task_compile_latex_paper(
    tex: Path = DOCUMENTS / "paper.tex",
    results_tex: Path = GENERATED_RESULTS,
    figures: list[Path] = PAPER_FIGURES,
    produces: Path = ROOT / "paper.pdf",
) -> None:
    """Compile LaTeX paper to PDF."""
    # First pass
    subprocess.run(
        ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name),
        check=True,
        cwd=DOCUMENTS,
    )
    # Second pass (refs/citations)
    subprocess.run(
        ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name),
        check=True,
        cwd=DOCUMENTS,
    )

    built = DOCUMENTS / tex.with_suffix(".pdf").name  # typically documents/paper.pdf
    produces.write_bytes(built.read_bytes())