# tasks/task_results_tex.py
from __future__ import annotations

from mattsson-weibull-extension.config import DOCUMENTS
from mattsson-weibull-extension.io import write_results_tex
from mattsson-weibull-extension.params import CASES
from mattsson-weibull-extension.solve import solve_with_optimagic


def task_results_tex(produces=DOCUMENTS / "generated" / "results.tex"):
    """Generate documents/generated/results.tex from cases in params.py."""
    results = {}

    for name, cfg0 in CASES.items():
        cfg = cfg0.copy()
        res, out = solve_with_optimagic(**cfg)

        if not getattr(res, "success", True):
            raise RuntimeError(
                f"Optimization failed for {name}: {getattr(res, 'message', '')}"
            )
        if out["U"] < float(cfg0["ubar"]) - 1e-8:
            raise RuntimeError(
                f"Infeasible for {name}: U={out['U']}, ubar={cfg0['ubar']}"
            )

        results[name] = out

    out_path = DOCUMENTS / "generated" / "results.tex"
    write_results_tex(out_path, results)
