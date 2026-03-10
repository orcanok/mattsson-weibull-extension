

# Flexible Moral Hazard

## Overview

This repository implements and extends the analytically tractable principal–agent model in **Mattsson & Weibull (2023, GEB)** with **KL-divergence effort costs**. The baseline is a single-period, finite-outcome moral-hazard model with a closed-form “softmax” agent response and a **threshold–sharing** optimal wage schedule.

The main extension in this repo is a **two-period** version with long-term contracting: the period-2 wage can be conditioned on the period-1 outcome. Despite the dynamic linkage, the model remains tractable and preserves the canonical structure:

- Period 2 is the one-period solution conditional on history.
- Period 1 behavior depends on an **effective utility index** that adds discounted continuation utility.
- Optimal wage schedules admit a **threshold** (risk neutral) / **threshold–sharing** (concave utility) representation, with history dependence through continuation terms.

The repo includes a reproducible pipeline that:
1) solves the model numerically for selected parameter cases,  
2) writes results into a LaTeX snippet (`documents/generated/results.tex`), and  
3) compiles the paper PDF (`paper.pdf`).

---

## What this repo does

Given a parameter case (outcomes, baseline probabilities, preferences, discount factors, outside option), the pipeline:

1. Solves the principal’s problem under the canonical parameterization (thresholds):
   - $$\(t_1\)$$ for period 1,
   - $$\(t_2(i)\)$$ for period 2 conditional on history \(i\).

2. Computes:
   - Thresholds
   - principal profit $$\(\Pi\)$$ and agent utility $$\(U\)$$,
   - continuation terms $$\(C_i\)$$.

3. Writes the key numbers into:
   - `documents/generated/results.tex` (LaTeX macros),
   so `paper.tex` can include them directly.

4. Compiles `documents/paper.tex` to a PDF via a `pytask` pipeline (requires `pdflatex`).

---

## Parameter cases (`CASES`)

The numerical experiments are defined in `src/flexible_moral_hazard/params.py` as a dictionary `CASES`.  
Each entry specifies $$\(\pi\)$$, $$\(p^0\)$$, discount factors $$\((\beta,\delta)\)$$, preference parameters $$\((\rho,\theta,\varepsilon)\)$$, and the outside option $$\(\bar u\)$$.

### Cases included

- **`baseline_one_period`**  
  One-period benchmark implemented by setting `beta=0.0`, `delta=0.0`, with outside option  
  $$\(\bar u^{(1)}=\ln(0.12)\)$$.

- **`one_period_low_outside_option`**  
  Same one-period benchmark implementation (`beta=0.0`, `delta=0.0`).  
  **Note:** in the current dictionary this case still uses \(\bar u^{(1)}=\ln(0.12)\); if you intend a lower outside option, update `ubar` accordingly.

- **`baseline_two_period`**  
  Two-period long-term contract with `beta=1.0`, `delta=1.0` and two-period outside option  
  $$\(\bar u^{(2)}=(1+\delta)\ln(0.12)=2\ln(0.12)\)$$.

- **`two_period_low_outside_option`**  
  Two-period long-term contract with `beta=1.0`, `delta=1.0` and  
  $$\(\bar u^{(2)}=(1+\delta)\ln(0.05)=2\ln(0.05)\)$$.

- **`two_period_impatient_principal`**  
  Two-period long-term contract with impatient principal (`beta=0.1`) and patient agent (`delta=1.0`),  
  with $$\(\bar u^{(2)}=2\ln(0.12)\)$$.

All cases currently use:
- outcomes $$\(\pi \in \{-1,0,1\}\)$$,
- baseline probabilities $$\(p^0=(0.5,0.3,0.2)\)$$,
- unit relative risk aversion $$\(\rho=1\)$$,
- $$\(\theta=1\)$$,
- external income $$\(\varepsilon=0.05\)$$.

---

## Repo layout

```text
.
├── documents/
│   ├── paper.tex                  # LaTeX paper (inputs generated/results.tex)
│   ├── generated/                 # Auto-generated TeX snippets (not committed)
│   │   └── results.tex
│   └── task_documents.py          # pdflatex task to compile the paper
├── tasks/
│   ├── task_results_tex.py        # Runs model for cases and writes results.tex
│   └── task_documents.py          # Compiles the LaTeX paper to PDF
├── src/flexible_moral_hazard/
│   ├── model.py                   # Core formulas (utility, softmax, evaluator)
│   ├── solve.py                   # Optimizers (optimagic wrapper, constraints)
│   ├── params.py                  # Case dictionaries (CASES)
│   ├── io.py                      # write_results_tex(...)
│   └── config.py                  # Paths: ROOT, DOCUMENTS, BLD, ...
├── bld/                           # Build outputs (optional; task-dependent)
├── pyproject.toml                 # pixi env + tooling config
└── README.md
## How to Run

This project uses `pixi` for the environment and `pytask` for reproducible execution.

Experience the reproducibility of this template in less than five minutes:

1. **Install [Pixi](https://pixi.sh/)**

1. **Clone this repository**.

1. **Install dependencies**:

   ```bash
   pixi install
   ```

1. **Run the full pipeline (generate TeX results + compile paper)**:

```bash
pixi run pytask
```

5. **Run only the TeX-results generation**:

```bash
pixi run pytask -k task_results_tex"
```


## LaTeX requirement (for PDF task)

This repository includes `pytask` tasks that compile a paper and slides to PDF using
LaTeX. To run those tasks, you need a **system LaTeX installation** that provides
`pdflatex` (e.g., TeX Live on Linux, MacTeX/BasicTeX on macOS, TeX Live/MiKTeX on
Windows).

Quick check:

```bash
pdflatex --version
```

	•	documents/generated/results.tex — generated LaTeX macros for all cases
	•	paper.pdf — compiled PDF (copied to repo root by the LaTeX task)

## Outputs

Running `pixi run pytask` creates three types of artifacts:

- **documents/generated/results.tex** — generated LaTeX macros for all cases

- **paper.pdf — compiled PDF (copied to repo root by the LaTeX task)**


