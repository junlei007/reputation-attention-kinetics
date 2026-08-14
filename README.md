# Reputation–Attention Kinetics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21927468.svg)](https://doi.org/10.5281/zenodo.21927468)

**Microscopic—macroscopic theory of reputation capital and attention memory
in directed trust networks** — the two-dimensional `(C, H)` state
augmentation (reputation capital + exponentially-decayed attention memory),
with a quantitative finite-time propagation-of-chaos result and
cross-platform stress tests on Bitcoin OTC/Alpha and Wikipedia RfA.

This repository contains the simulation and estimation code, numerical
outputs, and tests needed to reproduce the study's results. Manuscript and
submission files are maintained separately and are not included in this
public software repository.

## Setup

```bash
uv sync          # creates .venv with all dependencies (Python >= 3.14)
uv run python -m pytest tests/ -m "not slow"   # fast unit tests
```

## Data

Public SNAP datasets are not redistributed in this repository. Download them
from the linked source pages and place them at the paths below:

- [Bitcoin OTC web of trust](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html):
  `data/bitcoin_otc/soc-sign-bitcoinotc.csv.gz`
- [Bitcoin Alpha web of trust](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html):
  `data/bitcoin_alpha/soc-sign-bitcoinalpha.csv.gz`
- [Wikipedia Requests for Adminship](https://snap.stanford.edu/data/wiki-RfA.html):
  `data/wiki_rfa/wiki-RfA.txt.gz`

The raw files are excluded from version control. Dataset-specific notes are in
the corresponding `data/*/README.md` files.

## Reproducing the pipeline (in order)

```bash
uv run python scripts/01_audit.py              # data integrity audit (G1)
uv run python scripts/02_facts.py              # descriptive facts (G1)
uv run python scripts/03_synthetic_recovery.py # parameter recovery (G2)
uv run python scripts/04_micro_macro.py        # 1D micro-macro consistency (G3)
uv run python scripts/06_holdout.py            # estimation + holdout stress test (G5)
uv run python scripts/07_robustness.py         # robustness & replication
uv run python scripts/08_wiki_rfa.py           # Wikipedia RfA analysis
uv run python scripts/09_wiki_rfa_robustness.py# RfA robustness
uv run python scripts/12_ablation_attention.py # attention-memory ablation (OTC/Alpha)
uv run python scripts/13_micro_macro_2d.py     # 2D (C,H) validation
uv run python scripts/10_figures.py            # figures (v1)
uv run python scripts/11_figures_v2.py         # main-text figures (v2)
```

Every script writes its results to `experiments/*.json` (self-contained,
reproducible).  The gate decisions are recorded in `notes/gates_log.md`.

## Layout

```
src/dengyunetwork/   package: data, audit, facts, simulator, kinetic
                     (1D and 2D solvers), estimation, plotstyle
scripts/             01..13 pipeline scripts
experiments/         all result JSONs
figures/             paper figures
notes/               technical notes: model freeze, kinetic derivation,
                     proof sketch, gate log, estimation design
tests/               unit tests (kinetic 2D solver)
data/                public SNAP datasets (gitignored)
```

## Key facts

- Bitcoin OTC: 35,592 directed rating events, 5,881 users, 2010-11 to 2016-01,
  every directed pair rated at most once.
- Main split: 70% of the time span trains (cutoff 2014-07-03), 30% tests
  (2,601 events, the platform's declining tail).
- Two-dimensional validation (`experiments/micro_macro_2d.json`): marginal
  W1 broadly consistent with an N^{-1/2} fluctuation component on top of a
  discretisation floor (C: 0.0038 + 0.20 N^{-1/2}; H: 0.05 + 1.0 N^{-1/2}),
  event-rate trajectory within 1.1%, decay remap conservative to machine
  precision.

## Citation

Please cite the archived software release:

> Du, J., & Li, S. (2026). *Reputation–Attention Kinetics* (Version v0.1.0)
> [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21927468

The version DOI above identifies the code used for the manuscript. The Zenodo
concept DOI for all software versions is
https://doi.org/10.5281/zenodo.21927467. Machine-readable metadata are in
`CITATION.cff`.

## License

The source code and repository materials are released under the MIT License.
The SNAP datasets remain subject to their original terms and are not included.
