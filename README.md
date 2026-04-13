# SCBE Experiments

> Reproducible experiment scripts + results for the SCBE-AETHERMOORE governance framework.

Every published number on [aethermoore.com/research/evidence.html](https://aethermoore.com/research/evidence.html)
sources from one of these files. Clone this repo to verify the claims yourself.

## Headline results

| File | Finding |
|---|---|
| `three_mechanism_results.json` | **99.42% combined AUC** across 6 attack types, 50 trials each (real 14-layer pipeline) |
| `pipeline_vs_baseline_results.json` | Pipeline-depth ablation — note the raw 14-layer hits 0.054 AUC on subtle attacks (why the combined defense matters) |
| `hyperbolic_experiment_results.json` | Hyperbolic vs Euclidean — Euclidean wins (0.9995 vs 0.9553), honestly published |
| `sacred_eggs_results.json` | Predicate-gated truth table: only (T,T,T,T) passes, 100% gate accuracy |
| `trajectory_curvature_results.json` | Phase curvature discrimination |

## Reproduce

```bash
git clone https://github.com/issdandavis/scbe-experiments.git
cd scbe-experiments
pip install numpy scipy scikit-learn  # if needed
python three_mechanism_combined.py    # should produce ~0.9942 combined AUC
python pipeline_vs_baseline.py
```

## Honest ledger

See [CLAIMS_AUDIT_V4.md](https://github.com/issdandavis/SCBE-AETHERMOORE/blob/main/docs/CLAIMS_AUDIT_V4.md)
in the main repo for what's proven, caveated, or disproven.

## Why a separate repo?

The main SCBE-AETHERMOORE monolith has 80+ top-level directories. ML engineers and
researchers who just want to verify the numbers shouldn't have to clone everything.
This repo is that isolated slice.

## License

MIT
