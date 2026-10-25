# Reviewer experiment: forest hard votes versus leaf probabilities

Date: 2026-10-25. Data source: `results/dataset3.csv.gz` from the uploaded repository archive.

## Protocol
Four published train/validation/test seed rotations, five leave-one-attack-out conditions, 15 observable inputs, and a freshly fitted 300-tree `RandomForestClassifier` (`random_state=0`, `class_weight="balanced"`). AUROC compares each held-out attack with normal nodes. Hard vote score: 1 minus maximum fraction of tree-level hard labels. Leaf-probability score: 1 minus maximum class probability from `predict_proba`. Both share the training partition and test observations.

## Observations
See `results/reviewer_rf_votes_vs_leaf_probability_20261025.csv`.
All 20 reported pairs of AUROCs agree to four decimal places. Across attacks, mean hard-vote / leaf-probability AUROC: black hole 0.9961 / 0.9961; flooding 0.5907 / 0.5907; grey hole 0.9658 / 0.9658; sinkhole 0.7260 / 0.7260; wormhole 0.4902 / 0.4902. Overall macro mean: 0.7537 / 0.7537.

## Reproducibility limitations
The fresh execution environment used scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3, and PyTorch 2.10.0. The original published repository manifest documents a different environment. The resulting forest AUROC levels differ from the manuscript's original saved forest metrics, so these values are **supplementary environment-specific rerun evidence**, not replacements for the paper's established Table 4.

The dataset and manifest checks previously matched recorded SHA-256 hashes, but the full original evaluation pipeline has not been rerun end-to-end in its exact environment. This experiment does not test wormhole detection on physical MANET hardware or implement a protected monitor.

Important: equality of rounded AUROCs does not prove that tree-vote and leaf-probability scores are mathematically the same for all models/data.

## How to repeat
From the repository root, run `python analysis/check_rf_vote_vs_proba.py`. It creates `results/reviewer_rf_comparison_rerun.json` with unrounded scores.
