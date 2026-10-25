# Independent metric recomputation, 25 October 2026

The archived node-level forest predictions in `results/node_scores.csv.gz` were loaded and their ROC AUROCs recalculated separately for each of the 4 folds and 5 held-out attacks. All 20 calculated AUROCs match `results/openset_results.json` to the manuscript's four-decimal precision.

| Attack | Mean recalculated forest-vote AUROC |
|---|---:|
| Black hole | 0.9787 |
| Flooding | 0.6618 |
| Grey hole | 0.9668 |
| Sinkhole | 0.7692 |
| Wormhole | 0.4963 |

The frozen dataset ledger contains 120 runs, 347900 node-window rows and 11332 retained sequences. Correct normal-node test counts: fold 1 2223, fold 2 2190, fold 3 2213, fold 4 2041. Old illustrative counts 2207 and 2215 are not the frozen fold populations and should be removed.

This confirms reproduction of **metrics from saved predictions**. It is NOT confirmation of full model-training reproduction: a new forest training run did not reproduce the archived AUROC values, and the reason remains unverified. The new hard-vote-vs-leaf comparison is a separate rerun and should not overwrite Table 4.

Wormhole invariant scope: the existing simulated stress study found zero detections among 16 relay-tunnel endpoints and two detections among six endpoints in the 200-node setting. The protected monitor was assumed rather than physically implemented. Thus its simulated out-of-band tunnel claim can be bounded and properly reported, but live hardware validation cannot be marked complete.

Suggested corrected phrasing: "The published fold-level random-forest AUROCs were recomputed from archived node predictions, with agreement at reported precision in all 20 fold-by-attack comparisons. The separate retraining reproducibility question remains open because a new environment produced different absolute AUROCs."

To recompute: use sklearn.metrics.roc_auc_score on rows in `results/node_scores.csv.gz` with matching held_out/fold and cls either normal or the held-out attack; compare to `results/openset_results.json`, key open_set[attack].rf_vote.auroc.folds[fold-1].
