# Verified reviewer corrections — 25 October 2026

## Exact fold populations

The repository's `results/row_ledger.csv` has 120 runs, 347900 node-window records, 157723 active node-window records, 12000 node sequences and 11332 retained sequences. The sole 28-window run is `bha_s17`, accounting for 100 fewer records than 120 x 100 x 29.

The current four-fold evaluation test counts are:

| Fold | Normal | Black hole | Flooding | Grey hole | Sinkhole | Wormhole |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2223 | 20 | 19 | 19 | 20 | 7 |
| 2 | 2190 | 20 | 19 | 16 | 20 | 8 |
| 3 | 2213 | 19 | 20 | 19 | 20 | 8 |
| 4 | 2041 | 18 | 20 | 19 | 20 | 8 |

Historical illustrative normal counts 2207 and 2215 could not be mapped to this frozen four-fold evaluation and should be removed from the manuscript rather than explained speculatively.

## Artifact integrity

The uploaded ZIP's SHA-256 hashes match `MANIFEST.md` for `dataset3.csv.gz`, `row_ledger.csv`, `ijies_dilution.json`, `ijies_gate.json`, and `openset_results.json`. Hash matching is not the same as independently re-running ns-3 simulations or neural training.

## Forest score comparison

The merged supplementary experiment records 20 pairs of hard-vote versus leaf-probability AUROCs agreeing to four decimal places **in the local rerun**. The local rerun's absolute forest AUROC values differ from the saved manuscript results; do not substitute those values into Table 4 without reproducing the documented environment and resolving the discrepancy. Equal observed AUROC is not proof of pointwise equality of the score functions.

## Wormhole scope and negative controls

The invariant performs well for the implemented out-of-band receive-handler tunnel but flags 0/16 relay-tunnel endpoints and 2/6 endpoints in the 200-node stress scenario. The protected endpoint monitor is assumed, not hardware-implemented. Hardware effectiveness is unverified.

## Proposed manuscript correction text

**Population:** "The four test partitions retain 2223, 2190, 2213 and 2041 normal node sequences after selection by forwarding activity. Per-attack retained test counts are reported for each rotation. Obsolete illustrative populations from exploratory analyses are omitted."

**Forest:** "An additional four-fold comparison of hard-tree vote disagreement with uncertainty computed from the forest's averaged leaf probabilities gives AUROCs equal to four decimal places in the recorded rerun. Its absolute values differ from the original saved forest scores; this sensitivity comparison should not be interpreted as an independent reproduction of the original table."

**Wormhole:** "The unheard-neighbour invariant is specific to out-of-band tunnels whose peers are entered as one-hop routing neighbours without corresponding decoded radio frames, conditional on a trusted endpoint monitor. The tested relay mechanism defeats the invariant (0 of 16 endpoints detected), and hardware deployment has not been evaluated."

## Outstanding experiments

The full evaluation pipeline has **not** been independently reproduced in its documented package versions; the absolute forest discrepancy remains unresolved. Real protected-monitor hardware validation has **not** been performed. Neither point should be labeled experimentally complete.
