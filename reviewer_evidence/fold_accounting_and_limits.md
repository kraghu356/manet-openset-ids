# Reviewer verification notes

Dataset: 120 runs; 347900 recorded node-window rows; 157723 active rows; 12000 nodes; 11332 retained nodes; the sole incomplete run is bha_s17, with 28 windows.

Fold test class counts (normal, black hole, flooding, grey hole, sinkhole, wormhole):
- Fold 1: 2223, 20, 19, 19, 20, 7
- Fold 2: 2190, 20, 19, 16, 20, 8
- Fold 3: 2213, 19, 20, 19, 20, 8
- Fold 4: 2041, 18, 20, 19, 20, 8

The original illustrative counts 2207 and 2215 have not been traced to a single verified protocol. Replace them with the actual fold-specific counts.

The hard-vote and averaged-leaf-probability comparison is supplementary. The 20 AUROC pairs agree when rounded to 4 decimals in the local rerun; absolute values have not been reconciled with the manuscript's saved results.

The original reproducibility snapshot bd79a00 and later 1db7972 differ by README content. Manifest checksum matches do not substitute for executing every analysis script.

The protected wormhole-endpoint monitor has not been implemented or evaluated on real devices. Stress and relay results are simulated, not hardware validation.

A fully corrected native Springer manuscript with editable equations and tables remains separate outstanding work.