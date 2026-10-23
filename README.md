# Open-Set Detection of Unseen Routing Attacks in Mobile Ad Hoc Networks

Simulator, attack module, dataset and analysis code for the manuscript
"Open-Set Detection of Unseen Routing Attacks in Mobile Ad Hoc Networks"
(earlier working title: "When Unseen Attacks Look Normal: Open-Set Evaluation,
Feature Observability, and Protocol-Invariant Detection in Mobile Ad Hoc
Networks").

Amruth V (ORCID 0009-0005-6346-0201) and Devaraj Verma C (ORCID
0000-0002-1504-4263), Department of Computer Science and Engineering, JAIN
(Deemed-to-be University), Bengaluru. Correspondence: amruthv020@gmail.com

**Manuscript version.** Every number in the submitted manuscript comes from commit
[`8620bd4`](https://github.com/kraghu356/manet-openset-ids/tree/8620bd43e6ca74c0edc3b671739865244bd41291)
on branch `revision-openset-eval` (full hash 8620bd43e6ca74c0edc3b671739865244bd41291).
`MANIFEST.md` at that commit lists the command, inputs and SHA-256 checksums for
every table and figure. Earlier titles and the older `dataset2` belong to
previous versions; the paper uses `dataset3` only.

## Contents

| Path | What it is |
|---|---|
| `ns3/manet-gen.cc` | Scenario generator: 100 nodes, 1500 x 1500 m, 802.11b, AODV, per-node features every 10 s |
| `ns3/aodvatk/` | AODV module with the five attacks implemented inside the routing protocol |
| `results/dataset3.csv.gz` | **Dataset used in the paper**: 120 runs, 347,900 node-window rows |
| `results/dataset2.csv.gz` | Earlier dataset; identical except for the 20 wormhole runs (see below) |
| `analysis/run_campaign.py` | Regenerates the 6 scenarios x 20 seeds; exact attack flags are in its docstring |
| `analysis/build_dataset3.py` | Builds dataset3 from dataset2 and regenerated wormhole runs |
| `analysis/openset_eval.py` | Closed-set and leave-one-attack-out evaluation; writes `results/openset_results.json` |
| `analysis/extra_metrics.py` | PR-AUC, run-clustered bootstrap intervals, no-speed ablation; writes `results/extra_metrics.json` |
| `results/independent_runs.csv.gz` | Independent test campaign: 42 runs (24 at the paper's settings, seeds 101-104; 18 with 2 dB shadowing and speeds 2-8 m/s, seeds 201-203), 121,800 rows (139,200 with the calibration runs), never used to choose any method or threshold; plus 6 normal calibration runs in the shifted environment (condition `shift_cal`, seeds 204-209, 17,400 rows) used only by `analysis/recalibration.py` |
| `analysis/independent_campaign.py` | Regenerates the independent campaign |
| `analysis/fixed_detector_eval.py` | Frozen detectors (fitted once on seeds 1-16) on the independent campaign and the stress runs, and operating points at four thresholds; writes `results/fixed_detector.json` |
| `analysis/make_fixed_tables.py` | Tables 9-13 and 15 from `results/fixed_detector.json`, `results/recalibration.json`, `results/knn_ablation.json` |
| `results/stress_runs.csv.gz` | Wormhole stress runs (134 runs, 15 conditions; corrected and honest-node campaigns); unpack with `analysis/stress_runs_io.py unpack` |
| `analysis/openset_baselines.py` | Mahalanobis and reciprocal-point baselines and the fusion ablation (Table 14); writes `results/openset_baselines.json` |
| `analysis/baseline_checks.py` | RPL convergence, hyperparameter and sign checks; paired run-clustered bootstrap of k-NN + nu against Mahalanobis; writes `results/baseline_checks.json` |
| `analysis/recalibration.py` | Recalibration of the frozen detectors with local normal traffic in the shifted environment (Table 13); writes `results/recalibration.json` |
| `analysis/knn_ablation.py` | Ablation of k-NN + nu: k, feature groups, invariant term (Table 15); writes `results/knn_ablation.json` |
| `analysis/row_ledger.py` | Run-by-run ledger from raw rows to evaluated node sequences; writes `results/row_ledger.csv` |
| `analysis/ijies_checks.py` | Every-node evaluation, online classifier-plus-gate with time to alarm, per-feature reconstruction test, training seeds, node-level scores and checkpoints |
| `analysis/intensity_campaign.py`, `analysis/intensity_eval.py` | Attack-intensity campaign (17 runs released as `results/intensity_runs.csv.gz`) and frozen-detector results |
| `results/node_scores.csv.gz`, `results/checkpoints/` | Node-level scores of every non-neural detector and the autoencoder for every fold and held-out attack; autoencoder checkpoints per fold and recurrent-classifier checkpoints for fold 1 |
| `analysis/rf_latency.py` | Forest scoring time, hard-vote loop against the library's vectorised path; writes `results/rf_latency.json` |
| `analysis/make_manifest.py` | Writes `MANIFEST.md`: command, inputs and SHA-256 checksums for every table and figure |
| `analysis/check_paper_tables.py` | Confirms that every generated table row appears unchanged in the manuscript source |
| `analysis/complexity.py` | Model size and per-node latency; writes `results/complexity.json` |
| `analysis/nu_robustness.py`, `analysis/nu_robustness_report.py` | Stress test of the unheard-neighbour count; writes `results/nu_robustness.csv` |
| `analysis/make_tables.py`, `analysis/make_figures.py` | Every table body and figure in the paper, from the result files only |

Row count: 120 runs x 100 nodes x 29 windows = 348,000; the recorded output of
black hole seed 17 ends after 28 windows, giving 347,900 (58,000 rows per
scenario, 57,900 for the black hole). Every other run has 29 windows. The
wormhole run of seed 17 had been cut short by an interrupted simulation and was
regenerated with the same seed and build (commit bedce0f) before any reported
result was computed; this is the "seed 17 regenerated" of the commit history.
All tables and figures use this one snapshot, `dataset3.csv.gz`, with the
checksum below.

A fresh clone of this branch, run with `analysis/extra_metrics.py`, reproduces
`results/extra_metrics.json` exactly (largest difference 0 in every AUROC,
average precision and bootstrap bound).

### dataset2 versus dataset3

The tunnel replay registers each wormhole endpoint in its own AODV neighbour
table. The node never hears itself, so the unheard-neighbour count `nb_no_rssi`
in dataset2 included that self-entry at wormhole endpoints. The generator now
excludes the node's own address. Honest nodes and the attackers of the other
four scenarios never list themselves, so their rows are unchanged; dataset3
keeps them and replaces the 20 wormhole runs with runs from the corrected
generator.

## Map from paper to outputs

| Paper item | Produced by | From |
|---|---|---|
| Table 3, Table 4, Table 5, Table 6 | `analysis/make_tables.py` | `results/openset_results.json` |
| Table 7, Table 8 (fold-level, PR-AUC, onboard-only, bootstrap) | `analysis/make_tables.py` | `results/extra_metrics.json` |
| Table 9 (operating points), Tables 10-11 (full-detector stress, with Wilson intervals), Table 12 (independent campaign) | `analysis/make_fixed_tables.py` | `results/fixed_detector.json` |
| Table 13 (recalibration in the shifted environment) | `analysis/make_fixed_tables.py` | `results/recalibration.json` |
| Table 14 (open-set baselines, fusion rule) | `analysis/openset_baselines.py` | `results/openset_baselines.json`; RPL checks and paired bootstrap in `results/baseline_checks.json` |
| Table 15 (ablation of k-NN + nu) | `analysis/make_fixed_tables.py` | `results/knn_ablation.json` |
| Table 16 (every node), Table 17 (online classifier-plus-gate) | `analysis/ijies_checks.py`, `analysis/make_ijies_tables.py` | `results/ijies_allnodes.json`, `results/ijies_gate.json` |
| Table 18 (attack intensity) | `analysis/intensity_eval.py`, `analysis/make_ijies_tables.py` | `results/intensity.json` |
| Dilution test, training seeds (text of Sections 6.5 and 6.14) | `analysis/ijies_checks.py` | `results/ijies_dilution.json`, `results/ijies_seeds.json` |
| Table 19 (computational cost) | `analysis/make_tables.py` | `results/complexity.json`, `results/rf_latency.json` |
| Fig. 1 | `analysis/make_figures.py` | fold definition in `analysis/openset_eval.py` |
| Fig. 2, Fig. 3 | `analysis/make_figures.py` | `results/openset_results.json` |
| Fig. 4 | `analysis/make_figures.py` | `results/nu_robustness.csv` |

## Reproducing

Environment used: Python 3.13.16, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1,
PyTorch 2.14.1 (CPU), scipy 1.18.1, matplotlib 3.11.2; ns-3.48 built with
g++ 13.3.0 and CMake 3.28.3 on Linux.

1. Build the simulator. Copy `ns3/manet-gen.cc` to `scratch/` and `ns3/aodvatk/`
   to `src/` in an ns-3.48 tree, then

       ./ns3 configure -d optimized --enable-modules=aodvatk,aodv,wifi,mobility,applications,internet,flow-monitor
       ./ns3 build

   Run every simulation with address-space randomisation disabled
   (`setarch $(uname -m) -R ...`; the scripts do this). The tunnel code iterates
   a container ordered by memory address, so wormhole runs otherwise differ
   between repetitions of the same seed.

2. Analysis from the released dataset (no simulation needed):

       python3 analysis/openset_eval.py          # about 1-2 h on two CPU cores; resumes from results/parts/
       python3 analysis/extra_metrics.py
       python3 analysis/complexity.py
       python3 analysis/make_tables.py
       python3 analysis/make_figures.py

3. Regenerating data (optional):

       python3 analysis/run_campaign.py --ns3 ~/ns-3.48 --out runs --seeds 20
       python3 analysis/run_campaign.py --ns3 ~/ns-3.48 --out runs_fix --seeds 20 --scenarios worm
       python3 analysis/build_dataset3.py --worm-runs runs_fix
       python3 analysis/nu_robustness.py --ns3 ~/ns-3.48 --out runs_nu --seeds 5
       python3 analysis/nu_robustness.py --ns3 ~/ns-3.48 --out runs_nu_fix --seeds 3
       python3 analysis/nu_robustness_report.py --fixed runs_nu_fix --honest runs_nu
       python3 analysis/independent_campaign.py --ns3 ~/ns-3.48 --out runs_indep
       python3 analysis/fixed_detector_eval.py --indep runs_indep --stress-fix runs_nu_fix --stress-honest runs_nu
       python3 analysis/make_fixed_tables.py

   Regenerated runs reproduce the statistics of the released dataset; the
   original dataset2 runs predate the address-randomisation fix and cannot be
   reproduced row for row.

Clean network, seed 1: packet delivery ratio 0.824 over 2.18 hops on average.
Attack flags: `--nMalicious`, `--dropProb`, `--forge`, `--floodRate`,
`--wormhole`; the mapping to attack classes is in the paper's Section 4 and in
`analysis/run_campaign.py`.

## Checksums (SHA-256)

    results/dataset2.csv.gz  3f6038ba518a1fa581401ec36b8bd0fdefab74aa7d3e3a0dd9e05aa16a342a9e
    results/dataset3.csv.gz  78e3f767d3e6da19858e4b4c386f2b00c8dee62e9e1b058ce69a66eb44f3975a
    results/independent_runs.csv.gz  5e8c3235be74f79e29d999bfb64e29ecfaffe3191d8601cf8341532271f5b54d
