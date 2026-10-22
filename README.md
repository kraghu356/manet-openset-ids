# Open-Set Detection of Unseen Routing Attacks in Mobile Ad Hoc Networks

Simulator, attack module, dataset and analysis code for the manuscript
"Open-Set Detection of Unseen Routing Attacks in Mobile Ad Hoc Networks"
(earlier working title: "When Unseen Attacks Look Normal: Open-Set Evaluation,
Feature Observability, and Protocol-Invariant Detection in Mobile Ad Hoc
Networks").

Amruth V (ORCID 0009-0005-6346-0201) and Devaraj Verma C (ORCID
0000-0002-1504-4263), Department of Computer Science and Engineering, JAIN
(Deemed-to-be University), Bengaluru. Correspondence: amruthv020@gmail.com

The version that produced every number in the submitted manuscript is commit
`ec4f0aa` on branch `revision-openset-eval`.

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
| `analysis/complexity.py` | Model size and per-node latency; writes `results/complexity.json` |
| `analysis/nu_robustness.py`, `analysis/nu_robustness_report.py` | Stress test of the unheard-neighbour count; writes `results/nu_robustness.csv` |
| `analysis/make_tables.py`, `analysis/make_figures.py` | Every table body and figure in the paper, from the result files only |

Row count: 120 runs x 100 nodes x 29 windows = 348,000; the recorded output of
black hole seed 17 ends after 28 windows, giving 347,900.

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
| Table 9 | `analysis/make_tables.py` | `results/complexity.json`, `results/openset_results.json` |
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
