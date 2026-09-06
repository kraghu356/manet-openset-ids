# Open-Set Intrusion Detection for MANETs

Simulator, attack module, dataset and analysis for "When Unseen Attacks Look
Normal: Open-Set Evaluation, Feature Observability, and Protocol-Invariant
Detection in Mobile Ad Hoc Networks".

Amruth V and Devaraj Verma C, JAIN (Deemed-to-be University), Bengaluru.
Correspondence: amruthv020@gmail.com

## Contents

- `ns3/manet-gen.cc` — generator: 100 nodes, 1500x1500 m, 802.11b, AODV
- `ns3/aodvatk/` — AODV module with five attacks implemented in the routing
  protocol: black hole, grey hole, sinkhole, flooding, wormhole
- `analysis/` — feature extraction, seven detectors, leave-one-attack-out
- `results/dataset2.csv.gz` — 120 runs, 347,900 node-window rows

## Reproducing

Copy `ns3/manet-gen.cc` to `scratch/` and `ns3/aodvatk/` to `src/` in an ns-3.48
tree, then `./ns3 configure && ./ns3 build`. Baseline configuration:

    ./build/scratch/ns3.48-manet-gen-optimized --nNodes=100 --areaX=1500 \
      --areaY=1500 --simTime=300 --nFlows=10 --pktRate=4 --txPower=23 \
      --plExp=2.5 --standard=80211b --dataMode=DsssRate11Mbps --seed=1 \
      --outDir=out/normal_s1

Clean network: PDR 0.824, mean 2.18 hops.

Attack flags: `--nMalicious`, `--dropProb`, `--forge`, `--floodRate`,
`--wormhole`. See the paper's Section 5 for the mapping to attack classes.
