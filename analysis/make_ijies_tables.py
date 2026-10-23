"""LaTeX table bodies for the review analyses: all nodes (ijies_allnodes.json),
the online classifier-plus-gate run (ijies_gate.json) and the attack-intensity
campaign (intensity.json). Writes results/tables/{allnodes,online_gate,intensity}.tex."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
T = os.path.join(R, "tables"); os.makedirs(T, exist_ok=True)
A = ["bha", "flood", "gha", "sink", "worm"]
AN = {"bha": "Black hole", "flood": "Flooding", "gha": "Grey hole", "sink": "Sinkhole", "worm": "Wormhole"}
M = [("knn_dist", "$k$-NN distance"), ("knn_nu", "$k$-NN + $\\nu$"), ("hybrid", "Hybrid")]
f2 = lambda x: "--" if x is None else f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"


def save(n, rows):
    open(os.path.join(T, n), "w").write("\n".join(rows) + "\n"); print(f"% {n}"); print("\n".join(rows))


p = os.path.join(R, "ijies_allnodes.json")
if os.path.exists(p):
    a = json.load(open(p)); rows = []
    for var, lab in [("active_only", "Active nodes (main protocol)"), ("all_nodes", "All nodes"),
                     ("all_nodes_with_activity_input", "All nodes, activity as input")]:
        for k, kl in M:
            v = a[var][k]
            rows.append(" & ".join([lab if k == "knn_dist" else "", kl] + [f3(v[h]["auroc"]) for h in A] +
                                   [f3(v["mean_auroc"]), f2(v["mean_tpr"]), f2(v["max_fpr"])]) + " \\\\")
    save("allnodes.tex", rows)

p = os.path.join(R, "ijies_gate.json")
if os.path.exists(p):
    g = json.load(open(p)); rows = []
    for q, ql in [("p95", "95th"), ("p99", "99th")]:
        for h in A:
            v = g[q][h]["mean"]
            lat = "--" if v["latency_s_median_unseen"] is None else f"{v['latency_s_median_unseen']:.0f}"
            rows.append(" & ".join([ql if h == "bha" else "", AN[h], f2(v["unknown_detection"]), f2(v["any_alarm_unseen"]), lat,
                                    f3(v["false_alarm_normal"]), f2(v["known_attack_alarm"]), f2(v["known_macro_f1_classifier"]),
                                    f2(v["known_macro_f1_with_gate"])]) + " \\\\")
    save("online_gate.tex", rows)

p = os.path.join(R, "intensity.json")
if os.path.exists(p):
    it = json.load(open(p)); rows = []
    lab = {"gha_p025": "Grey hole, drop 0.25", "gha_p075": "Grey hole, drop 0.75", "flood_r1": "Flooding, 1 request/s",
           "flood_r2": "Flooding, 2 requests/s", "bha_n2": "Black hole, 2 attackers", "bha_n10": "Black hole, 10 attackers"}
    for c in lab:
        if c not in it:
            continue
        v = it[c]
        rows.append(" & ".join([lab[c], str(v["runs"]), str(v["knn_nu"]["n_pos"])] +
                               [f"{f3(v[k]['auroc'])} / {f2(v[k]['p95']['tpr'])} / {f2(v[k]['p95']['fpr'])}" for k, _ in M]) + " \\\\")
    save("intensity.tex", rows)
