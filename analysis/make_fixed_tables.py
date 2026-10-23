"""LaTeX bodies for the frozen-detector tables (operating points, stress test,
independent campaign) from results/fixed_detector.json. Writes
results/tables/fixed_*.tex and prints them."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "results", "fixed_detector.json")))
OUT = os.path.join(HERE, "..", "results", "tables"); os.makedirs(OUT, exist_ok=True)
A = ["bha", "flood", "gha", "sink", "worm"]
AN = {"bha": "Black hole", "flood": "Flooding", "gha": "Grey hole", "sink": "Sinkhole", "worm": "Wormhole"}
M = [("knn_dist", "$k$-NN distance"), ("knn_nu", "$k$-NN + $\\nu$"), ("hybrid", "Hybrid")]
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"


def save(name, rows):
    s = "\n".join(rows) + "\n"
    open(os.path.join(OUT, name), "w").write(s); print(f"% {name}\n{s}")


if R.get("thresholds"):
    rows = []
    for key, lab in M:
        for q in ["90", "95", "99", "99.5"]:
            t = [R["thresholds"][h][key][q] for h in A]
            rows.append(" & ".join([lab if q == "90" else "", q,
                                    f2(sum(x["tpr"] for x in t) / 5), f2(min(x["tpr"] for x in t)),
                                    f3(max(x["fpr"] for x in t)),
                                    f"{sum(x['false_alerts_per_run'] for x in t) / 5:.1f}",
                                    f"{max(x['false_alerts_per_run'] for x in t):.1f}",
                                    f2(sum((x['precision'] or 0) for x in t) / 5)]) + " \\\\")
    save("fixed_thresholds.tex", rows)

NAMES = {"baseline": "Baseline", "shadow_2dB": "Shadowing 2 dB", "shadow_4dB": "Shadowing 4 dB",
         "shadow_6dB": "Shadowing 6 dB", "loss_5pct": "Frame loss 5\\%", "loss_10pct": "Frame loss 10\\%",
         "speed_5_15": "Speed 5--15 m/s", "speed_10_20": "Speed 10--20 m/s", "dense_1000m": "Dense field",
         "large_200n": "200 nodes", "rayleigh_fading": "Rayleigh fading", "hello_on": "Hello messages",
         "heard_memory_30s": "Memory 30 s", "heard_memory_10s": "Memory 10 s", "relay_wormhole": "Relay wormhole"}
if R.get("stress"):
    rows = []
    for c, v in R["stress"].items():
        rows.append(" & ".join([NAMES[c], str(v["runs"]["endpoint_runs"]), str(v["knn_nu"]["n_pos"])] +
                               [f"{f3(v[k]['auroc'])} / {f2(v[k]['p95']['fpr'])}" for k, _ in M]) + " \\\\")
    save("fixed_stress.tex", rows)
    ci = lambda d: f"{d['tpr']:.2f} [{d['tpr_ci'][0]:.2f}, {d['tpr_ci'][1]:.2f}]"
    cf = lambda d: f"{d['fpr']:.3f} [{d['fpr_ci'][0]:.3f}, {d['fpr_ci'][1]:.3f}]"
    rows = []
    for c in ["baseline", "shadow_6dB", "speed_10_20", "large_200n", "heard_memory_10s", "relay_wormhole"]:
        v = R["stress"].get(c)
        if not v or "tpr_ci" not in v["knn_nu"]["p95"]:
            continue
        rows.append(" & ".join([NAMES[c], str(v["knn_nu"]["n_pos"]), str(v["knn_nu"]["n_neg"]),
                                ci(v["knn_nu"]["p95"]), cf(v["knn_nu"]["p95"]),
                                ci(v["hybrid"]["p95"]), cf(v["hybrid"]["p95"])]) + " \\\\")
    save("fixed_stress_ci.tex", rows)

if R.get("independent"):
    rows = []
    for cond, lab in [("same_env", "Same"), ("shift_env", "Shifted")]:
        if cond not in R["independent"]:
            continue
        v = R["independent"][cond]
        for k, kl in M:
            au = [v[h][k]["auroc"] for h in A]
            rows.append(" & ".join([lab if k == "knn_dist" else "", kl] + [f3(x) for x in au] +
                                   [f3(sum(au) / 5),
                                    f2(sum(v[h][k]["p95"]["tpr"] for h in A) / 5),
                                    f2(max(v[h][k]["p95"]["fpr"] for h in A)),
                                    f2(sum(v[h][k]["p99"]["tpr"] for h in A) / 5),
                                    f3(max(v[h][k]["p99"]["fpr"] for h in A))]) + " \\\\")
    save("fixed_independent.tex", rows)
