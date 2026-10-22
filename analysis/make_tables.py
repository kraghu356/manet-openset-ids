"""LaTeX table bodies for the manuscript, generated from results/*.json so that
no number is typed by hand. Writes tables/*.tex."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "tables"); os.makedirs(OUT, exist_ok=True)
r = json.load(open(os.path.join(R, "openset_results.json")))
A = ["bha", "flood", "gha", "sink", "worm"]

def ms(d, bold=False):
    if d["mean"] is None:
        return "--"
    s = f"{d['mean']:.3f} $\\pm$ {d['sd']:.3f}" if len(d["folds"]) > 1 else f"{d['mean']:.3f}"
    return f"\\textbf{{{s}}}" if bold else s

# closed set
names = [("random_forest", "Random forest, 300 trees"), ("gru", "Gated recurrent unit"), ("knn", "$k$-nearest neighbours, $k=5$"),
         ("lstm", "Long short-term memory \\ct{r:venkat}"), ("cnn", "Convolutional network \\ct{r:abdan}"),
         ("svm", "Support vector machine"), ("decision_tree", "Decision tree")]
with open(os.path.join(OUT, "closed.tex"), "w") as f:
    for k, lab in names:
        d = r["closed_set"][k]
        f.write(f"{lab} & {d['mean']:.3f} $\\pm$ {d['sd']:.3f} \\\\\n")

# open set AUROC, compact (mean only) with row mean
rows = [("msp_cnn", "MSP, CNN"), ("msp_lstm", "MSP, LSTM"), ("msp_gru", "MSP, GRU"),
        ("energy_cnn", "Energy, CNN"), ("energy_lstm", "Energy, LSTM"), ("energy_gru", "Energy, GRU"),
        ("openmax_cnn", "OpenMax, CNN"), ("openmax_lstm", "OpenMax, LSTM"), ("openmax_gru", "OpenMax, GRU"),
        None, ("rf_vote", "Forest vote"), ("rf_entropy", "Forest vote entropy"), ("svm_margin", "SVM margin"),
        ("knn_dist", "$k$-NN distance"), ("autoencoder", "Reconstruction"), None, ("nu", "$\\nu$"),
        ("hybrid_rf_ae", "Vote + recon."), ("hybrid_rf_nu", "Vote + $\\nu$"), ("hybrid_ae_nu", "Recon. + $\\nu$"),
        ("hybrid", "Hybrid, Eq.~(\\ref{eq:hybrid})"), ("knn_nu", "$k$-NN + $\\nu$, Eq.~(\\ref{eq:knnnu})")]
best = {a: max(r["open_set"][a][m]["auroc"]["mean"] or 0 for m in r["open_set"][a]) for a in A}
with open(os.path.join(OUT, "openset_auroc.tex"), "w") as f:
    for row in rows:
        if row is None:
            f.write("\\hline\n"); continue
        k, lab = row
        cells, vals = [], []
        for a in A:
            d = r["open_set"][a][k]["auroc"]
            v = d["mean"]; vals.append(0.5 if v is None else v)
            cells.append("--" if v is None else (f"\\textbf{{{v:.3f}}}" if abs(v - best[a]) < 5e-4 else f"{v:.3f}"))
        f.write(f"{lab} & " + " & ".join(cells) + f" & {np.mean(vals):.3f} \\\\\n")

# operating point
op = [("msp_gru", "MSP, GRU"), ("energy_gru", "Energy, GRU"), ("openmax_gru", "OpenMax, GRU"), ("rf_vote", "Forest vote"),
      ("knn_dist", "$k$-NN distance"), ("autoencoder", "Reconstruction"), ("nu", "$\\nu$"), ("hybrid", "Hybrid"), ("knn_nu", "$k$-NN + $\\nu$")]
with open(os.path.join(OUT, "operating.tex"), "w") as f:
    for k, lab in op:
        c = [f"{r['open_set'][a][k]['tpr_at_val95']['mean']:.2f}/{r['open_set'][a][k]['fpr_at_val95']['mean']:.2f}" for a in A]
        dr = [r["open_set"][a][k]["tpr_at_val95"]["mean"] for a in A]
        f.write(f"{lab} & " + " & ".join(c) + f" & {np.mean(dr):.2f} \\\\\n")

# autoencoder ablation
ab = r["autoencoder_ablation"]
lbl = {"bha": "Black hole", "flood": "Flooding", "gha": "Grey hole", "sink": "Sinkhole", "worm": "Wormhole"}
with open(os.path.join(OUT, "ablation.tex"), "w") as f:
    for a in A:
        f.write(f"{lbl[a]} & " + " & ".join(ms(ab[s][a]) for s in ["F_all", "F_observable", "F_observable_minus_nu"]) + " \\\\\n")

# complexity
c = json.load(open(os.path.join(R, "complexity.json")))
cr = [("nu", "$\\nu$"), ("decision_tree", "Decision tree"), ("svm", "Support vector machine"), ("knn", "$k$-nearest neighbours"),
      ("random_forest", "Random forest, 300 trees"), ("cnn", "Convolutional network"), ("lstm", "Long short-term memory"),
      ("gru", "Gated recurrent unit"), ("autoencoder", "Autoencoder")]
tm = r["timing"]
trn = {"random_forest": tm["rf_train_s"]["mean"], "cnn": tm["cnn_train_s"]["mean"], "lstm": tm["lstm_train_s"]["mean"],
       "gru": tm["gru_train_s"]["mean"], "autoencoder": tm["autoencoder_train_s"]["mean"]}
with open(os.path.join(OUT, "complexity.tex"), "w") as f:
    for k, lab in cr:
        d = c[k]; p = f"{d['params']/1000:.1f}k" if "params" in d else "--"
        t = trn.get(k, d.get("train_s"))
        ts = "--" if t is None else (f"{t:.2f}" if t < 10 else f"{t:.0f}")
        lt = f"{d['latency_ms']:.4f}" if d["latency_ms"] < 0.01 else f"{d['latency_ms']:.2f}"
        f.write(f"{lab} & {p} & {d['size_kb']:.1f} & {ts} & {lt} \\\\\n")
print("tables written to", os.path.abspath(OUT))
