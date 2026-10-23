"""Scoring time of the random forest for one node, with the hard-vote loop used
in the paper and with the library's vectorised path, on one CPU thread.

The paper's score (Eq. 4) needs hard votes, which our implementation obtains
by querying the 300 trees one at a time in Python. A deployed node would use a
compiled traversal; predict_proba is the nearest library equivalent (soft
votes, same trees), and a 50-tree forest shows the effect of model size.
Writes results/rf_latency.json.
"""
import json, os, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import openset_eval as E

a = E.load(); feats = E.F_OBS; trs, vas, tes = E.FOLDS[0]; h = "worm"
known = sorted([c for c in E.ATTACKS if c != h] + ["normal"]); ki = {c: i for i, c in enumerate(known)}
Xtr, Mtr, ctr, _, _ = E.node_table(a[a.seed.isin(trs) & (a.scenario != h)], feats)
Xte, Mte, _, _, _ = E.node_table(a[a.seed.isin(tes)], feats)
sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
f = lambda X: sc.transform(X.reshape(-1, len(feats))).reshape(X.shape).astype(np.float32)
Ftr, Fte = E.flat(f(Xtr), Mtr), E.flat(f(Xte), Mte); ytr = np.array([ki[c] for c in ctr])
N = 200


def lat(fn):
    fn(0); t = time.perf_counter()
    for i in range(N): fn(i)
    return (time.perf_counter() - t) / N * 1e3


out = {}
for n in [300, 50]:
    rf = RandomForestClassifier(n, random_state=0, class_weight="balanced", n_jobs=1).fit(Ftr, ytr)
    out[f"trees_{n}"] = {"hard_vote_loop_ms": lat(lambda i: E.vote(rf, Fte[i:i + 1])),
                         "predict_proba_ms": lat(lambda i: rf.predict_proba(Fte[i:i + 1]))}
json.dump(out, open(os.path.join(E.HERE, "..", "results", "rf_latency.json"), "w"), indent=1)
print(out)
