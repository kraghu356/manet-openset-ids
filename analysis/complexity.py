"""Model size, memory and per-node latency for every detector (Reviewer A).

Fold 1, wormhole held out (the largest training set). Latency is measured
for one node at a time (batch size 1) on a single CPU thread, which is the
relevant case for a node scoring itself; the mean of 200 nodes is reported.
Writes results/complexity.json.
"""
import json, os, pickle, time
import numpy as np, torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import openset_eval as E

torch.set_num_threads(1)
a = E.load(); feats = E.F_OBS; trs, vas, tes = E.FOLDS[0]; h = "worm"
known = sorted([c for c in E.ATTACKS if c != h] + ["normal"]); ki = {c: i for i, c in enumerate(known)}
Xtr, Mtr, ctr, _, _ = E.node_table(a[a.seed.isin(trs) & (a.scenario != h)], feats)
Xte, Mte, _, _, _ = E.node_table(a[a.seed.isin(tes)], feats)
ytr = np.array([ki[c] for c in ctr])
sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
f = lambda X: sc.transform(X.reshape(-1, len(feats))).reshape(X.shape).astype(np.float32)
Xtr, Xte = f(Xtr), f(Xte); Ftr, Fte = E.flat(Xtr, Mtr), E.flat(Xte, Mte)
N = 200; idx = np.arange(N)
out = {"n_train_nodes": int(len(Ftr)), "n_features": len(feats), "seq_len": E.SEQ}

def lat(fn):
    fn(0); t = time.perf_counter()
    for i in idx: fn(i)
    return (time.perf_counter() - t) / N * 1e3  # ms

def skl(name, m, score):
    t = time.perf_counter(); m.fit(Ftr, ytr); tr = time.perf_counter() - t
    out[name] = {"train_s": tr, "size_kb": len(pickle.dumps(m)) / 1024,
                 "latency_ms": lat(lambda i: score(m, Fte[i:i + 1]))}

skl("decision_tree", DecisionTreeClassifier(random_state=0, class_weight="balanced"), lambda m, x: m.predict_proba(x))
skl("random_forest", RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=1), lambda m, x: E.vote(m, x))
skl("knn", KNeighborsClassifier(5), lambda m, x: m.kneighbors(x))
skl("svm", SVC(class_weight="balanced", random_state=0), lambda m, x: m.decision_function(x))

for nm in ["cnn", "lstm", "gru"]:
    m = E.Seq(nm, len(feats), len(known)).eval()
    p = sum(q.numel() for q in m.parameters())
    X1, M1 = torch.from_numpy(Xte), torch.from_numpy(Mte)
    def go(i):
        with torch.no_grad(): m(X1[i:i + 1], M1[i:i + 1])
    out[nm] = {"params": p, "size_kb": p * 4 / 1024, "latency_ms": lat(go)}
m = E.AE(len(feats)).eval(); p = sum(q.numel() for q in m.parameters())
X1 = torch.from_numpy(Xte)
def go(i):
    with torch.no_grad(): m(X1[i:i + 1])
out["autoencoder"] = {"params": p, "size_kb": p * 4 / 1024, "latency_ms": lat(go)}

# nu: one set difference per window between routing neighbours and heard set
rt = [set(np.random.randint(0, 100, 10)) for _ in range(N)]; ph = [set(np.random.randint(0, 100, 40)) for _ in range(N)]
out["nu"] = {"size_kb": 100 * 4 / 1024, "latency_ms": lat(lambda i: len(rt[i] - ph[i])),
             "note": "state is the heard-node set, at most one id per node in the network"}
json.dump(out, open(os.path.join(E.HERE, "..", "results", "complexity.json"), "w"), indent=1)
for k, v in out.items(): print(k, v)
