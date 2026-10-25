"""Re-run IJIES hard-tree vote versus averaged leaf-probability comparison.
Run from the repository root:
  python analysis/check_rf_vote_vs_proba.py
Uses 4 published fold splits, 5 held-out attacks, 15 observable features.
The stored October 2026 results were obtained under a different sklearn environment;
do not replace manuscript results with these values without full reconciliation.
"""
import json
import os
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import openset_eval as E

start = time.time()
data = E.load()
features = E.F_OBS
results = []
for fold, (train_seeds, val_seeds, test_seeds) in enumerate(E.FOLDS, 1):
    Xte, Mte, cte, _, _ = E.node_table(data[data.seed.isin(test_seeds)], features)
    for held_out in E.ATTACKS:
        Xtr, Mtr, ctr, _, _ = E.node_table(
            data[data.seed.isin(train_seeds) & (data.scenario != held_out)], features
        )
        scaler = StandardScaler().fit(Xtr.reshape(-1, len(features)))
        def transform(X, M):
            arr = scaler.transform(X.reshape(-1, len(features))).reshape(X.shape)
            return E.flat(arr, M)
        train, test = transform(Xtr, Mtr), transform(Xte, Mte)
        rf = RandomForestClassifier(
            n_estimators=300, random_state=0, class_weight="balanced", n_jobs=4
        )
        rf.fit(train, ctr)
        hard = 1 - E.vote(rf, test).max(1)
        leaf = 1 - rf.predict_proba(test).max(1)
        positives, negatives = cte == held_out, cte == "normal"
        y = np.r_[np.ones(positives.sum()), np.zeros(negatives.sum())]
        hard_auc = float(roc_auc_score(y, np.r_[hard[positives], hard[negatives]]))
        leaf_auc = float(roc_auc_score(y, np.r_[leaf[positives], leaf[negatives]]))
        results.append({
            "fold": fold, "attack": held_out,
            "hard_vote_auroc": hard_auc, "leaf_probability_auroc": leaf_auc,
            "delta": hard_auc - leaf_auc,
            "attack_nodes": int(positives.sum()), "normal_nodes": int(negatives.sum())
        })
        print(fold, held_out, round(hard_auc, 4), round(leaf_auc, 4), flush=True)
out = {
    "method": "300-tree RF, same held-out tests and 15 observable inputs",
    "environment": {"scikit_learn": __import__("sklearn").__version__},
    "elapsed_seconds": time.time() - start,
    "results": results,
}
dest = os.path.join(os.path.dirname(__file__), "..", "results", "reviewer_rf_comparison_rerun.json")
with open(dest, "w", encoding="utf8") as f:
    json.dump(out, f, indent=2)
print("Wrote", dest)
