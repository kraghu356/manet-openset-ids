#!/usr/bin/env python3
"""Open-set evaluation for the revised manuscript.

Every number comes from results/dataset2.csv.gz, one feature definition and
one fold scheme, so the paper can be reproduced from this repository alone.

Protocol (answers Reviewer A on data separation and threshold selection)
  * Four folds by simulation seed. In each fold 12 seeds train, 4 seeds
    validate and 4 seeds test; no run appears in more than one partition.
  * Leave one attack out: for held-out attack h, every run whose scenario is
    h is removed from training and validation, honest nodes included, so no
    information from h reaches any model, scaler, normaliser or threshold.
  * Validation data (known classes only) is used for neural model selection,
    for converting scores to percentile ranks in the hybrid detector, and for
    the operating threshold (95th percentile of validation-normal scores).
  * Test: nodes of class h are positives, normal nodes negatives. Nodes of the
    four known attacks are reported separately (they should not be called
    unknown, and an open-set detector that flags them is not wrong in
    deployment, but they are excluded from the unseen-vs-normal AUROC).

Methods, all scored so that a larger value means "more likely unknown"
  MSP          1 - max softmax probability (Hendrycks and Gimpel)
  Energy       -logsumexp(logits) (Liu et al.)
  OpenMax      Weibull-recalibrated unknown probability (Bendale and Boult)
  RF vote      1 - largest fraction of the 300 trees voting for one class
  RF entropy   entropy of the tree-vote distribution
  kNN dist     mean distance to the 5 nearest training nodes
  SVM margin   minus the largest one-vs-rest decision value
  Autoencoder  masked reconstruction error, trained on benign traffic only
  nu           unheard-neighbour count (protocol invariant)
  Hybrid       proposed: max(rank(RF vote), rank(AE)), with nu > 0 overriding

    python3 analysis/openset_eval.py            # full run
    python3 analysis/openset_eval.py --quick    # one fold, fewer epochs
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import weibull_min
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "results", "dataset2.csv.gz")
SEQ = 29
ATTACKS = ["bha", "flood", "gha", "sink", "worm"]

# 13 raw inputs + 4 moving averages = 17 (F-all). fwd_ok and fwd_drop are
# recorded but never used: fwd_drop is identically zero for honest nodes.
RAW = ["fwd_seen", "fwd_ratio", "rreq_recv", "rreq_sent", "no_route",
       "nb_count", "nb_churn", "rssi_min", "rssi_dist_max", "nb_no_rssi",
       "speed", "nb_mean_dist", "nb_max_dist"]
EWMA = ["fwd_ratio", "no_route", "rreq_sent", "nb_churn"]
GEOMETRY = ["nb_mean_dist", "nb_max_dist"]
F_ALL = RAW + [c + "_ewma" for c in EWMA]
F_OBS = [f for f in F_ALL if f not in GEOMETRY]
F_OBS_NO_NU = [f for f in F_OBS if f != "nb_no_rssi"]

# (train seeds, validation seeds, test seeds)
FOLDS = [(list(range(1, 13)), list(range(13, 17)), list(range(17, 21))),
         (list(range(5, 17)), list(range(17, 21)), list(range(1, 5))),
         (list(range(9, 21)), list(range(1, 5)), list(range(5, 9))),
         (list(range(1, 5)) + list(range(13, 21)), list(range(5, 9)), list(range(9, 13)))]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load():
    a = pd.read_csv(DATA).sort_values(["run", "node_id", "time_s"])
    for c in ["rreq_sent", "rreq_recv", "no_route", "fwd_seen"]:
        a[c] = np.log1p(a[c].clip(lower=0))
    g = a.groupby(["run", "node_id"])
    for c in EWMA:
        a[c + "_ewma"] = g[c].transform(lambda x: x.ewm(alpha=0.3).mean())
    return a


def node_table(df, feats):
    """Sequences for every node that was asked to forward at least once."""
    X, M, cls, nu, scen = [], [], [], [], []
    for _, q in df.groupby(["run", "node_id"], sort=False):
        if q.fwd_seen.sum() == 0:
            continue
        v = q[feats].to_numpy(np.float32)
        n = min(len(v), SEQ)
        p = np.zeros((SEQ, len(feats)), np.float32)
        m = np.zeros(SEQ, np.float32)
        p[:n] = v[:n]; m[:n] = 1.0
        X.append(p); M.append(m)
        cls.append(q.cls.iloc[0]); scen.append(q.scenario.iloc[0])
        nu.append(float(q.nb_no_rssi.max()))
    return (np.stack(X), np.stack(M), np.array(cls), np.array(nu), np.array(scen))


def flat(X, M):
    mm = M[:, :, None]
    return (X * mm).sum(1) / np.maximum(mm.sum(1), 1)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

class Seq(nn.Module):
    """CNN (Abdan and Seno design), LSTM (Venkatasubramanian design) or GRU."""

    def __init__(s, kind, f, n):
        super().__init__()
        s.kind = kind
        if kind == "cnn":
            s.body = nn.Sequential(nn.Conv1d(f, 64, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        elif kind == "lstm":
            s.body = nn.LSTM(f, 64, num_layers=2, batch_first=True, dropout=0.2)
        else:
            s.body = nn.GRU(f, 64, num_layers=2, batch_first=True, dropout=0.2)
        s.pen = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        s.out = nn.Linear(32, n)

    def features(s, x, m):
        o = (s.body(x.transpose(1, 2)).transpose(1, 2) if s.kind == "cnn"
             else s.body(x)[0])
        mm = m.unsqueeze(-1)
        return s.pen((o * mm).sum(1) / mm.sum(1).clamp(min=1.0))

    def forward(s, x, m):
        return s.out(s.features(x, m))


def train_seq(kind, X, M, y, Xv, Mv, yv, nc, ep, seed=0):
    """Train with inverse-frequency weights; keep the epoch with the best
    validation macro F1 (validation contains known classes only)."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = Seq(kind, X.shape[-1], nc)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    cnt = np.bincount(y, minlength=nc).astype(np.float32)
    lf = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (nc * np.maximum(cnt, 1))))
    Xt, yt, Mt = torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(M)
    Xvt, Mvt = torch.from_numpy(Xv), torch.from_numpy(Mv)
    best, state = -1.0, None
    for _ in range(ep):
        m.train()
        pm = torch.randperm(len(Xt))
        for i in range(0, len(pm), 256):
            j = pm[i:i + 256]
            opt.zero_grad(); lf(m(Xt[j], Mt[j]), yt[j]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            f1 = f1_score(yv, m(Xvt, Mvt).argmax(1).numpy(), average="macro")
        if f1 > best:
            best = f1; state = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    return m


class AE(nn.Module):
    def __init__(s, f, h=32):
        super().__init__()
        s.e = nn.GRU(f, h, batch_first=True)
        s.d = nn.GRU(h, h, batch_first=True)
        s.o = nn.Linear(h, f)

    def forward(s, x):
        _, hn = s.e(x)
        r = hn[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        return s.o(s.d(r)[0])


def train_ae(X, M, ep, seed=0):
    torch.manual_seed(seed)
    m = AE(X.shape[-1]); opt = torch.optim.Adam(m.parameters(), 1e-3)
    A, B = torch.from_numpy(X), torch.from_numpy(M)
    for _ in range(ep):
        m.train(); pm = torch.randperm(len(A))
        for i in range(0, len(pm), 256):
            j = pm[i:i + 256]; opt.zero_grad()
            r = m(A[j]); k = B[j].unsqueeze(-1)
            (((r - A[j]) ** 2 * k).sum() / k.sum().clamp(min=1)).backward(); opt.step()
    m.eval()
    return m


def recon(m, X, M):
    with torch.no_grad():
        r = m(torch.from_numpy(X)); k = torch.from_numpy(M).unsqueeze(-1)
        return ((((r - torch.from_numpy(X)) ** 2) * k).sum((1, 2))
                / k.sum((1, 2)).clamp(min=1)).numpy()


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------

def logits_and_features(m, X, M):
    with torch.no_grad():
        Xt, Mt = torch.from_numpy(X), torch.from_numpy(M)
        h = m.features(Xt, Mt)
        return m.out(h).numpy(), h.numpy()


def msp(lg):
    e = np.exp(lg - lg.max(1, keepdims=True))
    return 1.0 - (e / e.sum(1, keepdims=True)).max(1)


def energy(lg):
    mx = lg.max(1, keepdims=True)
    return -(mx[:, 0] + np.log(np.exp(lg - mx).sum(1)))


class OpenMax:
    """OpenMax (Bendale and Boult, 2016) on the logit vector.

    Per known class: mean activation vector (MAV) of correctly classified
    training nodes; a Weibull fitted to the `tail` largest euclidean-cosine
    distances to that MAV. At test time the top `alpha` classes are
    recalibrated and the removed mass becomes the unknown class.
    tail=20, alpha=min(3, classes) are the original paper's defaults and
    are fixed, not tuned on any held-out attack.
    """

    def __init__(s, tail=20, alpha=3):
        s.tail, s.alpha = tail, alpha

    @staticmethod
    def dist(av, mav):
        eu = np.linalg.norm(av - mav, axis=-1) / 200.0
        cos = 1 - (av @ mav) / (np.linalg.norm(av, axis=-1) * np.linalg.norm(mav) + 1e-12)
        return eu + cos

    def fit(s, lg, y):
        s.nc = lg.shape[1]; s.alpha = min(s.alpha, s.nc)
        pred = lg.argmax(1)
        s.mav, s.wb = [], []
        for c in range(s.nc):
            ok = lg[(y == c) & (pred == c)]
            if len(ok) < 3:
                ok = lg[y == c]
            mav = ok.mean(0)
            d = np.sort(s.dist(ok, mav))[-min(s.tail, len(ok)):]
            shape, loc, scale = weibull_min.fit(d, floc=0)
            s.mav.append(mav); s.wb.append((shape, loc, scale))
        return s

    def unknown_prob(s, lg):
        out = np.zeros(len(lg))
        ranks = np.argsort(-lg, 1)
        for i, v in enumerate(lg):
            w = np.ones(s.nc)
            for r in range(s.alpha):
                c = ranks[i, r]
                shape, loc, scale = s.wb[c]
                cdf = weibull_min.cdf(s.dist(v[None], s.mav[c])[0], shape, loc, scale)
                w[c] = 1 - ((s.alpha - r) / s.alpha) * cdf
            rv = v * w
            unk = np.sum(v * (1 - w))
            z = np.concatenate([rv, [unk]])
            z = np.exp(z - z.max())
            out[i] = z[-1] / z.sum()
        return out


def vote(rf, F):
    """Fraction of trees voting for each class (hard votes, not averaged
    leaf probabilities)."""
    votes = np.stack([t.predict(F) for t in rf.estimators_], 1).astype(int)
    frac = np.stack([(votes == c).mean(1) for c in range(len(rf.classes_))], 1)
    return frac


def entropy(p):
    q = np.clip(p, 1e-12, 1)
    return -(q * np.log(q)).sum(1)


def rank_against(ref, s):
    """Empirical CDF of s against reference scores (validation normal)."""
    ref = np.sort(ref)
    return np.searchsorted(ref, s, side="right") / len(ref)


def auroc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = np.r_[pos, neg]
    if len(np.unique(sc)) == 1:
        return None
    return float(roc_auc_score(y, sc))


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------

def run(quick=False):
    a = load()
    folds = FOLDS[:1] if quick else FOLDS
    ep_nn, ep_ae = (6, 8) if quick else (30, 40)
    feats = F_OBS
    res = {"protocol": {"features": feats, "n_features": len(feats),
                        "folds": [{"train": f[0], "val": f[1], "test": f[2]} for f in folds],
                        "epochs_classifier": ep_nn, "epochs_autoencoder": ep_ae,
                        "threshold": "95th percentile of validation-normal score"},
           "open_set": {}, "closed_set": {}, "autoencoder_ablation": {}, "timing": {}}
    T = {}

    # ---------------- closed set, four folds ----------------
    print("closed set", flush=True)
    cl = sorted(a.cls.unique()); ci = {c: i for i, c in enumerate(cl)}
    closed = {}
    for fi, (trs, vas, tes) in enumerate(folds):
        Xtr, Mtr, ctr, _, _ = node_table(a[a.seed.isin(trs)], feats)
        Xva, Mva, cva, _, _ = node_table(a[a.seed.isin(vas)], feats)
        Xte, Mte, cte, _, _ = node_table(a[a.seed.isin(tes)], feats)
        ytr, yva, yte = (np.array([ci[c] for c in z]) for z in (ctr, cva, cte))
        sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
        f = lambda X: sc.transform(X.reshape(-1, len(feats))).reshape(X.shape).astype(np.float32)
        Xtr, Xva, Xte = f(Xtr), f(Xva), f(Xte)
        Ftr, Fte = flat(Xtr, Mtr), flat(Xte, Mte)
        for nm, mk in [("decision_tree", lambda: DecisionTreeClassifier(random_state=0, class_weight="balanced")),
                       ("knn", lambda: KNeighborsClassifier(5)),
                       ("svm", lambda: SVC(class_weight="balanced", random_state=0)),
                       ("random_forest", lambda: RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=-1))]:
            p = mk().fit(Ftr, ytr).predict(Fte)
            closed.setdefault(nm, []).append(f1_score(yte, p, average="macro"))
        for nm in ["cnn", "lstm", "gru"]:
            m = train_seq(nm, Xtr, Mtr, ytr, Xva, Mva, yva, len(cl), ep_nn)
            lg, _ = logits_and_features(m, Xte, Mte)
            closed.setdefault(nm, []).append(f1_score(yte, lg.argmax(1), average="macro"))
        print(f"  fold {fi + 1}: " + " ".join(f"{k}={v[-1]:.3f}" for k, v in closed.items()), flush=True)
    res["closed_set"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                             "folds": [round(x, 4) for x in v]} for k, v in closed.items()}

    # ---------------- open set ----------------
    methods = ["msp_cnn", "msp_lstm", "msp_gru", "energy_cnn", "energy_lstm", "energy_gru",
               "openmax_cnn", "openmax_lstm", "openmax_gru", "rf_vote", "rf_entropy",
               "knn_dist", "svm_margin", "autoencoder", "nu",
               "hybrid_rf_ae", "hybrid_rf_nu", "hybrid_ae_nu", "hybrid"]
    store = {h: {m: {"auroc": [], "tpr_at_val95": [], "fpr_at_val95": [],
                     "known_attacks_flagged": []} for m in methods} for h in ATTACKS}
    loao_assign = {h: [] for h in ATTACKS}

    for fi, (trs, vas, tes) in enumerate(folds):
        print(f"open set, fold {fi + 1}", flush=True)
        # autoencoder: benign nodes of benign runs, train seeds; independent of h
        t0 = time.time()
        Xn, Mn, _, _, _ = node_table(a[a.seed.isin(trs) & (a.cls == "normal") & (a.scenario == "normal")], feats)
        sca = StandardScaler().fit(Xn.reshape(-1, len(feats)))
        fa = lambda X: sca.transform(X.reshape(-1, len(feats))).reshape(X.shape).astype(np.float32)
        ae = train_ae(fa(Xn), Mn, ep_ae)
        T.setdefault("autoencoder_train_s", []).append(time.time() - t0)

        Xte_raw, Mte, cte, nute, scte = node_table(a[a.seed.isin(tes)], feats)
        ae_te = recon(ae, fa(Xte_raw), Mte)

        for h in ATTACKS:
            known = sorted([c for c in ATTACKS if c != h] + ["normal"])
            ki = {c: i for i, c in enumerate(known)}
            tr_df = a[a.seed.isin(trs) & (a.scenario != h)]
            va_df = a[a.seed.isin(vas) & (a.scenario != h)]
            Xtr, Mtr, ctr, _, _ = node_table(tr_df, feats)
            Xva, Mva, cva, nuva, _ = node_table(va_df, feats)
            ytr = np.array([ki[c] for c in ctr]); yva = np.array([ki[c] for c in cva])
            sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
            f = lambda X: sc.transform(X.reshape(-1, len(feats))).reshape(X.shape).astype(np.float32)
            Xtr_s, Xva_s, Xte_s = f(Xtr), f(Xva), f(Xte_raw)
            Ftr, Fva, Fte = flat(Xtr_s, Mtr), flat(Xva_s, Mva), flat(Xte_s, Mte)

            pos = cte == h; neg = cte == "normal"
            kn_att = np.isin(cte, [c for c in ATTACKS if c != h])
            vneg = cva == "normal"
            S_te, S_va = {}, {}

            # random forest: vote disagreement and entropy
            t0 = time.time()
            rf = RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=-1).fit(Ftr, ytr)
            T.setdefault("rf_train_s", []).append(time.time() - t0)
            t0 = time.time(); vt = vote(rf, Fte); T.setdefault("rf_score_s_per_node", []).append((time.time() - t0) / len(Fte))
            vv = vote(rf, Fva)
            S_te["rf_vote"], S_va["rf_vote"] = 1 - vt.max(1), 1 - vv.max(1)
            S_te["rf_entropy"], S_va["rf_entropy"] = entropy(vt), entropy(vv)
            rf_pred_te = vt.argmax(1)

            kn = KNeighborsClassifier(5).fit(Ftr, ytr)
            S_te["knn_dist"] = kn.kneighbors(Fte)[0].mean(1)
            S_va["knn_dist"] = kn.kneighbors(Fva)[0].mean(1)
            sv = SVC(class_weight="balanced", random_state=0).fit(Ftr, ytr)
            S_te["svm_margin"] = -sv.decision_function(Fte).max(1)
            S_va["svm_margin"] = -sv.decision_function(Fva).max(1)

            for nm in ["cnn", "lstm", "gru"]:
                t0 = time.time()
                m = train_seq(nm, Xtr_s, Mtr, ytr, Xva_s, Mva, yva, len(known), ep_nn)
                T.setdefault(f"{nm}_train_s", []).append(time.time() - t0)
                lg_tr, _ = logits_and_features(m, Xtr_s, Mtr)
                t0 = time.time(); lg_te, _ = logits_and_features(m, Xte_s, Mte)
                T.setdefault(f"{nm}_score_s_per_node", []).append((time.time() - t0) / len(Xte_s))
                lg_va, _ = logits_and_features(m, Xva_s, Mva)
                S_te[f"msp_{nm}"], S_va[f"msp_{nm}"] = msp(lg_te), msp(lg_va)
                S_te[f"energy_{nm}"], S_va[f"energy_{nm}"] = energy(lg_te), energy(lg_va)
                om = OpenMax().fit(lg_tr, ytr)
                S_te[f"openmax_{nm}"], S_va[f"openmax_{nm}"] = om.unknown_prob(lg_te), om.unknown_prob(lg_va)
                if nm == "gru":
                    ass = pd.Series(np.array(known)[lg_te[pos].argmax(1)]).value_counts()
                    loao_assign[h].append(ass.to_dict())

            ae_va = recon(ae, fa(node_table(va_df, feats)[0]), Mva)
            S_te["autoencoder"], S_va["autoencoder"] = ae_te, ae_va
            S_te["nu"], S_va["nu"] = nute, nuva

            # proposed hybrid: percentile ranks against validation normal only
            r_rf_te = rank_against(S_va["rf_vote"][vneg], S_te["rf_vote"])
            r_ae_te = rank_against(S_va["autoencoder"][vneg], S_te["autoencoder"])
            r_rf_va = rank_against(S_va["rf_vote"][vneg], S_va["rf_vote"])
            r_ae_va = rank_against(S_va["autoencoder"][vneg], S_va["autoencoder"])
            ov_te, ov_va = (nute > 0) * (1.0 + nute), (nuva > 0) * (1.0 + nuva)
            S_te["hybrid_rf_ae"], S_va["hybrid_rf_ae"] = np.maximum(r_rf_te, r_ae_te), np.maximum(r_rf_va, r_ae_va)
            S_te["hybrid_rf_nu"], S_va["hybrid_rf_nu"] = np.maximum(r_rf_te, ov_te), np.maximum(r_rf_va, ov_va)
            S_te["hybrid_ae_nu"], S_va["hybrid_ae_nu"] = np.maximum(r_ae_te, ov_te), np.maximum(r_ae_va, ov_va)
            S_te["hybrid"] = np.maximum(np.maximum(r_rf_te, r_ae_te), ov_te)
            S_va["hybrid"] = np.maximum(np.maximum(r_rf_va, r_ae_va), ov_va)

            for mth in methods:
                st, sv_ = S_te[mth], S_va[mth]
                thr = np.quantile(sv_[vneg], 0.95)
                d = store[h][mth]
                d["auroc"].append(auroc(st[pos], st[neg]))
                d["tpr_at_val95"].append(float((st[pos] > thr).mean()))
                d["fpr_at_val95"].append(float((st[neg] > thr).mean()))
                d["known_attacks_flagged"].append(float((st[kn_att] > thr).mean()) if kn_att.any() else None)

            line = " ".join(f"{k}={store[h][k]['auroc'][-1]:.3f}" if store[h][k]['auroc'][-1] is not None
                            else f"{k}=undef" for k in ["msp_gru", "energy_gru", "openmax_gru", "rf_vote", "autoencoder", "nu", "hybrid"])
            print(f"  {h}: n_pos={int(pos.sum())} {line}", flush=True)

    def summ(v):
        v = [x for x in v if x is not None]
        return {"mean": float(np.mean(v)) if v else None, "sd": float(np.std(v)) if v else None,
                "folds": [round(x, 4) for x in v]}

    res["open_set"] = {h: {m: {k: summ(v) for k, v in store[h][m].items()} for m in methods} for h in ATTACKS}
    res["gru_assignment_of_unseen"] = loao_assign
    res["timing"] = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in T.items()}

    # ---------------- autoencoder feature-set ablation ----------------
    print("autoencoder ablation", flush=True)
    for nm, FF in [("F_all", F_ALL), ("F_observable", F_OBS), ("F_observable_minus_nu", F_OBS_NO_NU)]:
        per = {c: [] for c in ATTACKS}
        for trs, vas, tes in folds:
            Xn, Mn, _, _, _ = node_table(a[a.seed.isin(trs) & (a.cls == "normal") & (a.scenario == "normal")], FF)
            s2 = StandardScaler().fit(Xn.reshape(-1, len(FF)))
            g = lambda X: s2.transform(X.reshape(-1, len(FF))).reshape(X.shape).astype(np.float32)
            m = train_ae(g(Xn), Mn, ep_ae)
            Xt, Mt, ct, _, _ = node_table(a[a.seed.isin(tes)], FF)
            e = recon(m, g(Xt), Mt)
            for c in ATTACKS:
                if (ct == c).any():
                    per[c].append(auroc(e[ct == c], e[ct == "normal"]))
        res["autoencoder_ablation"][nm] = {c: summ(v) for c, v in per.items()}
        print(f"  {nm}: " + " ".join(f"{c}={np.mean(v):.3f}" for c, v in per.items() if v), flush=True)

    out = os.path.join(HERE, "..", "results", "openset_results_quick.json" if quick else "openset_results.json")
    json.dump(res, open(out, "w"), indent=1)
    print("written", os.path.abspath(out))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(**vars(ap.parse_args()))
