"""CDAS Step 3: repeated grouped CV (more stable than one 80/20 split).
Out-of-sample R2 is measured against a train-mean baseline, folds split by user_id."""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import data_prep as dp
from train_models import make_pre, RESULTS

warnings.filterwarnings("ignore")
SEED, REPS, K = 42, 10, 5


def oos_r2(X, y, g, make_model):
    rng = np.random.default_rng(SEED)
    users = np.unique(g)
    scores = []
    for _ in range(REPS):
        perm = rng.permutation(users)
        fold_of = {u: i % K for i, u in enumerate(perm)}
        folds = np.array([fold_of[u] for u in g])
        pred, base = np.zeros(len(y)), np.zeros(len(y))
        for f in range(K):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            m = make_model()
            m.fit(X.iloc[tr], y[tr])
            pred[te] = m.predict(X.iloc[te])
            base[te] = y[tr].mean()
        scores.append(1 - ((y - pred) ** 2).sum() / ((y - base) ** 2).sum())
    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    raw = dp.load()
    ycol = dp._find(raw.columns, "year")
    raw["year_num"] = pd.to_numeric(raw[ycol], errors="coerce")

    frames = {
        "RE only (A rows)": (dp.model_a(raw), ["RE"], []),
        "Model A": (dp.model_a(raw), dp.NUM_A, dp.CAT),
        "Model A + year": (raw[dp.NUM_A + ["year_num"] + dp.CAT + ["anxiety", "user_id"]]
                           .dropna().reset_index(drop=True), dp.NUM_A + ["year_num"], dp.CAT),
        "Model B": (dp.model_b(raw), dp.NUM_B, dp.CAT),
    }
    makers = {
        "Ridge": lambda n, c: Pipeline([("pre", make_pre(n, c)), ("m", Ridge(alpha=10))]),
        "RandomForest": lambda n, c: Pipeline([("pre", make_pre(n, c)), ("m", RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=15, random_state=SEED, n_jobs=-1))]),
        "XGBoost": lambda n, c: Pipeline([("pre", make_pre(n, c)), ("m", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8,
            tree_method="hist", random_state=SEED))]),
    }

    rows = []
    for fname, (d, num, cat) in frames.items():
        X, y, g = d[num + cat], d["anxiety"].to_numpy(), d["user_id"].to_numpy()
        for mname, mk in makers.items():
            m, s = oos_r2(X, y, g, lambda: mk(num, cat))
            rows.append(dict(features=fname, model=mname, n=len(d), r2_mean=m, r2_sd=s))
            print(f"{fname:<18} {mname:<13} n={len(d):<5} out-of-sample R2 = {m:.3f} (sd {s:.3f})")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "cv_repeated.csv"), index=False)
    print("\n", out.round(3).to_string(index=False))