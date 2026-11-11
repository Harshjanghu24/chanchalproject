"""CDAS Step 2: Ridge / Random Forest / XGBoost on Model A and Model B.
Grouped split by user_id, grouped CV, grouped bootstrap CIs, permutation importance,
partial dependence, and a CM x L interaction test. Correlational only."""
import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

import data_prep as dp

warnings.filterwarnings("ignore")
SEED = 42
DEVICE = "cpu"  # set to "cuda" to try the GPU for XGBoost (no real benefit at this data size)
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def make_pre(num, cat):
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def group_bootstrap(y, p, g, n=1000, seed=SEED):
    """95% CIs, resampling whole users (groups) so repeated users stay together."""
    rng = np.random.default_rng(seed)
    g = np.asarray(g)
    uniq = np.unique(g)
    idx_map = {u: np.where(g == u)[0] for u in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_map[u] for u in pick])
        out.append((r2_score(y[idx], p[idx]),
                    mean_absolute_error(y[idx], p[idx]),
                    rmse(y[idx], p[idx])))
    out = np.array(out)
    lo, hi = np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)
    return lo, hi


def run_analysis(name, df, num, cat):
    print(f"\n{'=' * 60}\nAnalysis {name}: n={len(df)}, users={df['user_id'].nunique()}")
    feats = num + cat
    # Leakage guard: no A-item may ever be a feature
    assert not any(re.fullmatch(r"A\d", c) for c in feats), "A-item in features!"
    assert "anxiety" not in feats

    X, y, g = df[feats], df["anxiety"].to_numpy(), df["user_id"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, g))
    Xtr, Xte, ytr, yte, gtr, gte = X.iloc[tr], X.iloc[te], y[tr], y[te], g[tr], g[te]
    assert len(set(gtr) & set(gte)) == 0, "user in both train and test!"
    print(f"train={len(tr)} test={len(te)} (no user overlap: OK)")

    cv = GroupKFold(n_splits=5)
    models = {
        "Baseline(mean)": Pipeline([("pre", make_pre(num, cat)), ("m", DummyRegressor())]),
        "Ridge": GridSearchCV(
            Pipeline([("pre", make_pre(num, cat)), ("m", Ridge())]),
            {"m__alpha": [0.1, 1, 10, 100]}, cv=cv, scoring="r2"),
        "RandomForest": GridSearchCV(
            Pipeline([("pre", make_pre(num, cat)),
                      ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1))]),
            {"m__max_depth": [4, 8, None], "m__min_samples_leaf": [5, 15]},
            cv=cv, scoring="r2"),
        "XGBoost": GridSearchCV(
            Pipeline([("pre", make_pre(num, cat)),
                      ("m", XGBRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8,
                                         tree_method="hist", device=DEVICE, random_state=SEED))]),
            {"m__max_depth": [2, 3, 4]}, cv=cv, scoring="r2"),
    }

    rows, fitted = [], {}
    for mname, model in models.items():
        if isinstance(model, GridSearchCV):
            model.fit(Xtr, ytr, groups=gtr)
            best = model.best_estimator_
            extra = model.best_params_
        else:
            model.fit(Xtr, ytr)
            best, extra = model, {}
        p = best.predict(Xte)
        lo, hi = group_bootstrap(yte, p, gte)
        r2, mae, rm = r2_score(yte, p), mean_absolute_error(yte, p), rmse(yte, p)
        rows.append(dict(analysis=name, model=mname, n_train=len(tr), n_test=len(te),
                         r2=r2, r2_lo=lo[0], r2_hi=hi[0], mae=mae, mae_lo=lo[1], mae_hi=hi[1],
                         rmse=rm, rmse_lo=lo[2], rmse_hi=hi[2], best_params=str(extra)))
        fitted[mname] = best
        flag = "  <-- CHECK FOR LEAKAGE" if r2 > 0.8 else ""
        print(f"{mname:<15} R2={r2:.3f} [{lo[0]:.3f},{hi[0]:.3f}]  MAE={mae:.3f}  RMSE={rm:.3f}{flag}")

    # Permutation importance + partial dependence on the Random Forest
    rf = fitted["RandomForest"]
    pi = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=SEED, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=feats).sort_values()
    plt.figure(figsize=(7, 5))
    imp.plot.barh()
    plt.xlabel("Drop in R2 when shuffled (test set)")
    plt.title(f"Random Forest permutation importance, Model {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"perm_importance_{name}.png"), dpi=150)
    plt.close()
    print("Top features:", imp.sort_values(ascending=False).head(5).round(4).to_dict())

    top3 = [f for f in imp.sort_values(ascending=False).index if f in num][:3]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    PartialDependenceDisplay.from_estimator(rf, Xte, top3, ax=ax)
    fig.suptitle(f"Partial dependence (association, not causation), Model {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"pdp_{name}.png"), dpi=150)
    plt.close(fig)

    # Save RF for the simulator
    ranges = {c: (float(df[c].min()), float(df[c].max()), float(df[c].median())) for c in num}
    joblib.dump({"models": fitted, "num": num, "cat": cat, "ranges": ranges,
                 "cat_levels": {c: sorted(df[c].unique().tolist()) for c in cat}},
                os.path.join(RESULTS, f"models_{name}.joblib"))
    return rows


def interaction_test(df):
    """CM x L interaction (career motivation x literacy), Model A, cluster-robust SEs."""
    d = df.copy()
    d["CM_c"], d["L_c"] = d["CM"] - d["CM"].mean(), d["L"] - d["L"].mean()
    codes = pd.factorize(d["user_id"])[0]
    fit = smf.ols("anxiety ~ CM_c * L_c + RE + R + SG + Age + Grade + AI_hours", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": codes})
    print("\nInteraction test (Model A, cluster-robust by user_id):")
    print(fit.summary().tables[1])
    return fit


if __name__ == "__main__":
    raw = dp.load()
    A, B = dp.model_a(raw), dp.model_b(raw)
    rows = run_analysis("A", A, dp.NUM_A, dp.CAT) + run_analysis("B", B, dp.NUM_B, dp.CAT)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "metrics.csv"), index=False)
    print("\n", out[["analysis", "model", "n_test", "r2", "r2_lo", "r2_hi", "mae", "rmse"]]
          .round(3).to_string(index=False))
    interaction_test(A)
    print(f"\nSaved to {RESULTS}")