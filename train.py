"""Train on UCI AI4I 2020 and rebuild the model plus evaluation artifacts."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import train_test_split

from inference import DATA, FEATURES, MODEL_COLUMNS, TARGET, prepare

SEED = 42
CONFIGURATIONS = [
    ("Cơ sở", {}),
    ("Trọng số lớp cân bằng", {"class_weight": "balanced"}),
]
SOURCE = "https://doi.org/10.24432/C5HS5C"


def evaluate(truth, scores, threshold):
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(truth)), "positive": int(sum(truth)),
        "ap": float(average_precision_score(truth, scores)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold(truth, scores):
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def main():
    data = pd.read_csv(DATA / "ai4i2020.csv")
    x, errors = prepare(data)
    if errors.ne("").any() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("Dữ liệu AI4I có giá trị không hợp lệ.")
    y = data[TARGET].astype(int)
    train, remainder = train_test_split(data.index, test_size=0.30, random_state=SEED, stratify=y)
    validation, test = train_test_split(remainder, test_size=0.50, random_state=SEED, stratify=y.loc[remainder])
    candidates = []
    models = []
    for name, params in CONFIGURATIONS:
        model = lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.035, max_depth=5, num_leaves=20,
            min_child_samples=22, reg_lambda=1.0, random_state=SEED,
            n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True, **params,
        )
        model.fit(x.loc[train], y.loc[train], categorical_feature=["product_type"])
        scores = model.booster_.predict(x.loc[validation], num_threads=1)
        threshold = select_threshold(y.loc[validation], scores)
        candidates.append({"configuration": name, "threshold": threshold,
                           **evaluate(y.loc[validation], scores, threshold)})
        models.append(model.booster_)
    chosen = max(range(len(candidates)), key=lambda i: (candidates[i]["ap"], candidates[i]["f1"]))
    model = models[chosen]
    threshold = candidates[chosen]["threshold"]
    groups = {"Train": train, "Validation": validation, "Test": test}
    metrics = {
        name: evaluate(y.loc[indices], model.predict(x.loc[indices], num_threads=1), threshold)
        for name, indices in groups.items()
    }
    model.save_model(str(DATA / "model.txt"))

    pd.DataFrame({
        "UDI": data.loc[validation, "UDI"], TARGET: y.loc[validation],
        "score": model.predict(x.loc[validation], num_threads=1),
    }).sort_values("UDI").to_csv(DATA / "validation_scores.csv", index=False)

    records = data.loc[test, ["UDI", *FEATURES, TARGET]].sort_values("UDI").copy()
    scores = model.predict(x.loc[records.index], num_threads=1)
    records["score"] = scores
    records["prediction"] = (scores >= threshold).astype(int)
    records.to_csv(DATA / "test_results.csv", index=False)
    precision, recall, _ = precision_recall_curve(records[TARGET], scores)
    pd.DataFrame({"Recall": recall[::-1], "Precision": precision[::-1]}).to_csv(DATA / "pr_curve.csv", index=False)
    pd.concat([pd.DataFrame({"UDI": data.loc[indices, "UDI"], "Tập": name})
               for name, indices in groups.items()]).sort_values("UDI").to_csv(DATA / "splits.csv", index=False)
    alert_idx = np.flatnonzero(records["prediction"].to_numpy() == 1)
    start = max(0, int(alert_idx[0]) - 2) if len(alert_idx) else 0
    records.iloc[start:start + 40][FEATURES].to_csv(DATA / "mau_du_doan.csv", index=False)
    metadata = {
        "seed": SEED, "source": SOURCE, "configuration": candidates[chosen]["configuration"],
        "threshold": threshold, "n": len(data), "positive": int(y.sum()), "demo_start": start,
        "metrics": metrics, "validation_comparison": candidates,
        "train_ranges": {name: [float(x.loc[train, name].min()), float(x.loc[train, name].max())]
                         for name in MODEL_COLUMNS[1:]},
        "versions": {"lightgbm": lgb.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    (DATA / "report.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
