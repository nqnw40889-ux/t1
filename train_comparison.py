"""Rebuild the two comparison models and the validation-based distance reference."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inference import DATA, TARGET
from prediction_tools import encoded, neighbors
from train import SEED, select_threshold


def main():
    raw = pd.read_csv(DATA / "ai4i2020.csv").set_index("UDI")
    split = pd.read_csv(DATA / "splits.csv")
    train = raw.loc[split.loc[split["Tập"].eq("Train"), "UDI"]]
    validation = raw.loc[split.loc[split["Tập"].eq("Validation"), "UDI"]]
    train_x, valid_x = encoded(train), encoded(validation)
    fitted = []
    for name, model in [
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=3,
                                                 class_weight="balanced_subsample", random_state=SEED, n_jobs=2)),
        ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", random_state=SEED, max_iter=1000))),
    ]:
        model.fit(train_x, train[TARGET])
        scores = model.predict_proba(valid_x)[:, 1]
        fitted.append({"name": name, "model": model, "threshold": select_threshold(validation[TARGET], scores)})
    joblib.dump(fitted, DATA / "comparison_models.joblib", compress=3)
    cutoffs = {}
    for label, samples in validation.groupby("Type"):
        distances = [float(neighbors(samples.iloc[[i]], limit=1)["Khoảng cách chuẩn hóa"].iloc[0])
                     for i in range(len(samples))]
        cutoffs[label] = float(np.quantile(distances, .99))
    (DATA / "reference_distances.json").write_text(json.dumps(cutoffs), encoding="utf-8")
    print("Saved two comparison models and validation distance cutoffs.", cutoffs)


if __name__ == "__main__":
    main()
