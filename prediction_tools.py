"""Prediction support shared by the interactive screen and offline preparation."""
from functools import lru_cache
from io import BytesIO
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from inference import CATEGORIES, DATA, FEATURES, MODEL_COLUMNS, prepare

NUMERIC = MODEL_COLUMNS[1:]


def encoded(frame):
    """Reuse the base feature validation before encoding the comparison models."""
    raw, errors = prepare(frame)
    if errors.ne("").any():
        raise ValueError(errors.loc[errors.ne("")].iloc[0])
    product = raw["product_type"].to_numpy()
    one_hot = np.column_stack([product == level for level in CATEGORIES])
    return np.column_stack((raw[NUMERIC].to_numpy(), one_hot))


@lru_cache(maxsize=1)
def comparison_models():
    return joblib.load(DATA / "comparison_models.joblib")


def compare_record(frame, lightgbm_score, threshold):
    rows = [("LightGBM", lightgbm_score, threshold)]
    matrix = encoded(frame)
    for entry in comparison_models():
        score = float(entry["model"].predict_proba(matrix)[0, 1])
        rows.append((entry["name"], score, entry["threshold"]))
    return pd.DataFrame([
        {"Mô hình": name, "Điểm": score, "Ngưỡng": cutoff, "Cảnh báo": int(score >= cutoff)}
        for name, score, cutoff in rows
    ])


@lru_cache(maxsize=1)
def reference_data():
    raw = pd.read_csv(DATA / "ai4i2020.csv")
    split = pd.read_csv(DATA / "splits.csv")
    train_ids = split.loc[split["Tập"].eq("Train"), "UDI"]
    train = raw.loc[raw["UDI"].isin(train_ids)].reset_index(drop=True)
    matrix = train[FEATURES[1:]].to_numpy(float)
    scale = matrix.std(axis=0)
    return train, matrix, np.where(scale > 0, scale, 1)


def neighbors(frame, limit=3):
    raw, errors = prepare(frame)
    if len(raw) != 1 or errors.ne("").any():
        raise ValueError("Cần một bản ghi hợp lệ để đối chiếu.")
    train, matrix, scale = reference_data()
    indices = np.flatnonzero(train["Type"].eq(str(raw["product_type"].iloc[0])))
    distances = np.linalg.norm((matrix[indices] - raw[NUMERIC].to_numpy()[0]) / scale, axis=1)
    chosen = np.argsort(distances, kind="stable")[:limit]
    found = train.iloc[indices[chosen]][["UDI", *FEATURES, "Machine failure"]].copy()
    found["Khoảng cách chuẩn hóa"] = distances[chosen]
    return found


def range_issues(frame, limits):
    raw, _ = prepare(frame)
    issues = []
    for name, key in zip(FEATURES[1:], NUMERIC):
        value = float(raw.iloc[0][key])
        low, high = limits[key]
        if not low <= value <= high:
            issues.append({"Thông số": name, "Giá trị": value, "Khoảng train": f"{low:g} – {high:g}"})
    return issues


def read_paste(text):
    text = text.strip()
    if not text:
        raise ValueError("Dán các dòng từ bảng tính vào ô phía trên.")
    if len(text.encode("utf-8")) > 8 * 1024 * 1024:
        raise ValueError("Nội dung dán tối đa 8 MB.")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    first = text.splitlines()[0].split(delimiter)
    if len(first) == len(FEATURES) and first[0].strip().upper() in CATEGORIES:
        return pd.read_csv(BytesIO(text.encode("utf-8")), sep=delimiter, header=None, names=FEATURES, nrows=20001)
    return pd.read_csv(BytesIO(text.encode("utf-8")), sep=delimiter, nrows=20001)


def excel_reader(payload):
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError("Tệp Excel tối đa 8 MB.")
    with ZipFile(BytesIO(payload)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
            raise ValueError("Tệp Excel giải nén vượt giới hạn 64 MB.")
    return pd.ExcelFile(BytesIO(payload), engine="openpyxl")
