"""Shared feature contract for training, CSV validation and prediction."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARGET = "Machine failure"
FEATURES = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
NAMES = dict(zip(FEATURES, [
    "Loại sản phẩm", "Nhiệt độ không khí", "Nhiệt độ quá trình",
    "Tốc độ quay", "Mô-men xoắn", "Độ mòn dao",
]))
MODEL_COLUMNS = ["product_type", "air_temp", "process_temp", "speed", "torque", "tool_wear"]
CATEGORIES = ["L", "M", "H"]
SCORE = "Điểm mô hình"
ALERT = "Cảnh báo"
ERROR = "Lỗi đầu vào"
CUTOFF = "Vượt ngưỡng"
CLEAR = "Dưới ngưỡng"
MAX_ROWS = 20_000


def normalized(name):
    return " ".join(str(name).strip().lstrip("\ufeff").split()).casefold()


def prepare(frame):
    """Return model features and row-level input errors without changing row order."""
    groups = {}
    for column in frame.columns:
        groups.setdefault(normalized(column), []).append(column)
    absent = [name for name in FEATURES if normalized(name) not in groups]
    if absent:
        raise ValueError("Thiếu cột: " + ", ".join(absent))
    ambiguous = [
        name for name in FEATURES
        if len(groups[normalized(name)]) != 1
        or any(key.startswith(normalized(name) + ".") and key.rsplit(".", 1)[-1].isdigit()
               for key in groups)
    ]
    if ambiguous:
        raise ValueError("Trùng tên cột: " + ", ".join(ambiguous))
    raw = frame[[groups[normalized(name)][0] for name in FEATURES]].copy()
    raw.columns = MODEL_COLUMNS
    errors = pd.Series("", index=frame.index, dtype=object)
    types = raw["product_type"].astype("string").str.strip().str.upper()
    errors.loc[~types.isin(CATEGORIES)] = "Type phải là L, M hoặc H"
    raw["product_type"] = pd.Categorical(types, categories=CATEGORIES)
    for original, model_name in zip(FEATURES[1:], MODEL_COLUMNS[1:]):
        numbers = pd.to_numeric(raw[model_name], errors="coerce").astype(float)
        invalid = ~np.isfinite(numbers) | (numbers < 0)
        if model_name in {"air_temp", "process_temp"}:
            invalid |= numbers <= 0
        errors.loc[invalid & errors.eq("")] = f"Kiểm tra giá trị và đơn vị của {original}"
        raw[model_name] = numbers
    return raw, errors


def load_model():
    return lgb.Booster(model_file=str(DATA / "model.txt"))


def score_batch(frame, model, threshold, ranges=None):
    raw, errors = prepare(frame)
    valid = errors.eq("")
    output = frame.copy()
    output[SCORE] = np.nan
    output[ALERT] = "Không dự đoán"
    output[ERROR] = errors
    if ranges:
        outside = pd.Series(False, index=frame.index)
        for model_name, limits in ranges.items():
            outside |= ~raw[model_name].between(limits[0], limits[1])
        output["Ngoài khoảng train"] = valid & outside
    if valid.any():
        scores = model.predict(raw.loc[valid], num_threads=1)
        output.loc[valid, SCORE] = scores
        output.loc[valid, ALERT] = np.where(scores >= threshold, CUTOFF, CLEAR)
    return output


def explain(frame, model):
    prepared, errors = prepare(frame)
    if len(prepared) != 1 or errors.ne("").any():
        raise ValueError("Giải thích cần đúng một bản ghi hợp lệ.")
    contributions = np.asarray(model.predict(prepared, pred_contrib=True, num_threads=1))[0, :-1]
    return pd.DataFrame({
        "Thông số": list(NAMES.values()),
        "Đóng góp vào điểm thô": contributions,
    }).sort_values("Đóng góp vào điểm thô", key=lambda values: values.abs(), ascending=False)
