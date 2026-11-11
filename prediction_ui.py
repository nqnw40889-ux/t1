"""Interactive one-record prediction, comparison, and a printable result."""
import html
import json

import numpy as np
import pandas as pd
import streamlit as st

from card_ui import matching_test_label
from inference import DATA, ERROR, FEATURES, NAMES, SCORE, explain, score_batch
from prediction_tools import compare_record, neighbors, range_issues

UNITS = ("", "K", "K", "rpm", "N·m", "phút")
STEPS = (0.1, 0.1, 1.0, 0.1, 1.0)


def restore_form(record):
    st.session_state["single_udi"] = int(record["UDI"])
    for i, name in enumerate(FEATURES):
        st.session_state[f"single_{i}"] = str(record[name]) if i == 0 else float(record[name])
    st.session_state.pop("single_result", None)


def prediction_status(score, threshold):
    return "Cảnh báo hỏng hóc" if score >= threshold else "Dưới ngưỡng cảnh báo"


def printable_result(values, score, threshold, conclusion):
    rows = "".join(f"<tr><th>{html.escape(NAMES[key])}</th><td>{html.escape(str(value))} "
                   f"{html.escape(unit)}</td></tr>" for (key, value), unit in zip(values.items(), UNITS))
    return f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>Phiếu dự đoán AI4I</title>
<style>body{{font:16px system-ui;max-width:700px;margin:3rem auto;padding:1rem;color:#112537}}
th,td{{border-bottom:1px solid #ccd7df;text-align:left;padding:.65rem 1rem}} table{{width:100%}}</style>
<h1>Phiếu kết quả dự đoán AI4I</h1><h2>{html.escape(conclusion)}</h2>
<p>LightGBM · Điểm {score:.4f} · Ngưỡng {threshold:.4f}</p><table>{rows}</table>
<p>Điểm mô hình mô tả bản ghi đầu vào. AI4I là dữ liệu tổng hợp.</p></html>""".encode("utf-8")


def single_prediction_page(model, metadata, test):
    st.subheader("Dự đoán một bản ghi")
    if "single_0" not in st.session_state:
        restore_form(test.loc[test["UDI"].eq(243)].iloc[0])
    picker, random_box = st.columns([2, 1])
    sample = picker.selectbox("Mẫu test theo mã UDI", test["UDI"].tolist(),
                              format_func=lambda udi: f"UDI {udi:05d}")
    left, right = random_box.columns(2)
    left.button("Nạp mẫu", width="stretch", on_click=restore_form,
                args=(test.loc[test["UDI"].eq(sample)].iloc[0],))
    if right.button("Mẫu ngẫu nhiên", width="stretch"):
        restore_form(test.sample().iloc[0])
    st.caption("Chọn mẫu có sẵn hoặc chỉnh các số đo. Mã UDI chỉ dùng để đối chiếu bản ghi.")
    with st.form("prediction_form"):
        fields = st.columns(3)
        for i, (name, unit) in enumerate(zip(FEATURES, UNITS)):
            column = fields[i % 3]
            if i == 0:
                column.selectbox(NAMES[name], ["L", "M", "H"], key="single_0")
            else:
                column.number_input(f"{NAMES[name]} ({unit})", min_value=0.0, step=STEPS[i - 1],
                                    format="%.2f", key=f"single_{i}")
        submitted = st.form_submit_button("Chạy dự đoán", type="primary", width="stretch")
    if submitted:
        values = {name: st.session_state[f"single_{i}"] for i, name in enumerate(FEATURES)}
        output = score_batch(pd.DataFrame([values]), model, metadata["threshold"], metadata["train_ranges"]).iloc[0]
        if output[ERROR]:
            st.session_state.pop("single_result", None)
            st.error(output[ERROR])
        else:
            st.session_state["single_result"] = {
                "values": values, "score": float(output[SCORE]), "udi": st.session_state["single_udi"],
            }
            st.session_state["reveal_truth"] = False

    result = st.session_state.get("single_result")
    if result is None:
        return
    values, score = result["values"], result["score"]
    frame = pd.DataFrame([values])
    threshold = metadata["threshold"]
    matches = neighbors(frame)
    cutoff = json.loads((DATA / "reference_distances.json").read_text(encoding="utf-8"))[values["Type"]]
    unusual = float(matches["Khoảng cách chuẩn hóa"].iloc[0]) > cutoff
    issues = range_issues(frame, metadata["train_ranges"])
    uncertain = bool(issues) or unusual
    conclusion = "Chưa đủ cơ sở để kết luận" if uncertain else prediction_status(score, threshold)
    if uncertain or score >= threshold:
        st.warning(conclusion)
    else:
        st.success(conclusion)
    a, b, c = st.columns(3)
    a.metric("Điểm LightGBM", f"{score:.4f}")
    b.metric("Ngưỡng cảnh báo", f"{threshold:.4f}")
    c.metric("Nhãn mô hình", int(score >= threshold))
    if issues:
        st.dataframe(pd.DataFrame(issues), hide_index=True, width="stretch")
    if unusual:
        st.caption("Tổ hợp số đo xa các bản ghi cùng loại trong tập train theo mốc khoảng cách tham chiếu.")
    actual = matching_test_label(result["udi"], {name: str(value) for name, value in values.items()}, test)
    if actual is not None:
        if st.button("Hiện nhãn thật", key="reveal"):
            st.session_state["reveal_truth"] = True
        if st.session_state.get("reveal_truth"):
            predicted = int(score >= threshold)
            st.info(f"Nhãn thật: {actual} · " + ("Dự đoán đúng" if actual == predicted else
                                                  "Cảnh báo sai" if predicted else "Bỏ sót mẫu hỏng"))

    actions = st.columns(2)
    if actions[0].button("Lưu làm mốc so sánh", width="stretch"):
        st.session_state["prediction_baseline"] = result
    actions[1].download_button("Tải phiếu kết quả để in", printable_result(values, score, threshold, conclusion),
                               file_name="phieu_du_doan_ai4i.html", mime="text/html", width="stretch")
    baseline = st.session_state.get("prediction_baseline")
    if baseline:
        comparison = [{"Thông số": NAMES[name], "Bản gốc": str(baseline["values"][name]), "Bản đang xem": str(values[name])}
                      for name in FEATURES]
        comparison.append({"Thông số": "Điểm mô hình", "Bản gốc": f'{baseline["score"]:.4f}', "Bản đang xem": f"{score:.4f}"})
        comparison.append({"Thông số": "Nhãn mô hình", "Bản gốc": str(int(baseline["score"] >= threshold)),
                           "Bản đang xem": str(int(score >= threshold))})
        st.markdown("#### So sánh hai kịch bản")
        st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")

    with st.expander("Giải thích và đối chiếu bản ghi tương tự"):
        st.dataframe(explain(frame, model).head(3), hide_index=True, width="stretch")
        st.caption("Dấu dương làm tăng điểm mô hình, dấu âm làm giảm điểm.")
        st.dataframe(matches, hide_index=True, width="stretch")
        st.caption("Ba bản ghi train cùng loại sản phẩm, gần nhất theo khoảng cách chuẩn hóa.")
    with st.expander("So sánh kết quả giữa ba mô hình"):
        st.dataframe(compare_record(frame, score, threshold), hide_index=True, width="stretch",
                     column_config={"Điểm": st.column_config.NumberColumn(format="%.4f"),
                                    "Ngưỡng": st.column_config.NumberColumn(format="%.4f")})
        st.caption("Mỗi mô hình dùng ngưỡng riêng. LightGBM là kết quả chính.")
    with st.expander("Kiểm tra khi số đo dao động"):
        delta = st.slider("Mức dao động giả định (%)", 0.1, 10.0, 2.0, 0.1)
        cases = []
        for name in FEATURES[1:]:
            cases.extend([{**values, name: values[name] * (1 - delta / 100)},
                          {**values, name: values[name] * (1 + delta / 100)}])
        scores = score_batch(pd.DataFrame(cases), model, threshold, metadata["train_ranges"])
        changes = []
        for i, name in enumerate(FEATURES[1:]):
            pair = scores.iloc[i * 2:i * 2 + 2]
            changes.append({"Thông số": NAMES[name], "Khi giảm": pair[SCORE].iloc[0],
                            "Khi tăng": pair[SCORE].iloc[1],
                            "Đổi nhãn": bool(np.any(pair[SCORE].ge(threshold) != (score >= threshold))),
                            "Ngoài khoảng train": bool(pair["Ngoài khoảng train"].any())})
        st.dataframe(pd.DataFrame(changes), hide_index=True, width="stretch")
        st.caption("Mỗi lượt chỉ thay đổi một biến. Mức dao động do người dùng đặt.")
