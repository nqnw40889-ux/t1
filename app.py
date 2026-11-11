import hashlib
import io
import json
from time import monotonic
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

from openpyxl.utils.exceptions import InvalidFileException

import altair as alt
import pandas as pd
import streamlit as st

from appearance import render_hero
from card_ui import card_prediction_page
from inference import (ALERT, CLEAR, CUTOFF, DATA, ERROR, FEATURES, MAX_ROWS, MODEL_COLUMNS, NAMES,
                       SCORE, TARGET, explain, load_model, prepare, score_batch)
from maintenance import initialize, save_alerts
from operations_ui import maintenance_page
from prediction_tools import excel_reader, read_paste
from prediction_ui import single_prediction_page
from thresholds import confusion_counts, rates

st.set_page_config(page_title="Giám sát hỏng hóc máy · AI4I", page_icon="⚙️", layout="wide")

render_hero()

@st.cache_resource
def assets():
    metadata = json.loads((DATA / "report.json").read_text(encoding="utf-8"))
    records = pd.read_csv(DATA / "test_results.csv", float_precision="round_trip")
    metadata["model_id"] = hashlib.sha256((DATA / "model.txt").read_bytes()).hexdigest()
    initialize()
    return load_model(), metadata, records


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def pct(value):
    return f"{value:.1%}"


def contribution_chart(record, model):
    contributions = explain(record, model)
    st.altair_chart(
        alt.Chart(contributions).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Đóng góp vào điểm thô:Q"),
            y=alt.Y("Thông số:N", sort=contributions["Thông số"].tolist(), title=None),
            color=alt.condition(alt.datum["Đóng góp vào điểm thô"] > 0,
                                alt.value("#CA6841"), alt.value("#158579")),
            tooltip=["Thông số:N", alt.Tooltip("Đóng góp vào điểm thô:Q", format=".4f")],
        ).properties(height=186), width="stretch")
    st.caption("Thanh dương đẩy điểm dự đoán lên, thanh âm kéo điểm xuống. Giá trị thể hiện đóng góp vào đầu ra mô hình.")


def make_alert(metadata, origin, reference, source_id, features, score, outside_train, target=None):
    return {
        "identity": [metadata["model_id"], metadata["threshold"], *source_id],
        "origin": origin, "reference": reference, "features": features,
        "score": float(score), "threshold": metadata["threshold"],
        "target": target, "outside_train": outside_train,
    }


def show_record(record, model, reveal, outside_train=False):
    row = record.iloc[0]
    status = "Có dấu hiệu hỏng theo mô hình" if row["prediction"] else "Dưới ngưỡng cảnh báo"
    if row["prediction"]:
        st.warning(status, icon="⚠️")
    else:
        st.success(status)
    if outside_train:
        st.warning("Thông số ngoài phạm vi train. Kết quả chưa đủ cơ sở để kết luận.")
    a, b, c = st.columns(3)
    a.metric("UDI", str(int(row["UDI"])))
    b.metric("Điểm mô hình", f'{row["score"]:.4f}')
    c.metric("Nhãn kiểm thử" if reveal else "Loại sản phẩm",
             ("Có hỏng" if row[TARGET] else "Không hỏng") if reveal else str(row["Type"]))
    values = pd.DataFrame({
        "Thông số": [NAMES[name] for name in FEATURES],
        "Giá trị": [str(row[name]) for name in FEATURES],
        "Đơn vị": ["", "K", "K", "rpm", "Nm", "phút độ mòn"],
    })
    st.dataframe(values, hide_index=True, width="stretch")
    with st.expander("Các thông số đóng góp vào dự đoán"):
        contribution_chart(record, model)


def replay(model, metadata, test):
    st.subheader("Mô phỏng luồng bản ghi")
    st.caption("Phát lại các mẫu kiểm thử theo UDI. Các mẫu là quan sát tổng hợp của AI4I.")
    if "replay_pos" not in st.session_state:
        st.session_state.replay_pos = metadata["demo_start"]
        st.session_state.replay_active = False
        st.session_state.replay_due = 0.0

    def jump(to_alert=False):
        position = st.session_state.replay_pos
        if to_alert:
            upcoming = test.index[(test.index > position) & test["prediction"].eq(1)]
            if len(upcoming):
                st.session_state.replay_pos = int(upcoming[0])
        else:
            st.session_state.replay_pos = min(position + 1, len(test) - 1)
        st.session_state.replay_active = False

    def resume(interval):
        if st.session_state.replay_pos >= len(test) - 1:
            st.session_state.replay_pos = metadata["demo_start"]
        st.session_state.replay_active = True
        st.session_state.replay_due = monotonic() + interval

    def pause():
        st.session_state.replay_active = False

    pos = st.session_state.replay_pos
    active = st.session_state.replay_active
    buttons = st.columns([1.1, 1, 1.2, 1.7])
    speed, feature, display = st.columns([1, 1.4, 1.1])
    interval = speed.selectbox("Thời gian mỗi mẫu", [1, 2, 4], index=1,
                               format_func=lambda value: f"{value} giây")
    field = feature.selectbox("Thông số trên biểu đồ", FEATURES[1:], format_func=NAMES.get)
    reveal = display.checkbox("Hiện nhãn kiểm thử", value=True)
    following = test.index[(test.index > pos) & test["prediction"].eq(1)]

    buttons[0].button("▶ Bắt đầu", type="primary", disabled=active, width="stretch",
                      on_click=resume, args=(interval,))
    buttons[1].button("Tạm dừng", disabled=not active, width="stretch", on_click=pause)
    buttons[2].button("Mẫu tiếp theo", disabled=pos >= len(test) - 1, width="stretch",
                      on_click=jump)
    buttons[3].button("Đến cảnh báo tiếp theo", disabled=following.empty, width="stretch",
                      on_click=jump, args=(True,))

    @st.fragment(run_every=interval if active else None)
    def live_panel():
        current = st.session_state.replay_pos
        now = monotonic()
        if st.session_state.replay_active and now + 0.05 >= st.session_state.replay_due:
            current = min(current + 1, len(test) - 1)
            st.session_state.replay_pos = current
            st.session_state.replay_due = now + interval
            if current == len(test) - 1:
                pause()
                st.rerun(scope="app")
        st.progress((current + 1) / len(test), text=f"Bản ghi {current + 1:,} / {len(test):,}")
        panel, detail = st.columns([1.15, 1], gap="large")
        with panel:
            st.markdown("#### 20 bản ghi gần nhất")
            timeline = test.iloc[max(0, current - 19):current + 1][["UDI", field, "prediction"]].copy()
            timeline["Bản ghi"] = timeline.index + 1
            chart = alt.Chart(timeline)
            axes = dict(
                x=alt.X("Bản ghi:Q", title="Thứ tự phát lại", scale=alt.Scale(zero=False)),
                y=alt.Y(f"{field}:Q", title=NAMES[field], scale=alt.Scale(zero=False)),
            )
            st.altair_chart(
                (chart.mark_line(color="#177F78").encode(**axes) +
                 chart.mark_point(filled=True, size=72).encode(
                     **axes, color=alt.condition(alt.datum.prediction == 1,
                                                 alt.value("#CA6841"), alt.value("#177F78")),
                     tooltip=["UDI:Q", alt.Tooltip(f"{field}:Q"), "prediction:N"],
                 )).properties(height=280), width="stretch")
            st.caption(f"Chấm cam đánh dấu bản ghi vượt ngưỡng {metadata['threshold']:.4f}.")
        with detail:
            row = test.iloc[current]
            outside = False
            for name, model_name in zip(FEATURES[1:], MODEL_COLUMNS[1:]):
                lower, upper = metadata["train_ranges"][model_name]
                outside |= not lower <= float(row[name]) <= upper
            if row["prediction"]:
                values = {FEATURES[0]: str(row[FEATURES[0]]),
                          **{name: float(row[name]) for name in FEATURES[1:]}}
                save_alerts([make_alert(
                    metadata, "Mô phỏng test", f"UDI {int(row['UDI'])}",
                    ["replay", int(row["UDI"])], values, row["score"], outside, int(row[TARGET]),
                )])
                st.caption("Cảnh báo được lưu vào mục Bảo trì.")
            show_record(test.iloc[[current]], model, reveal, outside_train=outside)

    live_panel()


def read_upload(upload):
    if upload.size > 8 * 1024 * 1024:
        raise ValueError("Tệp CSV tối đa 8 MB.")
    raw = upload.getvalue()
    try:
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", nrows=MAX_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
        raise ValueError("Cần tệp CSV UTF-8, dùng dấu phẩy phân cột và có hàng tiêu đề.") from error
    if not isinstance(frame.index, pd.RangeIndex):
        raise ValueError("Số trường trong một số dòng không khớp với hàng tiêu đề CSV.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Mỗi lượt tối đa {MAX_ROWS:,} dòng.")
    if frame.empty:
        raise ValueError("Tệp CSV chưa có bản ghi.")
    return frame


def batch(model, metadata):
    st.subheader("Dự đoán hàng loạt")
    st.caption("Dự đoán cho từng dòng từ 6 trường AI4I. Các dòng thiếu hoặc sai định dạng được đánh dấu riêng.")
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        source = st.radio("Nguồn dữ liệu", ["Dữ liệu mẫu", "Tải CSV", "Tải Excel", "Dán từ bảng tính"])
        st.download_button("Tải CSV mẫu", (DATA / "mau_du_doan.csv").read_bytes(),
                           file_name="mau_du_doan.csv", mime="text/csv")
    try:
        with right:
            if source == "Tải CSV":
                upload = st.file_uploader("Tệp thông số vận hành", type=["csv"])
                if upload is None:
                    return
                frame = read_upload(upload)
                origin = upload.name
                file_id = hashlib.sha256(upload.getvalue()).hexdigest()
            elif source == "Tải Excel":
                upload = st.file_uploader("Bảng thông số Excel", type=["xlsx"])
                if upload is None:
                    return
                payload = upload.getvalue()
                with excel_reader(payload) as workbook:
                    sheet = st.selectbox("Trang tính", workbook.sheet_names)
                    frame = workbook.parse(sheet, nrows=MAX_ROWS + 1)
                origin = f"{upload.name} · {sheet}"
                file_id = hashlib.sha256(payload + sheet.encode("utf-8")).hexdigest()
            elif source == "Dán từ bảng tính":
                pasted = st.text_area("Dán các dòng từ Excel hoặc CSV",
                                      placeholder="Type   Air temperature [K]   Process temperature [K]   ...")
                if not pasted.strip():
                    return
                frame = read_paste(pasted)
                origin = "Dữ liệu dán"
                file_id = hashlib.sha256(pasted.encode("utf-8")).hexdigest()
            else:
                sample_path = DATA / "mau_du_doan.csv"
                frame = pd.read_csv(sample_path)
                origin = "CSV mẫu"
                file_id = hashlib.sha256(sample_path.read_bytes()).hexdigest()
                st.info(f"Đã nạp {len(frame)} bản ghi mẫu từ tập kiểm thử.")
        if frame.empty or len(frame) > MAX_ROWS:
            raise ValueError(f"Cần từ 1 đến {MAX_ROWS:,} bản ghi.")
        if not isinstance(frame.index, pd.RangeIndex):
            raise ValueError("Một số dòng có số trường không khớp với hàng tiêu đề.")
        prepare(frame)
        with st.expander("Xem trước và sửa dữ liệu đầu vào"):
            preview = frame.head(200).astype("string").fillna("")
            edited = st.data_editor(preview, hide_index=True, width="stretch",
                                    key=f"batch_editor_{file_id}")
            if len(frame) > len(preview):
                st.caption("Có thể sửa 200 dòng đầu trên web. Dự đoán chạy cho toàn bộ tệp.")
            if not edited.equals(preview):
                frame = frame.astype("string")
                frame.loc[edited.index, edited.columns] = edited
        output = score_batch(frame, model, metadata["threshold"], metadata["train_ranges"])
    except (BadZipFile, InvalidFileException, ParseError, KeyError):
        st.error("Không đọc được tệp Excel. Hãy kiểm tra định dạng .xlsx.")
        return
    except ValueError as error:
        st.error(str(error))
        return

    valid = output[ERROR].eq("")
    alerts = output[ALERT].eq(CUTOFF)
    a, b, c = st.columns(3)
    a.metric("Số dòng hợp lệ", int(valid.sum()))
    b.metric("Vượt ngưỡng cảnh báo", int(alerts.sum()))
    c.metric("Số dòng cần sửa", int((~valid).sum()))
    if output["Ngoài khoảng train"].any():
        st.warning(f"{int(output['Ngoài khoảng train'].sum())} dòng có giá trị ngoài khoảng tập train. "
                   "Hãy kiểm tra đơn vị và điều kiện vận hành trước khi dùng kết quả.")
    if st.button("Lưu cảnh báo vào lịch sử", disabled=not alerts.any(), key="save_batch"):
        prepared, _ = prepare(frame.loc[alerts])
        events = []
        for index, row in prepared.iterrows():
            values = {FEATURES[0]: str(row.iloc[0]),
                      **{name: float(value) for name, value in zip(FEATURES[1:], row.iloc[1:])}}
            events.append(make_alert(
                metadata, origin, f"Dòng {index + 1}", ["batch", file_id, int(index), list(values.values())],
                values, output.at[index, SCORE], output.at[index, "Ngoài khoảng train"],
            ))
        added = save_alerts(events)
        st.success(f"Đã lưu {added} cảnh báo mới. {len(events) - added} dòng đã có trong lịch sử.")

    choice = st.selectbox("Lọc kết quả", ["Tất cả", "Vượt ngưỡng", "Dưới ngưỡng", "Cần sửa"])
    filters = {"Tất cả": pd.Series(True, index=output.index), "Vượt ngưỡng": alerts,
               "Dưới ngưỡng": output[ALERT].eq(CLEAR), "Cần sửa": ~valid}
    visible = output.loc[filters[choice]].sort_values(SCORE, ascending=False, na_position="last")
    with st.expander("Lập danh sách kiểm tra theo số lượt"):
        capacity = st.number_input("Số lượt kiểm tra", min_value=1, max_value=1000, value=5)
        queue = output.loc[alerts & ~output["Ngoài khoảng train"]].sort_values(SCORE, ascending=False).head(capacity)
        st.dataframe(queue, width="stretch", column_config={SCORE: st.column_config.NumberColumn(format="%.4f")})
        st.caption("Danh sách xếp theo điểm. Các dòng ngoài khoảng train cần kiểm tra đầu vào trước.")
    result_columns = [ALERT, SCORE, ERROR, "Ngoài khoảng train"]
    input_columns = [column for column in frame.columns if column not in result_columns]
    st.dataframe(visible, hide_index=True, width="stretch",
                 column_order=[*result_columns, *input_columns],
                 column_config={SCORE: st.column_config.NumberColumn(format="%.4f")})
    st.download_button("Xuất toàn bộ kết quả CSV", csv_bytes(output),
                       file_name="du_doan_ai4i.csv", mime="text/csv")
    if valid.any():
        with st.expander("Giải thích một dòng hợp lệ"):
            selected = st.selectbox("Dòng trong dữ liệu nguồn", output.index[valid],
                                    format_func=lambda index: f"Dòng {index + 1}")
            contribution_chart(frame.loc[[selected]], model)


def evaluation(metadata, test, model):
    st.subheader("Kết quả kiểm thử")
    metrics = metadata["metrics"]["Test"]
    cols = st.columns(4)
    for column, name, key in zip(cols, ["Precision", "Recall", "F1", "Average Precision"],
                                 ["precision", "recall", "f1", "ap"]):
        column.metric(name, pct(metrics[key]))
    st.caption(f"Mô hình LightGBM · {metadata['configuration']} · Ngưỡng {metadata['threshold']:.4f} "
               "chọn trên validation theo F1. Mô hình được chọn theo Average Precision trên validation.")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown("#### Ma trận nhầm lẫn")
        matrix = pd.DataFrame(
            [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
            index=pd.Index(["Thật · Không hỏng", "Thật · Có hỏng"], name="Nhãn"),
            columns=["Dự đoán · Không hỏng", "Dự đoán · Có hỏng"],
        )
        st.dataframe(matrix, width="stretch")
        st.caption(f"Phát hiện {metrics['tp']}/{metrics['positive']} mẫu hỏng · "
                   f"Báo nhầm {metrics['fp']} mẫu · Bỏ sót {metrics['fn']} mẫu.")
    with right:
        st.markdown("#### Đường Precision–Recall")
        pr = pd.read_csv(DATA / "pr_curve.csv").reset_index(names="Thứ tự")
        line = alt.Chart(pr).mark_line(color="#177F78", strokeWidth=2.5).encode(
            x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            order="Thứ tự:Q", tooltip=["Recall:Q", "Precision:Q"],
        )
        baseline = alt.Chart(pd.DataFrame({"Precision": [metrics["positive"] / metrics["n"]]})).mark_rule(
            color="#B37250", strokeDash=[4, 4]).encode(y="Precision:Q")
        st.altair_chart((line + baseline).properties(height=205), width="stretch")
        st.caption("Đường đứt biểu thị tỷ lệ mẫu hỏng trong tập test.")

    st.markdown("#### Đối chiếu từng bản ghi")
    case = st.selectbox("Lọc mẫu kiểm thử", ["Tất cả", "Bỏ sót hỏng", "Cảnh báo sai", "Phát hiện đúng"])
    wrong = test[TARGET].ne(test["prediction"])
    filters = {"Tất cả": pd.Series(True, index=test.index),
               "Bỏ sót hỏng": wrong & test[TARGET].eq(1),
               "Cảnh báo sai": wrong & test[TARGET].eq(0),
               "Phát hiện đúng": ~wrong & test[TARGET].eq(1)}
    visible = test.loc[filters[case]].rename(columns={
        TARGET: "Nhãn thật", "prediction": "Dự đoán", "score": SCORE, **NAMES,
    })
    st.dataframe(visible, hide_index=True, width="stretch",
                 column_order=["UDI", "Nhãn thật", "Dự đoán", SCORE, *NAMES.values()],
                 column_config={SCORE: st.column_config.NumberColumn(format="%.4f")})
    st.download_button("Xuất toàn bộ dự đoán test CSV", csv_bytes(test),
                       file_name="ai4i_ket_qua_test.csv", mime="text/csv")

    with st.expander("Phân tích sai số theo loại sản phẩm L, M, H"):
        summary = []
        for product, group in test.groupby("Type", sort=True):
            counts = confusion_counts(group[TARGET], group["score"], metadata["threshold"])
            summary.append({"Loại sản phẩm": product, "Số bản ghi": len(group),
                            "Mẫu hỏng": counts["tp"] + counts["fn"], "Phát hiện đúng": counts["tp"],
                            "Báo nhầm": counts["fp"], "Bỏ sót": counts["fn"],
                            "Recall": pct(rates(counts)["recall"])})
        st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)

    with st.expander("So sánh một ca phát hiện đúng với một ca bỏ sót"):
        detected, missed = st.columns(2, gap="large")
        groups = [test.loc[filters["Phát hiện đúng"]], test.loc[filters["Bỏ sót hỏng"]]]
        if all(len(group) for group in groups):
            for pane, group, label in zip([detected, missed], groups, ["Phát hiện đúng", "Bỏ sót"]):
                with pane:
                    chosen = st.selectbox(label + " · UDI", group["UDI"].tolist())
                    show_record(group.loc[group["UDI"].eq(chosen)], model, reveal=True)
        else:
            st.caption("Tập test cần có cả hai loại kết quả để so sánh.")

    with st.expander("Phân bố dữ liệu và so sánh cấu hình"):
        st.caption(f"AI4I có {metadata['n']:,} bản ghi và {metadata['positive']} mẫu hỏng "
                   f"({metadata['positive'] / metadata['n']:.2%}). Chia phân tầng theo nhãn, seed {metadata['seed']}.")
        splits = pd.DataFrame([
            {"Tập": name, "Số mẫu": row["n"], "Mẫu hỏng": row["positive"],
             "Precision": pct(row["precision"]), "Recall": pct(row["recall"]),
             "F1": pct(row["f1"]), "AP": pct(row["ap"])}
            for name, row in metadata["metrics"].items()
        ])
        st.dataframe(splits, hide_index=True, width="stretch")
        compare = pd.DataFrame([
            {"Cấu hình": row["configuration"], "Ngưỡng": f'{row["threshold"]:.4f}',
             "F1 validation": pct(row["f1"]), "AP validation": pct(row["ap"])}
            for row in metadata["validation_comparison"]
        ])
        st.dataframe(compare, hide_index=True, width="stretch")


@st.cache_data
def validation_assets():
    data = pd.read_csv(DATA / "validation_scores.csv", float_precision="round_trip")
    truth, scores = data[TARGET], data["score"]
    points = pd.DataFrame([
        {"Ngưỡng": i / 100, **rates(confusion_counts(truth, scores, i / 100))}
        for i in range(101)
    ]).rename(columns={"precision": "Precision", "recall": "Recall", "f1": "F1"})
    return data, points


def threshold_page(metadata):
    st.subheader("Thử ngưỡng cảnh báo trên validation")
    validation, points = validation_assets()
    chosen = metadata["threshold"]
    st.caption(f'{len(validation):,} bản ghi validation · {int(validation[TARGET].sum())} mẫu hỏng. '
               "Điều chỉnh dưới đây chỉ áp dụng trong màn hình này.")
    value = st.slider("Ngưỡng thử", min_value=0.0, max_value=1.0, value=chosen, step=0.0001,
                      format="%.4f")
    old = confusion_counts(validation[TARGET], validation["score"], chosen)
    current = confusion_counts(validation[TARGET], validation["score"], value)
    current_rates = rates(current)
    a, b, c, d = st.columns(4)
    for column, name, key in zip((a, b, c), ("Phát hiện đúng", "Cảnh báo sai", "Bỏ sót hỏng"),
                                 ("tp", "fp", "fn")):
        column.metric(name, current[key], delta=int(current[key] - old[key]), delta_color="off")
    d.metric("F1", pct(current_rates["f1"]))
    label = {"tn": "Không hỏng, dự đoán đúng", "fp": "Cảnh báo sai",
             "fn": "Bỏ sót hỏng", "tp": "Phát hiện đúng"}
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.markdown("#### So với ngưỡng triển khai")
        rows = [{"Chỉ số": label[key], f"Đã chọn ({chosen:.4f})": old[key], "Ngưỡng thử": current[key]}
                for key in ("tp", "fp", "fn", "tn")]
        rows.extend({"Chỉ số": name.upper() if name == "f1" else name.capitalize(),
                     f"Đã chọn ({chosen:.4f})": pct(rates(old)[name]), "Ngưỡng thử": pct(current_rates[name])}
                    for name in ("precision", "recall", "f1"))
        st.dataframe(pd.DataFrame(rows).astype(str), hide_index=True, width="stretch")
    with right:
        st.markdown("#### Precision, Recall và F1 theo ngưỡng")
        long = points.melt(id_vars="Ngưỡng", var_name="Chỉ số", value_name="Giá trị")
        lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X("Ngưỡng:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Giá trị:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Chỉ số:N", scale=alt.Scale(domain=["Precision", "Recall", "F1"],
                                                       range=["#177F78", "#CB663D", "#466BC6"])),
            tooltip=[alt.Tooltip("Ngưỡng:Q", format=".2f"), "Chỉ số:N",
                     alt.Tooltip("Giá trị:Q", format=".1%")],
        )
        selected = alt.Chart(pd.DataFrame({"Ngưỡng": [value]})).mark_rule(
            color="#7A8193", strokeDash=[5, 4]).encode(x="Ngưỡng:Q")
        st.altair_chart((lines + selected).properties(height=230), width="stretch")


model, metadata, test = assets()
page = st.radio("Màn hình", ["Dự đoán", "Dự đoán từ ảnh", "Dự đoán hàng loạt", "Giám sát mô phỏng", "Bảo trì",
                             "Thử ngưỡng", "Kết quả kiểm thử"],
                horizontal=True, label_visibility="collapsed", key="page")
if page != "Giám sát mô phỏng":
    st.session_state.replay_active = False
if page == "Dự đoán":
    single_prediction_page(model, metadata, test)
elif page == "Giám sát mô phỏng":
    replay(model, metadata, test)
elif page == "Dự đoán từ ảnh":
    card_prediction_page(model, metadata, test)
elif page == "Dự đoán hàng loạt":
    batch(model, metadata)
elif page == "Bảo trì":
    maintenance_page()
elif page == "Thử ngưỡng":
    threshold_page(metadata)
else:
    evaluation(metadata, test, model)

st.divider()
st.caption("AI4I 2020 là dữ liệu tổng hợp. Nhãn phản ánh tình trạng hỏng của từng bản ghi. "
           "Điểm mô hình được dùng để so với ngưỡng cảnh báo. [Nguồn UCI](https://doi.org/10.24432/C5HS5C).")
