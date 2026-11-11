"""Streamlit prediction from a fixed-layout AI4I PNG."""
import hashlib

import pandas as pd
import streamlit as st

from card_ocr import read_card, read_photo
from inference import DATA, ERROR, FEATURES, NAMES, SCORE, TARGET, score_batch

UNITS = ("", "K", "K", "rpm", "N·m", "phút")
OUTCOMES = {
    (0, 0): "Dự đoán đúng mẫu không hỏng",
    (1, 1): "Phát hiện đúng mẫu hỏng",
    (0, 1): "Cảnh báo sai",
    (1, 0): "Bỏ sót mẫu hỏng",
}


def matching_test_label(udi, values, test):
    found = test.loc[test["UDI"].eq(udi)]
    if len(found) != 1:
        return None
    row = found.iloc[0]
    if str(row[FEATURES[0]]) != values[FEATURES[0]]:
        return None
    if any(float(row[feature]) != float(values[feature]) for feature in FEATURES[1:]):
        return None
    return int(row[TARGET])


def card_prediction_page(model, metadata, test):
    st.subheader("Dự đoán hỏng hóc từ ảnh phiếu đo")
    st.caption("Tải PNG mẫu để dự đoán ngay. Với ảnh chụp phiếu, kiểm tra sáu giá trị OCR trước khi dự đoán.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        upload = st.file_uploader("Tải phiếu PNG hoặc JPG", type=["png", "jpg", "jpeg"], key="card_png")
        with st.expander("Tải hai ảnh mẫu để thử"):
            for path in sorted((DATA / "phieu_mau").glob("*.png")):
                st.download_button(f"Tải {path.name}", data=path.read_bytes(), file_name=path.name,
                                   mime="image/png", key=f"get_{path.stem}")
        if upload is None:
            return
        payload = upload.getvalue()
        exact = True
        try:
            udi, values = read_card(payload)
        except ValueError:
            exact = False
            try:
                udi, values = read_photo(payload)
            except ValueError as error:
                st.error(str(error))
                return
        caption = f"Phiếu đọc được · UDI {udi:05d}" if udi is not None else "Phiếu đọc được"
        st.image(payload, caption=caption, width="stretch")
    with right:
        st.markdown("#### Sáu thông số đọc từ ảnh")
        table = pd.DataFrame({
            "Thông số": [NAMES[name] for name in FEATURES],
            "Giá trị": [values[name] for name in FEATURES],
            "Đơn vị": UNITS,
        })
        st.dataframe(table, hide_index=True, width="stretch")
        with st.expander("Sửa thông số đọc từ ảnh", expanded=not exact):
            key = hashlib.sha256(payload).hexdigest()
            with st.form(f"card_edit_{key}"):
                corrected = {name: st.text_input(NAMES[name], value=values[name], key=f"ocr_{key}_{i}")
                             for i, name in enumerate(FEATURES)}
                if st.form_submit_button("Dự đoán lại" if exact else "Xác nhận và dự đoán"):
                    st.session_state["ocr_correction"] = (key, corrected)
    correction = st.session_state.get("ocr_correction")
    if correction and correction[0] == key:
        values = {name: value.strip().upper() if name == "Type" else value.strip().replace(",", ".")
                  for name, value in correction[1].items()}
        st.caption("Kết quả sử dụng thông số đã xác nhận.")
    elif not exact:
        st.info("Kiểm tra hoặc điền các ô còn trống, rồi bấm Xác nhận và dự đoán.")
        return

    threshold = metadata["threshold"]
    result = score_batch(pd.DataFrame([values]), model, threshold, metadata["train_ranges"]).iloc[0]
    if result[ERROR]:
        st.error(result[ERROR])
        return
    score = float(result[SCORE])
    predicted = int(score >= threshold)
    actual = matching_test_label(udi, values, test)
    if result["Ngoài khoảng train"]:
        st.warning("Thông số ngoài phạm vi train. Điểm dưới đây cần được xem xét cùng điều kiện vận hành.")
    st.markdown("#### Kết quả dự đoán")
    a, b, c = st.columns(3)
    a.metric("P(hỏng) ước lượng", f"{score:.2%}")
    b.metric("Nhãn dự đoán", f"{predicted} · " + ("Cảnh báo" if predicted else "Dưới ngưỡng"))
    c.metric("Nhãn thực tế", "Chưa có đối chiếu" if actual is None else f"{actual} · " + ("Có hỏng" if actual else "Không hỏng"))
    if actual is not None:
        message = OUTCOMES[(actual, predicted)]
        if actual == predicted:
            st.success(message)
        else:
            st.warning(message)
    else:
        st.caption("Chỉ đối chiếu nhãn khi UDI và toàn bộ thông số đọc được khớp với một bản ghi test.")
    st.caption(f"Ngưỡng phân loại {threshold:.2%}. P(hỏng) là đầu ra LightGBM, chưa hiệu chuẩn thành xác suất hỏng thực tế.")
