"""Streamlit page for reviewing saved alerts and updating maintenance tickets."""
import json

import pandas as pd
import streamlit as st

from inference import NAMES
from maintenance import STATUSES, create_ticket, list_alerts, update_ticket


def history_table(alerts):
    return pd.DataFrame([
        {
            "Mã bản ghi": alert["reference"], "Nguồn": alert["origin"],
            "Giờ ghi nhận": local_time(alert["created_at"]),
            "Điểm": alert["score"], "Ngưỡng": alert["threshold"],
            "Nhãn thật": label(alert["target"]),
            "Ngoài phạm vi train": bool(alert["outside_train"]),
            "Phiếu": ticket_number(alert["ticket_id"]) if alert["ticket_id"] is not None else "",
            "Trạng thái": alert["status"] or "Chưa lập phiếu",
        }
        for alert in alerts
    ])


def local_time(value):
    return pd.Timestamp(value).tz_convert("Asia/Ho_Chi_Minh").strftime("%d/%m/%Y %H:%M:%S")


def label(target):
    if target is None:
        return "Chưa có nhãn"
    return "Có hỏng" if target else "Không hỏng"


def ticket_number(identifier):
    return f"BT-{identifier:04d}"


def spreadsheet_text(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def maintenance_page():
    st.subheader("Cảnh báo và phiếu bảo trì")
    if "maintenance_notice" in st.session_state:
        st.success(st.session_state.pop("maintenance_notice"))
    alerts = list_alerts()
    if not alerts:
        st.info("Hãy mở Giám sát mô phỏng và bấm Đến cảnh báo tiếp theo, hoặc lưu cảnh báo từ Dự đoán hàng loạt.")
        return

    a, b, c = st.columns(3)
    a.metric("Cảnh báo đã lưu", len(alerts))
    b.metric("Phiếu chưa hoàn thành", sum(row["status"] in STATUSES[:2] for row in alerts))
    c.metric("Phiếu đã xử lý", sum(row["status"] == STATUSES[2] for row in alerts))
    if st.session_state.pop("maintenance_filter_reset", False):
        st.session_state.maintenance_filter = "Tất cả"
    selection = st.selectbox("Lọc theo trạng thái phiếu", ["Tất cả", "Chưa lập phiếu", *STATUSES],
                             key="maintenance_filter")
    filtered = [row for row in alerts if selection in ("Tất cả", row["status"] or "Chưa lập phiếu")]
    export = history_table(alerts)
    export["Ghi chú phiếu"] = [row["notes"] or "" for row in alerts]
    st.download_button(
        "Tải lịch sử và phiếu CSV", export.map(spreadsheet_text).to_csv(index=False).encode("utf-8-sig"),
        file_name="lich_su_bao_tri_ai4i.csv", mime="text/csv",
    )
    if not filtered:
        st.info("Không có cảnh báo ở trạng thái đã chọn.")
        return
    st.dataframe(history_table(filtered), hide_index=True, width="stretch",
                 column_config={"Điểm": st.column_config.NumberColumn(format="%.4f"),
                                "Ngưỡng": st.column_config.NumberColumn(format="%.4f")})

    by_id = {row["id"]: row for row in filtered}
    selected_id = st.selectbox(
        "Chọn cảnh báo để xem chi tiết", list(by_id),
        format_func=lambda key: f'{by_id[key]["reference"]} · {by_id[key]["origin"]} · {by_id[key]["score"]:.4f}',
    )
    alert = by_id[selected_id]
    if alert["outside_train"]:
        st.warning("Thông số ngoài phạm vi train. Kết quả chưa đủ cơ sở để kết luận.")
    details, ticket = st.columns([1, 1], gap="large")
    with details:
        st.markdown("#### Thông số đã lưu")
        values = json.loads(alert["features"])
        st.dataframe(pd.DataFrame({"Thông số": [NAMES[name] for name in values],
                                   "Giá trị": [str(value) for value in values.values()]}),
                     hide_index=True, width="stretch")
        st.caption(f'Điểm {alert["score"]:.4f} · Ngưỡng {alert["threshold"]:.4f} · {label(alert["target"])}')
    with ticket:
        st.markdown("#### Phiếu kiểm tra")
        if alert["ticket_id"] is None:
            if st.button("Tạo phiếu kiểm tra", type="primary"):
                create_ticket(selected_id)
                st.session_state.maintenance_notice = "Đã tạo phiếu kiểm tra."
                st.session_state.maintenance_filter_reset = True
                st.rerun()
        else:
            number = ticket_number(alert["ticket_id"])
            st.caption(f'{number} · Cập nhật {local_time(alert["updated_at"])}')
            with st.form(f'ticket_{alert["ticket_id"]}_{alert["revision"]}'):
                status = st.selectbox("Trạng thái phiếu", STATUSES, index=STATUSES.index(alert["status"]))
                notes = st.text_area("Ghi chú kiểm tra", value=alert["notes"], max_chars=2000)
                if st.form_submit_button("Lưu thay đổi", type="primary"):
                    if update_ticket(alert["ticket_id"], alert["revision"], status, notes):
                        st.session_state.maintenance_notice = f"Đã lưu phiếu {number}."
                        st.session_state.maintenance_filter_reset = True
                        st.rerun()
                    st.error("Phiếu vừa được cập nhật ở phiên khác. Hãy tải lại trang trước khi lưu.")
    st.caption("Lịch sử và phiếu được lưu trên máy chạy Streamlit. Bản ghi AI4I phục vụ mô phỏng quy trình kiểm tra.")
