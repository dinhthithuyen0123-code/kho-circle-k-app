"""
driver_db.py
Database dùng chung (SQLite) giữa app Quản lý kho (app_kho_v4.py) và app Tài xế
(sẽ xây dựng sau) — đây là "cầu nối" để 2 app độc lập nhưng thấy cùng 1 dữ liệu.

2 bảng gốc:
  - leave_requests: yêu cầu Làm việc/Xin nghỉ của tài xế theo từng ngày, kèm trạng thái duyệt
  - delivery_performance: kết quả giao hàng đúng/trễ giờ mỗi ngày, kèm lý do nếu trễ

Các bảng tổng hợp (tháng, tuần) trong giao diện quản lý được TÍNH RA từ 2 bảng gốc
này, không lưu trùng lặp.
"""

import sqlite3
import datetime
import pandas as pd

DB_PATH = "outputs/driver_shared.db"
LATE_REASONS = ["Kẹt xe", "Hư xe", "Thời tiết xấu", "Khác"]


def get_connection():
    import os
    os.makedirs("outputs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS leave_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER, driver_name TEXT, ngay TEXT,
            loai TEXT,              -- 'Làm việc' hoặc 'Xin nghỉ'
            ly_do TEXT, minh_chung TEXT,
            trang_thai TEXT DEFAULT 'Chờ duyệt',   -- Chờ duyệt / Đã duyệt / Từ chối
            nguoi_duyet TEXT, thoi_gian_gui TEXT, thoi_gian_duyet TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS delivery_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id INTEGER, driver_name TEXT, ngay TEXT,
            dung_gio INTEGER,       -- 1 = đúng giờ, 0 = trễ
            ly_do_tre TEXT, ghi_chu TEXT
        )
    """)
    return conn


# ---------------------------------------------------------------------------
# Ghi dữ liệu (app Tài xế sẽ gọi các hàm này sau này)
# ---------------------------------------------------------------------------

def submit_leave_request(driver_id, driver_name, ngay, loai, ly_do="", minh_chung=""):
    """loai='Làm việc' -> tự động 'Đã duyệt' (không cần xét duyệt, chỉ là ghi nhận lịch làm việc thường).
    loai='Xin nghỉ' -> để 'Chờ duyệt', cần quản lý xác nhận."""
    trang_thai_khoi_tao = "Đã duyệt" if loai == "Làm việc" else "Chờ duyệt"
    conn = get_connection()
    conn.execute(
        "INSERT INTO leave_requests (driver_id, driver_name, ngay, loai, ly_do, minh_chung, trang_thai, thoi_gian_gui) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (driver_id, driver_name, ngay, loai, ly_do, minh_chung, trang_thai_khoi_tao, datetime.datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def approve_request(request_id, approved: bool, nguoi_duyet="Quản lý"):
    conn = get_connection()
    trang_thai = "Đã duyệt" if approved else "Từ chối"
    conn.execute(
        "UPDATE leave_requests SET trang_thai=?, nguoi_duyet=?, thoi_gian_duyet=? WHERE id=?",
        (trang_thai, nguoi_duyet, datetime.datetime.now().isoformat(), request_id),
    )
    conn.commit()
    conn.close()


def log_delivery_performance(driver_id, driver_name, ngay, dung_gio: bool, ly_do_tre="", ghi_chu=""):
    conn = get_connection()
    conn.execute(
        "INSERT INTO delivery_performance (driver_id, driver_name, ngay, dung_gio, ly_do_tre, ghi_chu) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (driver_id, driver_name, ngay, int(dung_gio), ly_do_tre, ghi_chu),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Đọc dữ liệu (app Quản lý dùng)
# ---------------------------------------------------------------------------

def get_pending_requests() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM leave_requests WHERE trang_thai='Chờ duyệt' ORDER BY ngay", conn)
    conn.close()
    return df


def get_all_requests(driver_names: list = None) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM leave_requests ORDER BY ngay", conn)
    conn.close()
    if driver_names:
        df = df[df["driver_name"].isin(driver_names)]
    return df


def get_monthly_summary(driver_names: list, month_start: str, month_end: str,
                         hours_per_workday: float = 8.0) -> pd.DataFrame:
    """Tổng hợp theo tháng: giờ làm, giờ nghỉ có phép/không phép, % đúng giờ, lý do trễ phổ biến."""
    conn = get_connection()
    leaves = pd.read_sql("SELECT * FROM leave_requests WHERE ngay BETWEEN ? AND ?",
                          conn, params=(month_start, month_end))
    perf = pd.read_sql("SELECT * FROM delivery_performance WHERE ngay BETWEEN ? AND ?",
                        conn, params=(month_start, month_end))
    conn.close()

    rows = []
    for name in driver_names:
        d_leaves = leaves[leaves["driver_name"] == name]
        d_perf = perf[perf["driver_name"] == name]

        n_work = (d_leaves["loai"] == "Làm việc").sum()
        n_leave_approved = ((d_leaves["loai"] == "Xin nghỉ") & (d_leaves["trang_thai"] == "Đã duyệt")).sum()
        n_leave_unapproved = ((d_leaves["loai"] == "Xin nghỉ") & (d_leaves["trang_thai"] != "Đã duyệt")).sum()

        n_total = len(d_perf)
        n_ontime = d_perf["dung_gio"].sum() if n_total > 0 else 0
        pct_ontime = (n_ontime / n_total * 100) if n_total > 0 else None
        pct_late = (100 - pct_ontime) if pct_ontime is not None else None

        late_reason_top = None
        late_rows = d_perf[d_perf["dung_gio"] == 0]
        if len(late_rows) > 0:
            late_reason_top = late_rows["ly_do_tre"].value_counts().idxmax()

        rows.append({
            "Tên tài xế": name,
            "Số giờ làm việc (tháng)": round(n_work * hours_per_workday, 1),
            "Số giờ nghỉ phép có lý do (tháng)": round(n_leave_approved * hours_per_workday, 1),
            "Số giờ nghỉ không lý do (tháng)": round(n_leave_unapproved * hours_per_workday, 1),
            "% Giao hàng đúng giờ": round(pct_ontime, 1) if pct_ontime is not None else "—",
            "% Giao hàng không đúng giờ": round(pct_late, 1) if pct_late is not None else "—",
            "Lý do giao hàng không đúng giờ phổ biến nhất": late_reason_top or "—",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dữ liệu mẫu — dùng TẠM cho đến khi app Tài xế hoàn thiện và ghi dữ liệu thật
# ---------------------------------------------------------------------------

def seed_sample_data(driver_id_name_map: dict, n_days: int = 30, seed: int = 42):
    """Sinh dữ liệu mẫu (hiệu suất giao hàng + vài yêu cầu nghỉ) để demo giao diện quản lý
    trước khi app Tài xế thật được xây xong. Kiểm tra RIÊNG TỪNG TÊN tài xế — chỉ sinh
    cho những tên chưa từng có dữ liệu (vd khi đổi tên tài xế, tên mới vẫn được sinh mẫu
    dù tên cũ đã có dữ liệu từ trước, tránh trường hợp bảng tổng hợp bị trống)."""
    import numpy as np
    conn = get_connection()
    names_with_data = set(pd.read_sql("SELECT DISTINCT driver_name FROM delivery_performance", conn)["driver_name"])
    conn.close()

    rng = np.random.default_rng(seed)
    today = datetime.date.today()

    for driver_id, name in driver_id_name_map.items():
        if name in names_with_data:
            continue  # tài xế này đã có dữ liệu (thật hoặc mẫu trước đó) — không ghi đè
        for d in range(n_days):
            day = today - datetime.timedelta(days=n_days - d)
            is_late = rng.random() < 0.12
            reason = rng.choice(LATE_REASONS[:-1]) if is_late else ""  # không dùng "Khác" cho dữ liệu mẫu
            log_delivery_performance(driver_id, name, day.isoformat(), dung_gio=not is_late, ly_do_tre=reason)
            submit_leave_request(driver_id, name, day.isoformat(), "Làm việc")

    # Vài yêu cầu xin nghỉ mẫu (để demo nút duyệt) — chỉ thêm cho tài xế MỚI (chưa từng có dữ liệu)
    new_names = [(did, name) for did, name in driver_id_name_map.items() if name not in names_with_data]
    if len(new_names) > 0:
        submit_leave_request(new_names[0][0], new_names[0][1],
                              (today + datetime.timedelta(days=2)).isoformat(), "Xin nghỉ",
                              "Việc gia đình", "don_xin_nghi_mau.jpg")
    if len(new_names) > 1:
        submit_leave_request(new_names[1][0], new_names[1][1],
                              (today + datetime.timedelta(days=3)).isoformat(), "Xin nghỉ",
                              "Ốm", "giay_kham_benh_mau.jpg")
