"""
layer1_ingestion.py
TẦNG 1 — THU THẬP DỮ LIỆU (Data Ingestion)

Trách nhiệm duy nhất của tầng này: nhận dữ liệu bán hàng/tồn kho hàng ngày
từ POS của từng cửa hàng, kiểm tra hợp lệ, và LƯU VÀO "kho dữ liệu thô"
(raw_sales_log.csv) — CHƯA xử lý tổng hợp, chưa dự báo gì cả.

Trong thực tế, hàm receive_daily_pos_data() sẽ được gọi mỗi khi 1 cửa hàng
đồng bộ dữ liệu (qua API, file, hoặc tích hợp trực tiếp phần mềm POS).
Ở đây mô phỏng bằng việc đọc file CSV/Excel người quản lý tải lên mỗi ngày,
dùng lại logic parse đã có ở data_import.py.
"""

import os
import pandas as pd
from data_import import validate_columns, build_fact_store_daily_from_upload

RAW_LOG_PATH = "outputs/raw_sales_log.csv"


def receive_daily_pos_data(raw_df: pd.DataFrame, dim_stores: pd.DataFrame, dim_skus: pd.DataFrame) -> dict:
    """
    Nhận 1 lô dữ liệu (1 ngày hoặc nhiều ngày) từ POS, kiểm tra hợp lệ,
    rồi GHI NỐI TIẾP vào raw_sales_log.csv (không ghi đè — mô phỏng việc
    dữ liệu liên tục đổ về trong suốt 7 ngày).

    Trả về dict: {"success": bool, "n_rows_added": int, "warnings": list, "errors": list}
    """
    errors = validate_columns(raw_df)
    if errors:
        return {"success": False, "n_rows_added": 0, "warnings": [], "errors": errors}

    parsed_df, warnings = build_fact_store_daily_from_upload(raw_df, dim_stores, dim_skus)
    if len(parsed_df) == 0:
        return {"success": False, "n_rows_added": 0, "warnings": warnings, "errors": ["Không có dòng hợp lệ."]}

    os.makedirs(os.path.dirname(RAW_LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(RAW_LOG_PATH)
    parsed_df.to_csv(RAW_LOG_PATH, mode="a", header=write_header, index=False, encoding="utf-8-sig")

    return {"success": True, "n_rows_added": len(parsed_df), "warnings": warnings, "errors": []}


def read_raw_log() -> pd.DataFrame:
    """Đọc toàn bộ dữ liệu thô đã thu thập được (dùng cho Tầng 2)."""
    if not os.path.exists(RAW_LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(RAW_LOG_PATH, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df


def clear_raw_log():
    """Xoá log thô sau khi Tầng 2 đã xử lý xong (bắt đầu chu kỳ 7 ngày mới)."""
    if os.path.exists(RAW_LOG_PATH):
        os.remove(RAW_LOG_PATH)
