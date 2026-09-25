"""
layer2_weekly_batch.py
TẦNG 2 — XỬ LÝ CUỐI TUẦN (Weekly Batch Processing)

Trách nhiệm duy nhất của tầng này: lấy dữ liệu thô đã tích luỹ suốt 7 ngày
từ Tầng 1, TỔNG HỢP lại theo từng cửa hàng × từng mặt hàng, rồi XUẤT RA FILE
CSV báo cáo — không chạy AI, không dự báo gì ở tầng này.

Chạy hàm run_weekly_batch() vào cuối mỗi tuần (thứ Bảy/Chủ Nhật).
"""

import os
import pandas as pd
from layer1_ingestion import read_raw_log, clear_raw_log

OUT_DIR = "outputs"


def run_weekly_batch(dim_stores: pd.DataFrame, dim_skus: pd.DataFrame, reset_log: bool = True) -> dict:
    """
    Tổng hợp toàn bộ dữ liệu thô đã thu thập trong tuần (từ Tầng 1) thành
    báo cáo: tổng bán ra + tồn còn lại, theo từng cửa hàng × từng mặt hàng.

    Trả về: {"success": bool, "report_path": str, "report_df": DataFrame, "message": str}
    """
    raw_df = read_raw_log()
    if len(raw_df) == 0:
        return {"success": False, "report_path": None, "report_df": None,
                "message": "Chưa có dữ liệu nào được thu thập trong tuần (Tầng 1 chưa nhận dữ liệu)."}

    report = (
        raw_df.groupby(["store_id", "sku_id"])
        .agg(tong_ban_ra=("units_sold", "sum"),
             tong_nhap=("units_received", "sum"),
             ton_cuoi_ky=("closing_stock", "last"))
        .reset_index()
        .merge(dim_stores[["store_id", "store_name"]], on="store_id")
        .merge(dim_skus[["sku_id", "sku_name", "category"]], on="sku_id")
    )
    report = report[["store_name", "sku_name", "category", "tong_ban_ra", "tong_nhap", "ton_cuoi_ky"]]
    report.columns = ["Cửa hàng", "Mặt hàng", "Nhóm hàng", "Tổng bán ra (7 ngày)", "Tổng nhập (7 ngày)", "Tồn cuối kỳ"]

    os.makedirs(OUT_DIR, exist_ok=True)
    week_label = pd.Timestamp.now().strftime("%Y-%m-%d")
    report_path = f"{OUT_DIR}/bao_cao_tuan_{week_label}.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    # Lưu lại dữ liệu thô tuần này vào kho lịch sử dài hạn (để Tầng 3 dùng huấn luyện dự báo)
    history_path = f"{OUT_DIR}/lich_su_ban_hang_full.csv"
    write_header = not os.path.exists(history_path)
    raw_df.to_csv(history_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")

    if reset_log:
        clear_raw_log()  # bắt đầu chu kỳ thu thập 7 ngày mới (Tầng 1)

    return {"success": True, "report_path": report_path, "report_df": report,
            "message": f"Đã xử lý {len(raw_df)} dòng dữ liệu thô, xuất báo cáo {len(report)} dòng."}
