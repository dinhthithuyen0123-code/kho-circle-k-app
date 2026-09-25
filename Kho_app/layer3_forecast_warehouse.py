"""
layer3_forecast_warehouse.py
TẦNG 3 — DỰ BÁO NHU CẦU & TÍNH TỒN KHO TỔNG (Forecasting & Warehouse Planning)

Trách nhiệm duy nhất của tầng này: dùng dữ liệu lịch sử đã tích luỹ qua nhiều
tuần (từ Tầng 2) để huấn luyện mô hình AI, dự báo nhu cầu tuần tới cho từng
cửa hàng × từng mặt hàng, rồi CỘNG GỘP lại thành tổng lượng cần nhập cho
KHO TRUNG TÂM (không phải từng cửa hàng lẻ).
"""

import os
import numpy as np
import pandas as pd
from forecasting_v3 import build_features_v3, train_quantile_models_v3, predict_quantiles_v3

OUT_DIR = "outputs"
HISTORY_PATH = f"{OUT_DIR}/lich_su_ban_hang_full.csv"


def compute_eoq(demand_per_period: float, fixed_order_cost: float, holding_cost_per_unit_per_period: float) -> float:
    """
    Công thức EOQ (Economic Order Quantity) — lượng đặt hàng tối ưu, cân bằng giữa
    chi phí cố định mỗi lần đặt hàng (K) và chi phí giữ hàng (h):

        Q* = sqrt(2 * D * K / h)

    Nếu K tương đối lớn so với h, Q* sẽ LỚN HƠN nhu cầu thô của kỳ hiện tại —
    tức "nhập nhiều hơn dự báo trước mắt" vẫn có thể là quyết định tối ưu chi phí,
    vì tránh phải đặt hàng thường xuyên (mỗi lần đặt đều tốn phí cố định).
    """
    if demand_per_period <= 0 or holding_cost_per_unit_per_period <= 0:
        return 0.0
    return float(np.sqrt(2 * demand_per_period * fixed_order_cost / holding_cost_per_unit_per_period))


def load_full_history() -> pd.DataFrame:
    """Đọc toàn bộ lịch sử đã tích luỹ, tính lại day_idx TOÀN CỤC theo ngày thực tế
    (không dùng day_idx đã lưu sẵn — vì mỗi lô tuần từ Tầng 1/2 tính day_idx riêng
    bắt đầu từ 0, sẽ bị trùng/chồng lấn giữa các tuần nếu ghép trực tiếp)."""
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    min_date = df["date"].min()
    df["day_idx"] = (df["date"] - min_date).dt.days
    return df


def run_forecast_and_plan_warehouse(dim_stores: pd.DataFrame, dim_skus: pd.DataFrame,
                                     test_days: int = 7, fixed_dispatch_cost: float = 25.0) -> dict:
    """
    Huấn luyện mô hình dự báo trên TOÀN BỘ lịch sử đã có (từ Tầng 2 tích luỹ qua các tuần),
    dự báo nhu cầu 7 ngày tới cho từng cửa hàng × SKU, rồi cộng gộp theo SKU để
    ra TỔNG LƯỢNG CẦN NHẬP CHO KHO TRUNG TÂM.

    Trả về: {"success": bool, "forecast_detail": DataFrame, "warehouse_plan": DataFrame, "message": str}
    """
    history_df = load_full_history()
    if len(history_df) == 0:
        return {"success": False, "forecast_detail": None, "warehouse_plan": None,
                "message": "Chưa có đủ lịch sử (cần Tầng 2 chạy ít nhất vài tuần trước khi dự báo được)."}

    feat_df = build_features_v3(history_df)
    if len(feat_df) < 30:
        return {"success": False, "forecast_detail": None, "warehouse_plan": None,
                "message": f"Lịch sử còn quá ít ({len(feat_df)} dòng sau khi tạo đặc trưng) — "
                           f"cần chạy Tầng 2 thêm vài tuần nữa để có đủ dữ liệu huấn luyện."}

    split_day = feat_df["day_idx"].max() - min(test_days, feat_df["day_idx"].max() // 3)
    train_df = feat_df[feat_df["day_idx"] < split_day]
    test_df = feat_df[feat_df["day_idx"] >= split_day]

    models = train_quantile_models_v3(train_df)
    forecast_detail = predict_quantiles_v3(models, test_df)
    forecast_detail = forecast_detail.merge(dim_stores[["store_id", "store_name"]], on="store_id") \
        .merge(dim_skus[["sku_id", "sku_name"]], on="sku_id")

    # Tồn kho hiện tại của cửa hàng (dòng cuối cùng trong lịch sử) để tính lượng CẦN NHẬP
    latest_stock = history_df.sort_values("day_idx").groupby(["store_id", "sku_id"])["closing_stock"].last()

    warehouse_plan = forecast_detail.groupby(["sku_id", "sku_name"]).agg(
        tong_du_bao_tuan_toi=("q50", "sum"),
        muc_an_toan_toi_da=("q90", "sum"),
    ).reset_index()

    tong_ton_hien_tai = latest_stock.reset_index().groupby("sku_id")["closing_stock"].sum()
    warehouse_plan = warehouse_plan.merge(
        tong_ton_hien_tai.rename("tong_ton_hien_tai"), on="sku_id", how="left"
    ).fillna({"tong_ton_hien_tai": 0})

    warehouse_plan["nhu_cau_tho"] = (
        warehouse_plan["tong_du_bao_tuan_toi"] - warehouse_plan["tong_ton_hien_tai"]
    ).clip(lower=0)

    # ---- Áp dụng EOQ: cân bằng phí cố định mỗi lần đặt hàng (K) và chi phí giữ hàng (h) ----
    # holding cost hiện có là theo NGÀY (unit_holding_cost trong dim_skus) -> quy đổi sang theo TUẦN
    holding_cost_map = dim_skus.set_index("sku_id")["unit_holding_cost"] * 7
    warehouse_plan["eoq"] = warehouse_plan.apply(
        lambda r: compute_eoq(r["tong_du_bao_tuan_toi"], fixed_dispatch_cost,
                               holding_cost_map.get(r["sku_id"], 0.5 * 7)),
        axis=1,
    )
    # Chỉ đề xuất nhập khi thực sự cần (nhu cầu thô > 0); khi đó lấy MAX(nhu cầu thô, EOQ)
    # -> có thể đề xuất nhập NHIỀU HƠN nhu cầu trước mắt nếu điều đó tối ưu chi phí tổng thể hơn
    warehouse_plan["can_nhap_kho_trung_tam"] = np.where(
        warehouse_plan["nhu_cau_tho"] > 0,
        np.maximum(warehouse_plan["nhu_cau_tho"], warehouse_plan["eoq"]),
        0.0,
    )

    warehouse_plan = warehouse_plan[[
        "sku_name", "tong_du_bao_tuan_toi", "tong_ton_hien_tai", "nhu_cau_tho", "eoq",
        "can_nhap_kho_trung_tam", "muc_an_toan_toi_da",
    ]].round(0)
    warehouse_plan.columns = ["Mặt hàng", "Dự báo tuần tới (toàn hệ thống)", "Tồn hiện tại (toàn hệ thống)",
                               "Nhu cầu thô (dự báo - tồn)", "Lượng đặt hàng tối ưu (EOQ)",
                               "➜ Đề xuất nhập kho trung tâm", "Mức an toàn tối đa"]

    plan_path = f"{OUT_DIR}/ke_hoach_nhap_kho_trung_tam.csv"
    os.makedirs(OUT_DIR, exist_ok=True)
    warehouse_plan.to_csv(plan_path, index=False, encoding="utf-8-sig")

    # ---- Kế hoạch EOQ ở MỨC TỪNG CỬA HÀNG (quy mô nhỏ hơn -> EOQ dễ "thắng" nhu cầu thô hơn) ----
    store_plan = forecast_detail.groupby(["store_id", "store_name", "sku_id", "sku_name"]).agg(
        du_bao_tuan_toi=("q50", "sum"),
    ).reset_index()
    store_plan["ton_hien_tai"] = store_plan.apply(
        lambda r: latest_stock.get((r["store_id"], r["sku_id"]), 0.0), axis=1
    )
    store_plan["nhu_cau_tho"] = (store_plan["du_bao_tuan_toi"] - store_plan["ton_hien_tai"]).clip(lower=0)
    store_plan["eoq"] = store_plan.apply(
        lambda r: compute_eoq(r["du_bao_tuan_toi"], fixed_dispatch_cost,
                               holding_cost_map.get(r["sku_id"], 0.5 * 7)),
        axis=1,
    )
    store_plan["de_xuat_nhap"] = np.where(
        store_plan["nhu_cau_tho"] > 0, np.maximum(store_plan["nhu_cau_tho"], store_plan["eoq"]), 0.0
    )
    store_plan["eoq_thang"] = store_plan["eoq"] > store_plan["nhu_cau_tho"]

    store_plan_display = store_plan[[
        "store_name", "sku_name", "du_bao_tuan_toi", "ton_hien_tai", "nhu_cau_tho", "eoq", "de_xuat_nhap", "eoq_thang",
    ]].round(0)
    store_plan_display.columns = ["Cửa hàng", "Mặt hàng", "Dự báo tuần tới", "Tồn hiện tại",
                                   "Nhu cầu thô", "EOQ", "Đề xuất nhập", "EOQ thắng?"]
    store_plan_path = f"{OUT_DIR}/ke_hoach_theo_tung_cua_hang.csv"
    store_plan_display.to_csv(store_plan_path, index=False, encoding="utf-8-sig")

    n_eoq_wins = int(store_plan["eoq_thang"].sum())

    return {"success": True, "forecast_detail": forecast_detail, "warehouse_plan": warehouse_plan,
            "store_plan": store_plan_display, "n_eoq_wins": n_eoq_wins,
            "message": f"Đã dự báo cho {forecast_detail['sku_id'].nunique()} mặt hàng, "
                       f"lưu kế hoạch nhập kho tại {plan_path}. "
                       f"Ở mức từng cửa hàng: {n_eoq_wins}/{len(store_plan)} trường hợp EOQ đề xuất "
                       f"nhập NHIỀU HƠN nhu cầu thô."}
