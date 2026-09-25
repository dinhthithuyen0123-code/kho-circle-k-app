"""
forecasting_v3.py
Dự báo nhu cầu xác suất ở cấp độ SKU x cửa hàng, có dùng thời tiết, lễ Tết, khuyến mãi.
Dùng chung kiến trúc Gradient Boosting Quantile Regression như forecasting.py gốc,
nhưng mở rộng thêm chiều SKU và các yếu tố ngoại sinh (thời tiết).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

QUANTILES = [0.1, 0.5, 0.9]

FEATURE_COLS = [
    "store_id", "sku_id", "lag_1", "lag_7",
    "roll_mean_7", "roll_mean_14",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "temperature_c", "is_weekend", "is_tet_period", "is_back_to_school",
]

# Nhóm hàng "ăn nhanh / tiện lợi" — nhóm hưởng lợi từ mùa tựu trường tháng 9
FAST_FOOD_CATEGORIES = ("Mì ly/ăn liền", "Thực phẩm chế biến")


def build_features_v3(fact_store_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo đặc trưng cho từng cặp (store_id, sku_id). Dùng 'true_demand' = units_sold + lost_sales
    làm biến mục tiêu (thay vì chỉ units_sold), vì units_sold bị "che khuất" khi hết hàng
    (censored demand) — dùng true_demand giúp mô hình không đánh giá thấp nhu cầu thực.
    """
    df = fact_store_daily.copy()
    df["true_demand"] = df["units_sold"] + df["lost_sales"]
    df = df.sort_values(["store_id", "sku_id", "day_idx"])

    grp = df.groupby(["store_id", "sku_id"])["true_demand"]
    df["lag_1"] = grp.shift(1)
    df["lag_7"] = grp.shift(7)
    df["roll_mean_7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll_mean_14"] = grp.transform(lambda s: s.shift(1).rolling(14).mean())

    doy = pd.to_datetime(df["date"]).dt.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # Mùa tựu trường: tháng 9 sinh viên nhập học -> nhu cầu mặt hàng ăn nhanh/tiện lợi tăng
    df["is_back_to_school"] = (pd.to_datetime(df["date"]).dt.month == 9).astype(int)

    df = df.dropna(subset=["lag_1", "lag_7", "roll_mean_7", "roll_mean_14"]).reset_index(drop=True)
    return df


def train_quantile_models_v3(train_df: pd.DataFrame) -> dict:
    X = train_df[FEATURE_COLS]
    y = train_df["true_demand"]
    models = {}
    for q in QUANTILES:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=150, max_depth=4,
            learning_rate=0.06, subsample=0.8, random_state=42,
        )
        m.fit(X, y)
        models[q] = m
    return models


def predict_quantiles_v3(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS]
    keep_cols = list(dict.fromkeys(["store_id", "sku_id", "date", "day_idx", "true_demand"] + FEATURE_COLS))
    out = df[keep_cols].copy()
    for q in QUANTILES:
        out[f"q{int(q*100)}"] = np.maximum(0, models[q].predict(X))
    out[["q10", "q50", "q90"]] = np.sort(out[["q10", "q50", "q90"]].values, axis=1)
    return out


def get_feature_importance(models: dict, top_n: int = 6) -> pd.DataFrame:
    """Mức độ quan trọng của từng yếu tố đối với mô hình dự báo trung vị (q50)."""
    importances = models[0.5].feature_importances_
    df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


FEATURE_LABELS_VI = {
    "store_id": "Cửa hàng cụ thể", "sku_id": "Mặt hàng cụ thể",
    "lag_1": "Nhu cầu ngày hôm trước", "lag_7": "Nhu cầu cùng thứ tuần trước",
    "roll_mean_7": "Trung bình 7 ngày gần đây", "roll_mean_14": "Trung bình 14 ngày gần đây",
    "dow_sin": "Chu kỳ ngày trong tuần", "dow_cos": "Chu kỳ ngày trong tuần",
    "doy_sin": "Chu kỳ mùa vụ trong năm", "doy_cos": "Chu kỳ mùa vụ trong năm",
    "temperature_c": "Nhiệt độ", "is_weekend": "Cuối tuần", "is_tet_period": "Giai đoạn cao điểm Tết",
    "is_back_to_school": "Mùa tựu trường (tháng 9)",
}


def explain_row(row: pd.Series, category: str = None) -> list:
    """
    Sinh giải thích ngôn ngữ tự nhiên cho MỘT dự báo cụ thể.
    `category`: nhóm hàng của SKU (tuỳ chọn) — dùng để giải thích chính xác hơn
    cho các yếu tố chỉ áp dụng với một số nhóm hàng nhất định (vd: mùa tựu trường
    chỉ thực sự ảnh hưởng tới nhóm hàng ăn nhanh/tiện lợi, không phải mọi mặt hàng).
    """
    reasons = []

    if row.get("is_tet_period", 0) == 1:
        reasons.append("🧧 Ngày này rơi vào **giai đoạn cao điểm Tết** — nhu cầu thường tăng mạnh so với ngày thường.")

    if row.get("is_weekend", 0) == 1:
        reasons.append("📅 Đây là **cuối tuần** — nhu cầu nhóm hàng tiêu dùng nhanh thường cao hơn ngày thường.")

    if row.get("is_back_to_school", 0) == 1:
        if category is None or category in FAST_FOOD_CATEGORIES:
            reasons.append("🎒 Đây là **tháng 9 — mùa sinh viên nhập học** — nhu cầu nhóm hàng ăn nhanh/tiện lợi "
                            "(mì ly, thực phẩm chế biến sẵn...) thường tăng do sinh viên mới chuyển đến khu vực.")
        else:
            reasons.append("🎒 Đây là tháng 9 (mùa tựu trường) — yếu tố này chủ yếu ảnh hưởng nhóm hàng ăn nhanh, "
                            "ít tác động trực tiếp đến mặt hàng này.")

    temp = row.get("temperature_c", None)
    if temp is not None and not pd.isna(temp):
        if temp >= 30:
            reasons.append(f"🌡️ Nhiệt độ dự kiến khá cao (**{temp:.1f}°C**) — có thể làm tăng nhu cầu với mặt hàng nhạy cảm thời tiết (đồ uống lạnh...).")
        elif temp <= 24:
            reasons.append(f"🌡️ Nhiệt độ dự kiến khá thấp (**{temp:.1f}°C**) — có thể làm giảm nhu cầu với mặt hàng nhạy cảm thời tiết.")

    roll7, roll14 = row.get("roll_mean_7"), row.get("roll_mean_14")
    if roll7 is not None and roll14 is not None and not pd.isna(roll7) and not pd.isna(roll14) and roll14 > 0:
        change = (roll7 - roll14) / roll14 * 100
        if change > 15:
            reasons.append(f"📈 Xu hướng gần đây đang **tăng** — trung bình 7 ngày cao hơn trung bình 14 ngày khoảng {change:.0f}%.")
        elif change < -15:
            reasons.append(f"📉 Xu hướng gần đây đang **giảm** — trung bình 7 ngày thấp hơn trung bình 14 ngày khoảng {abs(change):.0f}%.")

    lag1, lag7 = row.get("lag_1"), row.get("lag_7")
    if lag1 is not None and lag7 is not None and not pd.isna(lag1) and not pd.isna(lag7) and lag7 > 0:
        diff = (lag1 - lag7) / lag7 * 100
        if abs(diff) > 30:
            direction = "cao hơn" if diff > 0 else "thấp hơn"
            reasons.append(f"🔁 Nhu cầu ngày hôm trước {direction} đáng kể ({abs(diff):.0f}%) so với cùng thứ tuần trước.")

    if not reasons:
        reasons.append("➡️ Không có yếu tố bất thường — dự báo chủ yếu dựa trên mức nhu cầu nền ổn định gần đây.")

    return reasons
