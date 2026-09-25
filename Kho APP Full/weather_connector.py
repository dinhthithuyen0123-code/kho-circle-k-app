"""
weather_connector.py
Lấy dữ liệu THỜI TIẾT THẬT cho từng cửa hàng (theo tọa độ) từ Open-Meteo API
(https://open-meteo.com) — miễn phí, KHÔNG CẦN đăng ký / API key, phù hợp cho
mục đích phi thương mại (đồ án, nghiên cứu).

Thay thế cho phần nhiệt độ mô phỏng (generate_weather() trong data_generator_v3.py).
"""

import requests
import pandas as pd

BASE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_current_and_forecast_temperature(lat: float, lon: float, forecast_days: int = 7) -> pd.DataFrame:
    """
    Lấy nhiệt độ trung bình mỗi ngày: hôm nay + `forecast_days` ngày tới, cho 1 vị trí (lat, lon).
    Trả về DataFrame: date, temperature_c (trung bình ngày = (max+min)/2).
    """
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "forecast_days": forecast_days,
        "timezone": "Asia/Ho_Chi_Minh",
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["daily"]

    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temp_max": data["temperature_2m_max"],
        "temp_min": data["temperature_2m_min"],
    })
    df["temperature_c"] = (df["temp_max"] + df["temp_min"]) / 2
    return df[["date", "temperature_c"]]


def get_historical_temperature(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Lấy nhiệt độ LỊCH SỬ thật (để dùng huấn luyện AI, thay cho phần mô phỏng)
    cho 1 vị trí, từ start_date đến end_date (định dạng 'YYYY-MM-DD').
    Open-Meteo Archive có dữ liệu lịch sử miễn phí, thường trễ khoảng 5 ngày so với hiện tại.
    """
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Ho_Chi_Minh",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()["daily"]

    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temp_max": data["temperature_2m_max"],
        "temp_min": data["temperature_2m_min"],
    })
    df["temperature_c"] = (df["temp_max"] + df["temp_min"]) / 2
    return df[["date", "temperature_c"]]


def get_weather_for_all_stores(dim_stores: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Lấy nhiệt độ lịch sử THẬT cho TỪNG cửa hàng theo đúng tọa độ (lat, lon) của nó
    (dim_stores cần có cột 'lat', 'lon', 'store_id' — đúng định dạng từ real_store_data.py).
    Trả về: store_id, date, temperature_c — dùng thay thế cột temperature_c
    trong fact_store_daily hiện đang mô phỏng.
    """
    all_rows = []
    for _, store in dim_stores.iterrows():
        df = get_historical_temperature(store["lat"], store["lon"], start_date, end_date)
        df["store_id"] = store["store_id"]
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)[["store_id", "date", "temperature_c"]]


def merge_weather_into_fact_store_daily(fact_store_daily: pd.DataFrame, dim_stores: pd.DataFrame) -> pd.DataFrame:
    """
    Thay cột temperature_c (đang mô phỏng) trong fact_store_daily bằng nhiệt độ THẬT,
    lấy đúng theo khoảng ngày và tọa độ từng cửa hàng đã có trong dữ liệu.
    """
    start_date = pd.to_datetime(fact_store_daily["date"]).min().strftime("%Y-%m-%d")
    end_date = pd.to_datetime(fact_store_daily["date"]).max().strftime("%Y-%m-%d")

    weather_real = get_weather_for_all_stores(dim_stores, start_date, end_date)
    weather_real["date"] = pd.to_datetime(weather_real["date"])

    df = fact_store_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop(columns=["temperature_c"], errors="ignore").merge(
        weather_real, on=["store_id", "date"], how="left"
    )
    # Những ngày quá xa (Open-Meteo Archive thường trễ ~5 ngày so với hiện tại) sẽ bị thiếu
    # -> tạm điền bằng nhiệt độ trung bình đã lấy được, để không làm hỏng dữ liệu huấn luyện
    if df["temperature_c"].isna().any():
        df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].mean())
    return df


if __name__ == "__main__":
    # Ví dụ: lấy nhiệt độ thật cho 9 cửa hàng Circle K (Gò Vấp/Thủ Đức)
    from real_store_data import generate_stores_real

    dim_stores, _, _ = generate_stores_real()
    weather_real = get_weather_for_all_stores(dim_stores, "2024-08-01", "2024-10-31")
    print(weather_real.head(15))
    weather_real.to_csv("thoi_tiet_that_9_cua_hang.csv", index=False, encoding="utf-8-sig")
