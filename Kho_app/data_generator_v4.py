"""
data_generator_v4.py
Mở rộng data_generator_v3.py để đầy đủ dữ liệu theo đúng đặc tả:

KHO TRUNG TÂM:
  - dim_warehouse       : tọa độ kho trung tâm, tổng sức chứa
  - dim_shelves         : danh sách kệ (1..5), mục đích từng kệ, sức chứa từng kệ
  - dim_skus (mở rộng)  : thêm cột kich_thuoc (nhỏ/cồng kềnh), warehouse_shelf_id (kệ được xếp)
  - fact_warehouse_inbound  : mặt hàng nhập kho, ngày nhập, hạn sử dụng, kệ nào
  - fact_warehouse_outbound : mặt hàng xuất kho, ngày xuất, xuất cho cửa hàng nào, còn lại bao nhiêu
  - fact_shelf_stock_daily  : tồn kho theo TỪNG KỆ theo ngày (để kiểm tra vượt sức chứa kệ)

ĐỘI XE & TÀI XẾ:
  - dim_vehicles : loại xe, tải trọng, thể tích
  - dim_drivers  : tài xế, số giờ làm tối đa/ngày
  - fact_fleet_daily : mỗi ngày xe nào/tài xế nào rảnh hay đang bận, đã làm bao nhiêu giờ

CỬA HÀNG (Circle K): dùng lại toàn bộ từ data_generator_v3.py (đã có nhập/bán mỗi ngày,
  thời tiết, lễ Tết, tọa độ cửa hàng) — không cần viết lại.
"""

import numpy as np
import pandas as pd

from data_generator_v3 import (
    generate_stores, generate_skus, generate_weather, simulate_operations,
    CATEGORY_PROFILES,
)
from real_store_data import generate_stores_real


# ---------------------------------------------------------------------------
# 1) Kho trung tâm — tọa độ, sức chứa tổng
# ---------------------------------------------------------------------------

def generate_warehouse(coords, total_capacity: float = 50000.0):
    """coords lấy từ generate_stores() — index 0 chính là kho trung tâm."""
    return pd.DataFrame([{
        "warehouse_id": 1,
        "x": coords[0, 0], "y": coords[0, 1],
        "tong_suc_chua": total_capacity,
    }])


# ---------------------------------------------------------------------------
# 2) Kệ kho trung tâm — 5 kệ, mỗi kệ một công năng khác nhau
# ---------------------------------------------------------------------------

SHELF_DEFINITIONS = [
    # Sức chứa được tính lại dựa trên mức tồn kho CAO NHẤT thực tế từng quan sát được
    # trong dữ liệu (không phải số tùy ý) + biên độ an toàn ~40-90% tuỳ mức độ biến động
    # của từng nhóm hàng — tránh dư thừa sức chứa gây tính sai chi phí lưu kho.
    {"shelf_id": 1, "ten_ke": "Kệ 1", "cong_nang": "Hàng tần suất xuất/nhập cao hoặc cồng kềnh", "suc_chua": 3500},
    {"shelf_id": 2, "ten_ke": "Kệ 2", "cong_nang": "Hàng cần bảo quản lạnh / hạn dùng ngắn", "suc_chua": 350},
    {"shelf_id": 3, "ten_ke": "Kệ 3", "cong_nang": "Hàng khô, bảo quản lâu", "suc_chua": 400},
    {"shelf_id": 4, "ten_ke": "Kệ 4", "cong_nang": "Hàng giá trị cao / dễ vỡ", "suc_chua": 120},
    {"shelf_id": 5, "ten_ke": "Kệ 5", "cong_nang": "Hàng dự phòng / tồn kho an toàn", "suc_chua": 200},
]


def generate_shelves():
    return pd.DataFrame(SHELF_DEFINITIONS)


def assign_shelves_to_skus(dim_skus: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Gán mỗi SKU vào 1 kệ kho trung tâm, dựa trên:
      - Tần suất xuất/nhập (proxy: base_demand trung bình cao -> Kệ 1)
      - Kích thước (cồng kềnh -> Kệ 1)
      - Hạn dùng ngắn (cần bảo quản lạnh -> Kệ 2)
      - Còn lại -> Kệ 3 (hàng khô) hoặc Kệ 4/5 tuỳ ngẫu nhiên có kiểm soát
    """
    rng = np.random.default_rng(seed)
    df = dim_skus.copy()
    df["kich_thuoc"] = rng.choice(["Nhỏ gọn", "Cồng kềnh"], size=len(df), p=[0.75, 0.25])
    df["avg_base_demand"] = (df["base_demand_min"] + df["base_demand_max"]) / 2

    turnover_threshold = df["avg_base_demand"].quantile(0.6)

    def pick_shelf(row):
        if row["category"] == "Hàng giá trị cao":
            return 4  # Kệ 4: hàng giá trị cao / dễ vỡ — cố định theo đúng công năng kệ
        if row["category"] == "Đồ gia dụng":
            return 5  # Kệ 5: hàng dự phòng/tần suất thấp — cố định theo đúng công năng kệ
        if row["avg_base_demand"] >= turnover_threshold or row["kich_thuoc"] == "Cồng kềnh":
            return 1
        if row["shelf_life_days"] <= 21:
            return 2
        if row["category"] in ("Bánh kẹo", "Mì ly/ăn liền"):
            return 3
        return 5

    df["warehouse_shelf_id"] = df.apply(pick_shelf, axis=1)
    return df


# ---------------------------------------------------------------------------
# 3) Đội xe & tài xế
# ---------------------------------------------------------------------------

VEHICLE_TYPES = [
    # co_ngan_lanh: xe máy KHÔNG có ngăn lạnh (chỉ chở được hàng thường);
    # xe tải nhỏ/trung MẶC ĐỊNH có trang bị ngăn lạnh (thực tế xe tải phân phối
    # FMCG/cửa hàng tiện lợi thường được trang bị thùng lạnh do cần chở đồ uống/sữa)
    {"loai_xe": "Xe máy chở hàng", "tai_trong_kg": 150, "the_tich_m3": 0.1, "co_ngan_lanh": False},
    {"loai_xe": "Xe tải nhỏ (1 tấn)", "tai_trong_kg": 1000, "the_tich_m3": 6.0, "co_ngan_lanh": True},
    {"loai_xe": "Xe tải trung (3.5 tấn)", "tai_trong_kg": 3500, "the_tich_m3": 15.0, "co_ngan_lanh": True},
]


def generate_vehicle_fleet(n_vehicles: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_vehicles):
        vt = VEHICLE_TYPES[rng.integers(0, len(VEHICLE_TYPES))]
        rows.append({
            "vehicle_id": i + 1,
            "bien_so": f"51C-{10000+i}",
            "loai_xe": vt["loai_xe"],
            "tai_trong_kg": vt["tai_trong_kg"],
            "the_tich_m3": vt["the_tich_m3"],
            "co_ngan_lanh": vt["co_ngan_lanh"],
        })
    return pd.DataFrame(rows)


def generate_drivers(n_drivers: int, seed: int = 42, driver_names: list = None):
    """driver_names: danh sách tên thật (tuỳ chọn) — nếu truyền vào, dùng đúng số lượng
    và tên đó thay vì tên mặc định 'Tài xế 01, 02...'; n_drivers sẽ tự khớp theo độ dài danh sách."""
    rng = np.random.default_rng(seed)
    if driver_names:
        n_drivers = len(driver_names)
    rows = []
    for i in range(n_drivers):
        name = driver_names[i] if driver_names else f"Tài xế {i+1:02d}"
        rows.append({
            "driver_id": i + 1,
            "ten_tai_xe": name,
            "gio_lam_toi_da_ngay": int(rng.choice([8, 10])),
        })
    return pd.DataFrame(rows)


def simulate_fleet_daily(dim_vehicles: pd.DataFrame, dim_drivers: pd.DataFrame, n_days: int,
                          seed: int = 42, start_date: str = "2024-01-01"):
    """
    [Phiên bản cũ — lịch làm việc NGẪU NHIÊN, không gắn với nhu cầu giao hàng thực tế]
    Giữ lại để tương thích ngược; khuyến nghị dùng simulate_fleet_daily_linked() thay thế.
    """
    rng = np.random.default_rng(seed + 7)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    n_vehicles, n_drivers = len(dim_vehicles), len(dim_drivers)

    rows = []
    for t in range(n_days):
        maintenance_vehicles = set(dim_vehicles["vehicle_id"][rng.random(n_vehicles) < 0.05])
        off_drivers = set(dim_drivers["driver_id"][rng.random(n_drivers) < 0.08])
        available_vehicles = [v for v in dim_vehicles["vehicle_id"] if v not in maintenance_vehicles]
        rng.shuffle(available_vehicles)

        vi = 0
        for _, drv in dim_drivers.iterrows():
            did = drv["driver_id"]
            max_hours = drv["gio_lam_toi_da_ngay"]
            if did in off_drivers:
                rows.append({"date": dates[t], "day_idx": t, "driver_id": did, "vehicle_id": None,
                             "so_gio_da_lam": 0, "trang_thai": "Nghỉ phép"})
                continue
            hours_worked = round(rng.uniform(0, max_hours), 1)
            if vi < len(available_vehicles) and hours_worked > 0:
                vid = available_vehicles[vi]; vi += 1
                trang_thai = "Đang giao hàng" if hours_worked >= max_hours * 0.5 else "Rảnh (đang ở kho)"
            else:
                vid = None
                trang_thai = "Rảnh (chưa phân xe)" if hours_worked == 0 else "Chờ xe"
            rows.append({"date": dates[t], "day_idx": t, "driver_id": did, "vehicle_id": vid,
                         "so_gio_da_lam": hours_worked, "trang_thai": trang_thai})

        for v in maintenance_vehicles:
            rows.append({"date": dates[t], "day_idx": t, "driver_id": None, "vehicle_id": v,
                         "so_gio_da_lam": 0, "trang_thai": "Xe đang bảo trì"})

    return pd.DataFrame(rows)


def simulate_fleet_daily_linked(
    dim_vehicles: pd.DataFrame, dim_drivers: pd.DataFrame, fact_warehouse_outbound: pd.DataFrame,
    dist: np.ndarray, n_days: int, seed: int = 42, start_date: str = "2024-01-01",
    avg_speed_kmh: float = 30.0, service_hours_per_stop: float = 0.5,
):
    """
    Lịch làm việc đội xe/tài xế GẮN VỚI nhu cầu giao hàng thực tế mỗi ngày:
    - Với mỗi cửa hàng cần nhận hàng trong ngày (suy từ fact_warehouse_outbound),
      tính thời gian 1 chuyến đi-về (khứ hồi kho <-> cửa hàng) theo khoảng cách thực (dist)
      + thời gian giao nhận tại điểm (service_hours_per_stop).
    - Phân công tài xế còn đủ giờ làm + có xe rảnh cho từng chuyến, tuần tự theo cửa hàng.
    - Tài xế/xe không được phân chuyến nào -> Rảnh ở kho (trừ khi nghỉ phép/bảo trì ngẫu nhiên).
    """
    rng = np.random.default_rng(seed + 7)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    daily_stores_needed = fact_warehouse_outbound.groupby("day_idx")["store_id"].unique().to_dict()
    driver_max_hours = dim_drivers.set_index("driver_id")["gio_lam_toi_da_ngay"].to_dict()

    rows = []
    for t in range(n_days):
        stores_today = list(dict.fromkeys(daily_stores_needed.get(t, [])))  # loại trùng, giữ thứ tự

        maintenance_vehicles = set(dim_vehicles["vehicle_id"][rng.random(len(dim_vehicles)) < 0.05])
        off_drivers = set(dim_drivers["driver_id"][rng.random(len(dim_drivers)) < 0.08])

        available_vehicles = [v for v in dim_vehicles["vehicle_id"] if v not in maintenance_vehicles]
        available_drivers = [d for d in dim_drivers["driver_id"] if d not in off_drivers]
        rng.shuffle(available_vehicles)
        rng.shuffle(available_drivers)

        hours_left = {d: driver_max_hours[d] for d in available_drivers}
        hours_worked = {d: 0.0 for d in available_drivers}
        assigned_vehicle = {}
        status = {d: "Rảnh (đang ở kho)" for d in available_drivers}

        vi = 0
        for store_id in stores_today:
            trip_distance = 2 * dist[0, store_id]  # khứ hồi kho <-> cửa hàng
            trip_hours = trip_distance / avg_speed_kmh + service_hours_per_stop

            for d in available_drivers:
                if hours_left[d] >= trip_hours and vi < len(available_vehicles):
                    if d not in assigned_vehicle:
                        assigned_vehicle[d] = available_vehicles[vi]; vi += 1
                    hours_left[d] -= trip_hours
                    hours_worked[d] += trip_hours
                    status[d] = "Đang giao hàng"
                    break
            # Nếu không tài xế nào đủ giờ/xe rảnh: chuyến này không thực hiện được trong ngày
            # (dấu hiệu quá tải đội xe — có thể dùng để cảnh báo thiếu nguồn lực vận chuyển)

        for d in available_drivers:
            rows.append({
                "date": dates[t], "day_idx": t, "driver_id": d,
                "vehicle_id": assigned_vehicle.get(d, None),
                "so_gio_da_lam": round(hours_worked[d], 1),
                "trang_thai": status[d],
            })
        for d in off_drivers:
            rows.append({"date": dates[t], "day_idx": t, "driver_id": d, "vehicle_id": None,
                         "so_gio_da_lam": 0.0, "trang_thai": "Nghỉ phép"})
        for v in maintenance_vehicles:
            rows.append({"date": dates[t], "day_idx": t, "driver_id": None, "vehicle_id": v,
                         "so_gio_da_lam": 0.0, "trang_thai": "Xe đang bảo trì"})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4) Góc nhìn Nhập/Xuất/Tồn kho theo KỆ — suy ra từ dữ liệu vận hành v3 (không mô phỏng lại)
# ---------------------------------------------------------------------------

def build_warehouse_shelf_views(fact_store_daily: pd.DataFrame, fact_warehouse_daily: pd.DataFrame,
                                 dim_skus_with_shelf: pd.DataFrame, dim_stores: pd.DataFrame):
    sku_shelf_map = dim_skus_with_shelf.set_index("sku_id")["warehouse_shelf_id"]
    sku_life_map = dim_skus_with_shelf.set_index("sku_id")["shelf_life_days"]

    # ---- Nhập kho: những ngày received_from_supplier > 0 ----
    inbound = fact_warehouse_daily[fact_warehouse_daily["received_from_supplier"] > 0].copy()
    inbound["warehouse_shelf_id"] = inbound["sku_id"].map(sku_shelf_map)
    inbound["han_su_dung"] = inbound["date"] + pd.to_timedelta(inbound["sku_id"].map(sku_life_map), unit="D")
    fact_warehouse_inbound = inbound[["date", "day_idx", "sku_id", "received_from_supplier",
                                       "warehouse_shelf_id", "han_su_dung"]].rename(
        columns={"received_from_supplier": "so_luong_nhap"})

    # ---- Xuất kho: suy từ units_received > 0 ở cấp cửa hàng (chính là lượng kho xuất cho cửa hàng đó) ----
    outbound = fact_store_daily[fact_store_daily["units_received"] > 0].copy()
    outbound = outbound.merge(dim_stores[["store_id", "store_name"]], on="store_id")
    fact_warehouse_outbound = outbound[["date", "day_idx", "sku_id", "store_id", "store_name", "units_received"]] \
        .rename(columns={"units_received": "so_luong_xuat"})

    # ---- Tồn kho theo kệ mỗi ngày ----
    stock = fact_warehouse_daily.dropna(subset=["closing_stock_snapshot"]).copy()
    stock["warehouse_shelf_id"] = stock["sku_id"].map(sku_shelf_map)
    fact_shelf_stock_daily = stock.groupby(["date", "day_idx", "warehouse_shelf_id"])["closing_stock_snapshot"] \
        .sum().reset_index().rename(columns={"closing_stock_snapshot": "ton_kho_ke"})

    return fact_warehouse_inbound, fact_warehouse_outbound, fact_shelf_stock_daily


# ---------------------------------------------------------------------------
# 5) Hàm tổng hợp — tạo toàn bộ bộ dữ liệu trong 1 lần gọi
# ---------------------------------------------------------------------------

def build_full_dataset(n_stores=8, n_skus=12, n_clusters=3, n_days=200,
                        n_vehicles=5, n_drivers=6, replenish_cycle=7, seed=42,
                        use_real_stores=False, driver_names=None):
    if use_real_stores:
        dim_stores, coords, dist = generate_stores_real(n_clusters=n_clusters, seed=seed)
        n_stores = len(dim_stores)  # số cửa hàng thật cố định (9), bỏ qua slider n_stores
    else:
        dim_stores, coords, dist = generate_stores(n_stores, n_clusters, seed=seed)
    dim_skus = generate_skus(n_skus, seed=seed)
    dim_skus = assign_shelves_to_skus(dim_skus, seed=seed)
    weather_df = generate_weather(n_clusters, n_days, seed=seed)

    fact_store_daily, fact_warehouse_daily, batch_snapshot = simulate_operations(
        dim_stores, dim_skus, weather_df, n_days, seed=seed, replenish_cycle=replenish_cycle,
    )

    dim_warehouse = generate_warehouse(coords)
    dim_shelves = generate_shelves()
    dim_vehicles = generate_vehicle_fleet(n_vehicles, seed=seed)
    dim_drivers = generate_drivers(n_drivers, seed=seed, driver_names=driver_names)

    fact_warehouse_inbound, fact_warehouse_outbound, fact_shelf_stock_daily = build_warehouse_shelf_views(
        fact_store_daily, fact_warehouse_daily, dim_skus, dim_stores,
    )

    fact_fleet_daily = simulate_fleet_daily_linked(
        dim_vehicles, dim_drivers, fact_warehouse_outbound, dist, n_days, seed=seed,
    )

    return dict(
        dim_stores=dim_stores, dim_skus=dim_skus, dim_warehouse=dim_warehouse, dim_shelves=dim_shelves,
        dim_vehicles=dim_vehicles, dim_drivers=dim_drivers, weather_df=weather_df,
        fact_store_daily=fact_store_daily, fact_warehouse_daily=fact_warehouse_daily,
        batch_snapshot=batch_snapshot, fact_fleet_daily=fact_fleet_daily,
        fact_warehouse_inbound=fact_warehouse_inbound, fact_warehouse_outbound=fact_warehouse_outbound,
        fact_shelf_stock_daily=fact_shelf_stock_daily, coords=coords, dist=dist,
    )
