"""
data_generator_v3.py
Sinh dữ liệu gần sát thực tế cho hệ thống: Kho trung tâm quản lý nhiều cửa hàng lẻ,
mỗi cửa hàng bán nhiều mặt hàng (SKU). Đáp ứng đầy đủ các bảng cần cho:
  - Báo cáo định kỳ 7 ngày (bán ra / tồn kho theo SKU x cửa hàng)
  - Dự báo nhu cầu AI (có yếu tố lễ/Tết, thời tiết)
  - Cảnh báo cận date (theo dõi lô hàng — batch tracking, tiêu thụ FIFO)
  - Quản lý kho trung tâm (nhập/xuất, vị trí kệ)
  - So sánh tồn kho thực tế vs dự báo nhu cầu

Các bảng (DataFrame) trả về:
  dim_stores      : danh mục cửa hàng
  dim_skus        : danh mục mặt hàng (có hạn dùng, vị trí kệ trong kho)
  weather_daily   : nhiệt độ mỗi ngày theo cụm khu vực
  fact_store_daily: tồn đầu ngày/bán ra/tồn cuối ngày/nhận hàng, theo (store, sku, date)
  fact_store_batches_snapshot: lô hàng còn tồn tại cửa hàng ở NGÀY CUỐI dữ liệu (để demo cảnh báo cận date)
  fact_warehouse_daily: nhập kho từ NCC / xuất kho đi các cửa hàng / tồn kho trung tâm, theo (sku, date)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1) Danh mục cửa hàng (dùng lại logic cụm địa lý)
# ---------------------------------------------------------------------------

def generate_stores(n_stores: int, n_clusters: int = 3, seed: int = 42, area_size: float = 100.0):
    rng = np.random.default_rng(seed)
    cluster_centers = rng.uniform(area_size * 0.15, area_size * 0.85, size=(n_clusters, 2))
    store_cluster = rng.integers(0, n_clusters, size=n_stores)

    rows = []
    coords = np.zeros((n_stores + 1, 2))
    coords[0] = [area_size / 2, area_size / 2]
    for i in range(n_stores):
        c = store_cluster[i]
        offset = rng.normal(0, 8.0, size=2)
        coords[i + 1] = np.clip(cluster_centers[c] + offset, 0, area_size)
        rows.append({
            "store_id": i + 1,
            "store_name": f"CK{i+1:03d}",
            "cluster_id": int(c),
            "x": coords[i + 1, 0], "y": coords[i + 1, 1],
        })
    dim_stores = pd.DataFrame(rows)

    n = n_stores + 1
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])

    return dim_stores, coords, dist


# ---------------------------------------------------------------------------
# 2) Danh mục mặt hàng (SKU) — có hạn dùng, chi phí, vị trí kệ kho
# ---------------------------------------------------------------------------

CATEGORY_PROFILES = {
    # category: (base_demand_range, shelf_life_days_range, weather_sensitivity, holding_cost_range)
    "Đồ uống lạnh":      dict(base=(20, 50), shelf_life=(60, 180),  weather_sens=1.5,  holding=(0.3, 0.6)),
    "Thực phẩm chế biến": dict(base=(10, 30), shelf_life=(1, 3),     weather_sens=0.0,  holding=(0.8, 1.5)),
    "Bánh kẹo":          dict(base=(8, 25),  shelf_life=(90, 270),  weather_sens=0.0,  holding=(0.4, 0.8)),
    "Mì ly/ăn liền":     dict(base=(10, 28), shelf_life=(150, 365), weather_sens=-0.3, holding=(0.3, 0.6)),
    "Sữa/Sản phẩm lạnh": dict(base=(6, 18),  shelf_life=(7, 21),    weather_sens=0.5,  holding=(0.6, 1.2)),
    "Đồ gia dụng":       dict(base=(2, 8),   shelf_life=(365, 730), weather_sens=0.0,  holding=(0.2, 0.4)),
    "Hàng giá trị cao":  dict(base=(1, 5),   shelf_life=(365, 730), weather_sens=0.0,  holding=(1.5, 3.0)),
}


def generate_skus(n_skus: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    categories = list(CATEGORY_PROFILES.keys())
    rows = []
    shelf_zones = ["A", "B", "C", "D"]
    for i in range(n_skus):
        cat = categories[i % len(categories)]
        prof = CATEGORY_PROFILES[cat]
        rows.append({
            "sku_id": i + 1,
            "sku_name": f"{cat} #{i+1}",
            "category": cat,
            "base_demand_min": prof["base"][0],
            "base_demand_max": prof["base"][1],
            "shelf_life_days": int(rng.integers(prof["shelf_life"][0], prof["shelf_life"][1] + 1)),
            "weather_sensitivity": prof["weather_sens"],
            "unit_holding_cost": round(rng.uniform(*prof["holding"]), 2),
            "shelf_zone": shelf_zones[i % len(shelf_zones)],
            "shelf_slot": f"{shelf_zones[i % len(shelf_zones)]}-{(i // len(shelf_zones)) + 1:02d}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) Thời tiết (theo cụm khu vực) — yếu tố ngoại sinh cho dự báo
# ---------------------------------------------------------------------------

def generate_weather(n_clusters: int, n_days: int, seed: int = 42, start_date: str = "2024-01-01"):
    rng = np.random.default_rng(seed + 999)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rows = []
    for c in range(n_clusters):
        base_temp = rng.uniform(24, 30)
        for t in range(n_days):
            doy = dates[t].dayofyear
            seasonal = 4 * np.sin(2 * np.pi * (doy - 60) / 365)  # nóng hơn giữa năm
            noise = rng.normal(0, 1.5)
            temp = base_temp + seasonal + noise
            rows.append({"cluster_id": c, "date": dates[t], "day_idx": t, "temperature_c": round(temp, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4) Mô phỏng vận hành đầy đủ: nhu cầu -> bán hàng -> tồn kho -> lô hàng (FIFO) -> kho trung tâm
# ---------------------------------------------------------------------------

def simulate_operations(
    dim_stores: pd.DataFrame, dim_skus: pd.DataFrame, weather_df: pd.DataFrame,
    n_days: int, seed: int = 42, start_date: str = "2024-01-01",
    replenish_cycle: int = 7,
):
    """
    Mô phỏng dữ liệu vận hành 'hiện trạng' (chưa tối ưu bằng AI — đây là dữ liệu
    lịch sử mà doanh nghiệp SẼ CÓ trong thực tế, dùng để mô hình AI học từ đó).

    - Mỗi `replenish_cycle` ngày, kho trung tâm giao hàng cho từng cửa hàng
      theo chính sách đơn giản (order-up-to theo trung bình động), tạo lô hàng mới.
    - Bán hàng hàng ngày tiêu thụ theo FIFO từ các lô hàng tồn tại cửa hàng.
    - Kho trung tâm nhận hàng từ nhà cung cấp mỗi chu kỳ, xuất hàng đi các cửa hàng.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    n_stores = len(dim_stores)
    weather_map = weather_df.set_index(["cluster_id", "day_idx"])["temperature_c"]
    store_cluster = dim_stores.set_index("store_id")["cluster_id"].to_dict()

    # --- Hiệu ứng Tết dùng chung (giống v2) ---
    tet_effect = np.zeros(n_days)
    tet_day = rng.integers(20, 40)
    for t in range(n_days):
        doy_mod = t % 365
        d = min(abs(doy_mod - tet_day), 365 - abs(doy_mod - tet_day))
        if d <= 12:
            tet_effect[t] = 1.0 + 1.5 * np.exp(-0.5 * (d / 4) ** 2)  # hệ số nhân lên nhu cầu
        else:
            tet_effect[t] = 1.0

    # --- Trạng thái tồn kho ---
    store_batches = {}      # (store_id, sku_id) -> list[{'qty', 'expiry_day'}]
    store_stock_level = {}  # (store_id, sku_id) -> tổng tồn hiện tại (để quyết định châm hàng)
    warehouse_stock = {sku: 0.0 for sku in dim_skus["sku_id"]}

    daily_records = []
    warehouse_records = []
    batch_log = []  # để xuất snapshot cận date cuối kỳ

    base_demand = {}
    for _, sku in dim_skus.iterrows():
        for _, store in dim_stores.iterrows():
            base_demand[(store["store_id"], sku["sku_id"])] = rng.uniform(sku["base_demand_min"], sku["base_demand_max"])

    # Chu kỳ giao hàng RIÊNG cho từng SKU: hàng có hạn dùng ngắn phải giao thường xuyên hơn
    # (tối đa bằng 1/2 hạn dùng, không vượt quá chu kỳ chuẩn `replenish_cycle`)
    sku_cycle = {
        row["sku_id"]: max(1, min(replenish_cycle, row["shelf_life_days"] // 2))
        for _, row in dim_skus.iterrows()
    }

    for t in range(n_days):
        day_of_week = dates[t].dayofweek
        is_weekend = day_of_week >= 5

        # ---- 1) Kho trung tâm: nhận hàng từ NCC định kỳ (riêng theo từng SKU) — CHỈ bù phần thiếu hụt ----
        for sku_id in dim_skus["sku_id"]:
            cyc = sku_cycle[sku_id]
            if t % cyc == 0:
                total_expected = sum(base_demand[(s, sku_id)] for s in dim_stores["store_id"]) * cyc * 1.3
                replenish_qty = max(0.0, total_expected - warehouse_stock[sku_id])  # order-up-to
                replenish_qty = float(np.ceil(replenish_qty))  # số lượng hàng phải là số nguyên (làm tròn lên cho đủ)
                warehouse_stock[sku_id] += replenish_qty
                warehouse_records.append({
                    "date": dates[t], "day_idx": t, "sku_id": sku_id,
                    "received_from_supplier": replenish_qty, "shipped_to_stores": 0.0,
                })

        # ---- 2) Định kỳ: kho xuất hàng cho từng cửa hàng (order-up-to đơn giản, riêng chu kỳ theo SKU) ----
        shipped_today = {sku_id: 0.0 for sku_id in dim_skus["sku_id"]}          # tổng theo SKU (cho sổ kho)
        shipped_today_by_store = {}                                             # (store,sku) -> qty (cho từng cửa hàng)
        for _, store in dim_stores.iterrows():
            sid = store["store_id"]
            for _, sku in dim_skus.iterrows():
                skid = sku["sku_id"]
                cyc = sku_cycle[skid]
                if t % cyc != 0:
                    continue
                key = (sid, skid)
                current = store_stock_level.get(key, 0.0)
                target = base_demand[key] * (cyc + 2)  # đệm 2 ngày an toàn
                order_qty = max(0.0, target - current)
                order_qty = float(np.ceil(order_qty))  # số lượng hàng phải là số nguyên (làm tròn lên cho đủ)
                order_qty = min(order_qty, warehouse_stock[skid])  # không xuất quá tồn kho
                if order_qty > 0:
                    warehouse_stock[skid] -= order_qty
                    shipped_today[skid] += order_qty
                    shipped_today_by_store[key] = order_qty
                    store_stock_level[key] = current + order_qty
                    batch_list = store_batches.setdefault(key, [])
                    batch_list.append({"qty": order_qty, "receive_day": t,
                                        "expiry_day": t + sku["shelf_life_days"]})

        for sku_id, qty in shipped_today.items():
            if qty > 0:
                warehouse_records.append({
                    "date": dates[t], "day_idx": t, "sku_id": sku_id,
                    "received_from_supplier": 0.0, "shipped_to_stores": qty,
                })
        for sku_id in dim_skus["sku_id"]:
            warehouse_records.append({
                "date": dates[t], "day_idx": t, "sku_id": sku_id,
                "received_from_supplier": 0.0, "shipped_to_stores": 0.0,
                "closing_stock_snapshot": warehouse_stock[sku_id],
            })

        # ---- 3) Bán hàng tại từng cửa hàng, tiêu thụ FIFO theo lô ----
        for _, store in dim_stores.iterrows():
            sid = store["store_id"]
            cluster = store_cluster[sid]
            temp = weather_map.get((cluster, t), 27.0)

            for _, sku in dim_skus.iterrows():
                skid = sku["sku_id"]
                key = (sid, skid)
                opening_stock = store_stock_level.get(key, 0.0)

                weekly_effect = 1.15 if is_weekend else 1.0
                weather_effect = 1.0 + sku["weather_sensitivity"] * 0.02 * (temp - 27.0)
                promo_effect = 1.4 if rng.random() < 0.02 else 1.0

                mean_demand = max(1.0, base_demand[key] * weekly_effect * weather_effect
                                   * promo_effect * tet_effect[t])
                demand = rng.poisson(mean_demand)

                units_sold = min(demand, opening_stock)
                lost_sales = max(0.0, demand - opening_stock)

                # tiêu thụ FIFO từ các lô
                remaining_to_consume = units_sold
                batches = store_batches.get(key, [])
                for b in batches:
                    if remaining_to_consume <= 0:
                        break
                    take = min(b["qty"], remaining_to_consume)
                    b["qty"] -= take
                    remaining_to_consume -= take
                batches = [b for b in batches if b["qty"] > 1e-6]
                store_batches[key] = batches

                closing_stock = opening_stock - units_sold
                store_stock_level[key] = closing_stock

                daily_records.append({
                    "date": dates[t], "day_idx": t, "store_id": sid, "sku_id": skid,
                    "day_of_week": day_of_week, "is_weekend": int(is_weekend),
                    "temperature_c": temp, "is_tet_period": int(tet_effect[t] > 1.01),
                    "opening_stock": round(opening_stock, 1),
                    "units_sold": int(units_sold),
                    "lost_sales": round(lost_sales, 1),
                    "closing_stock": round(closing_stock, 1),
                    "units_received": round(shipped_today_by_store.get(key, 0.0), 1),
                })

    fact_store_daily = pd.DataFrame(daily_records)
    fact_warehouse_daily = pd.DataFrame(warehouse_records)

    # --- Snapshot lô hàng còn tồn ở ngày cuối cùng, để demo cảnh báo cận date ---
    snapshot_day = n_days - 1
    snap_rows = []
    for (sid, skid), batches in store_batches.items():
        for b in batches:
            days_to_expiry = b["expiry_day"] - snapshot_day
            snap_rows.append({
                "store_id": sid, "sku_id": skid, "qty_remaining": round(b["qty"], 1),
                "receive_day": b["receive_day"], "expiry_day": b["expiry_day"],
                "days_to_expiry": days_to_expiry,
            })
    fact_store_batches_snapshot = pd.DataFrame(snap_rows)

    return fact_store_daily, fact_warehouse_daily, fact_store_batches_snapshot
