"""
distribution_planning.py
Lên kế hoạch phân phối cho TƯƠNG LAI (dựa trên dự báo nhu cầu từ Tầng 3):
  1. CỬA XUẤT của mỗi MẶT HÀNG phụ thuộc vào KỆ nó được lưu trữ (kệ nào gần cửa nào
     thì hàng ở kệ đó xuất qua cửa đó) — đây là thuộc tính CỐ ĐỊNH theo vị trí lưu trữ
     trong kho, KHÔNG phụ thuộc vào việc hàng đó giao cho cửa hàng nào.
  2. Việc gom TUYẾN GIAO HÀNG (chọn xe, tài xế) vẫn theo cụm địa lý các cửa hàng đích
     đến (gọi là KHU VỰC giao hàng) — đây là khái niệm KHÁC với cửa xuất kho, vì 1
     tuyến giao hàng có thể cần lấy hàng từ nhiều cửa xuất khác nhau (do các cửa hàng
     trên tuyến cần nhiều mặt hàng nằm ở các kệ/cửa khác nhau).

GIẢ ĐỊNH CẦN LƯU Ý: hệ thống chưa có dữ liệu khối lượng/trọng lượng thật của từng
SKU, nên dùng trọng lượng trung bình ước tính (mặc định 0.5 kg/đơn vị) để quy đổi
sức chứa xe (kg) sang số lượng sản phẩm tối đa chở được. Nên thay bằng số liệu
thật của từng mặt hàng nếu có, để kết quả chính xác hơn.
"""

import numpy as np
import pandas as pd
from map_view import solve_optimal_route
from product_catalog import build_product_catalog, compute_volume_needed, compute_volume_by_packaging_type, select_best_fit_vehicle

AVG_UNIT_WEIGHT_KG = 0.5  # dự phòng — chỉ dùng khi không có danh mục thể tích (build_product_catalog)


def assign_gates_to_shelves(dim_shelves: pd.DataFrame, n_gates: int = 2, shelf_gate_map: dict = None) -> pd.DataFrame:
    """
    Gán mỗi KỆ vào 1 cửa xuất — mặc định chia đều theo thứ tự kệ (round-robin),
    nhưng có thể truyền `shelf_gate_map` (vd {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}) nếu bạn
    biết chính xác kệ nào nằm gần cửa xuất nào trong sơ đồ kho thật.
    """
    df = dim_shelves.copy()
    if shelf_gate_map:
        df["cua_xuat"] = df["shelf_id"].map(shelf_gate_map)
    else:
        df["cua_xuat"] = [(i % n_gates) + 1 for i in range(len(df))]
    return df


def assign_exit_gates(dim_stores: pd.DataFrame, n_gates: int = 2) -> pd.DataFrame:
    """Gom cửa hàng theo cụm địa lý -> dùng để lập TUYẾN GIAO HÀNG (khu vực), KHÔNG
    phải cửa xuất kho (cửa xuất kho nay phụ thuộc vào kệ — xem assign_gates_to_shelves)."""
    clusters = sorted(dim_stores["cluster_id"].unique())
    zone_map = {c: (i % n_gates) + 1 for i, c in enumerate(clusters)}
    df = dim_stores.copy()
    df["khu_vuc"] = df["cluster_id"].map(zone_map)
    return df


def plan_future_distribution(
    forecast_detail: pd.DataFrame, dim_stores: pd.DataFrame, dim_skus: pd.DataFrame,
    dim_shelves: pd.DataFrame, dim_vehicles: pd.DataFrame, dim_drivers: pd.DataFrame, dist: np.ndarray,
    n_gates: int = 2, shelf_gate_map: dict = None, max_route_hours: float = 4.0,
    avg_speed_kmh: float = 30.0, service_hours_per_stop: float = 0.5,
    avg_unit_weight_kg: float = AVG_UNIT_WEIGHT_KG,
) -> dict:
    """
    forecast_detail: kết quả dự báo từ Tầng 3 (có cột store_id, sku_id, q50 — nhu cầu tuần tới).
    Việc gom tuyến giao hàng dùng ĐÚNG thuật toán Clarke-Wright Savings như trang
    "Ghép tuyến" (route_merging.py) — không tự chia đều theo cụm địa lý cố định nữa,
    để 2 trang cho ra kết quả nhất quán với nhau.
    Trả về: {"route_plan": DataFrame, "sku_gate_plan": DataFrame, "store_zones": DataFrame,
             "shelf_gates": DataFrame}
    """
    from route_merging import clarke_wright_merge
    # ---- 1) Cửa xuất theo KỆ (cố định theo vị trí lưu trữ, không phụ thuộc cửa hàng đích) ----
    shelf_gates = assign_gates_to_shelves(dim_shelves, n_gates=n_gates, shelf_gate_map=shelf_gate_map)
    sku_gate_map = dim_skus.merge(shelf_gates[["shelf_id", "cua_xuat"]],
                                   left_on="warehouse_shelf_id", right_on="shelf_id")[["sku_id", "cua_xuat"]] \
        .set_index("sku_id")["cua_xuat"]

    demand = forecast_detail.groupby(["store_id", "sku_id"])["q50"].sum().reset_index()
    demand = demand.merge(dim_skus[["sku_id", "sku_name", "warehouse_shelf_id"]], on="sku_id")
    demand["cua_xuat"] = demand["sku_id"].map(sku_gate_map)

    sku_gate_plan = demand.groupby(["cua_xuat", "sku_id", "sku_name", "warehouse_shelf_id"])["q50"].sum().reset_index()
    sku_gate_plan.columns = ["Cửa xuất", "sku_id", "Mặt hàng", "Kệ", "Tổng nhu cầu tuần tới"]
    sku_gate_plan["Cửa xuất"] = "Cửa xuất " + sku_gate_plan["Cửa xuất"].astype(str)
    sku_gate_plan["Kệ"] = "Kệ " + sku_gate_plan["Kệ"].astype(str)
    sku_gate_plan = sku_gate_plan[["Cửa xuất", "Kệ", "Mặt hàng", "Tổng nhu cầu tuần tới"]].round(0) \
        .sort_values(["Cửa xuất", "Kệ", "Tổng nhu cầu tuần tới"], ascending=[True, True, False])

    # ---- 2) Gom TUYẾN GIAO HÀNG bằng Clarke-Wright (giống hệt trang "Ghép tuyến") ----
    # Chỉ những cửa hàng THỰC SỰ CẦN giao (nhu cầu > 0) mới được đưa vào thuật toán ghép tuyến
    stores_needing_delivery = demand[demand["q50"] > 0]["store_id"].unique().tolist()
    merged_routes = clarke_wright_merge(dist, stores_needing_delivery, avg_speed_kmh,
                                         service_hours_per_stop, max_route_hours, depot=0)

    catalog = build_product_catalog(dim_skus)

    driver_hours_left = dim_drivers.set_index("driver_id")["gio_lam_toi_da_ngay"].to_dict()
    driver_queue = list(dim_drivers["driver_id"])

    # Vẫn giữ lại thông tin cụm địa lý (cluster_id có sẵn) để hiển thị tham khảo, không dùng để gom tuyến nữa
    store_zones = dim_stores.copy()
    store_zones["khu_vuc"] = store_zones["cluster_id"] + 1

    route_rows = []
    for route_num, stores_needed in enumerate(merged_routes, start=1):
        zone_demand = demand[demand["store_id"].isin(stores_needed)]
        total_units = zone_demand["q50"].sum()
        gates_needed = sorted(zone_demand[zone_demand["q50"] > 0]["cua_xuat"].dropna().unique().tolist())

        route, route_km = solve_optimal_route(dist, stores_needed, depot=0)
        route_hours = route_km / avg_speed_kmh + len(stores_needed) * service_hours_per_stop

        # Chọn xe theo THỂ TÍCH THẬT, tách riêng hàng LẠNH/THƯỜNG — hàng lạnh BẮT BUỘC
        # phải lên xe có ngăn lạnh (co_ngan_lanh=True), tránh xếp nhầm làm hỏng hàng
        demand_dict = dict(zip(zone_demand["sku_id"], zone_demand["q50"]))
        vol_split = compute_volume_by_packaging_type(demand_dict, catalog)
        can_cold = vol_split["lanh"] > 0
        result = select_best_fit_vehicle(vol_split["tong"], dim_vehicles, requires_cold=can_cold)

        if result["loi"]:
            vehicle = pd.Series({"loai_xe": "⚠ Không có xe phù hợp"})
            note_vehicle = f" — {result['loi']}"
        else:
            vehicle = result["vehicle"]
            cold_note = " [có hàng lạnh, bắt buộc xe ngăn lạnh]" if can_cold else ""
            if result["du_the_tich"]:
                note_vehicle = f" (trống ~{result['empty_pct']:.0f}% thể tích){cold_note}"
            else:
                note_vehicle = f" (⚠ vượt thể tích xe lớn nhất — cần chia nhiều chuyến){cold_note}"

        assigned_driver = None
        for d in sorted(driver_queue, key=lambda x: -driver_hours_left.get(x, 0)):
            if driver_hours_left.get(d, 0) >= route_hours:
                assigned_driver = d
                driver_hours_left[d] -= route_hours
                break

        store_names = dim_stores.set_index("store_id").loc[stores_needed, "store_name"].tolist()
        driver_name = dim_drivers.set_index("driver_id").loc[assigned_driver, "ten_tai_xe"] if assigned_driver else "⚠ Không đủ tài xế rảnh"

        route_rows.append({
            "Tuyến giao hàng": f"Tuyến {route_num}",
            "Cửa hàng trên tuyến": ", ".join(store_names),
            "Cần lấy hàng từ": ", ".join(f"Cửa xuất {g}" for g in gates_needed),
            "Tổng nhu cầu (đơn vị)": round(total_units, 0),
            "Loại xe đề xuất": vehicle["loai_xe"] + note_vehicle,
            "Tài xế đề xuất": driver_name,
            "Quãng đường (km)": round(route_km, 1),
            "Thời gian ước tính (giờ)": round(route_hours, 1),
        })

    route_plan = pd.DataFrame(route_rows)
    return {"route_plan": route_plan, "sku_gate_plan": sku_gate_plan,
            "store_zones": store_zones, "shelf_gates": shelf_gates}


DAY_NAMES_VI = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]


def build_weekly_driver_schedule(
    dim_stores: pd.DataFrame, dim_vehicles: pd.DataFrame, dim_drivers: pd.DataFrame,
    fact_warehouse_outbound: pd.DataFrame, dist: np.ndarray,
    avg_speed_kmh: float = 30.0, service_hours_per_stop: float = 0.5,
) -> dict:
    """
    Ghép CỐ ĐỊNH mỗi tài xế với 1 xe và 1 khu vực giao hàng riêng cho cả tuần
    (số tài xế = số xe = số khu vực). Lịch làm việc từng ngày trong tuần dựa trên
    NHỊP GIAO HÀNG THẬT đã quan sát được trong lịch sử (7 ngày gần nhất): nếu khu vực
    đó từng có cửa hàng nhận hàng vào đúng thứ đó trong tuần, giả định tuần tới lặp
    lại nhịp tương tự -> tài xế đi làm; ngược lại -> ngày nghỉ.

    Trả về: {"assignment": DataFrame (tài xế-xe-khu vực), "schedule": DataFrame (lịch theo ngày)}
    """
    n = min(len(dim_vehicles), len(dim_drivers))
    store_zones = assign_exit_gates(dim_stores, n_gates=n)  # n khu vực = n tài xế/xe
    zone_map = store_zones.set_index("store_id")["khu_vuc"]

    drivers = dim_drivers.iloc[:n].reset_index(drop=True)
    vehicles = dim_vehicles.iloc[:n].reset_index(drop=True)

    assignment = pd.DataFrame({
        "Khu vực": [f"Khu vực {i+1}" for i in range(n)],
        "Tài xế": drivers["ten_tai_xe"].values,
        "Xe": vehicles["loai_xe"].values + " (" + vehicles["bien_so"].values + ")",
        "Cửa hàng phụ trách": [
            ", ".join(store_zones[store_zones["khu_vuc"] == i + 1]["store_name"]) for i in range(n)
        ],
    })

    # ---- Nhịp giao hàng thật: ngày trong tuần nào khu vực đó có phát sinh giao hàng ----
    ob = fact_warehouse_outbound.copy()
    ob["date"] = pd.to_datetime(ob["date"])
    ob["khu_vuc"] = ob["store_id"].map(zone_map)
    ob["thu"] = ob["date"].dt.dayofweek  # 0=Thứ 2 ... 6=Chủ Nhật

    last_7_days = sorted(ob["day_idx"].unique())[-7:]
    ob_recent = ob[ob["day_idx"].isin(last_7_days)]

    schedule_rows = []
    for day_num in range(7):
        for i in range(n):
            zone_stores_today = ob_recent[(ob_recent["thu"] == day_num) & (ob_recent["khu_vuc"] == i + 1)]["store_id"].unique().tolist()
            if len(zone_stores_today) > 0:
                _, route_km = solve_optimal_route(dist, zone_stores_today, depot=0)
                route_hours = route_km / avg_speed_kmh + len(zone_stores_today) * service_hours_per_stop
                status = "Đi làm"
                store_names = ", ".join(dim_stores.set_index("store_id").loc[zone_stores_today, "store_name"])
            else:
                route_km, route_hours, status, store_names = 0.0, 0.0, "Nghỉ", "—"

            schedule_rows.append({
                "Ngày": DAY_NAMES_VI[day_num],
                "Tài xế": drivers.iloc[i]["ten_tai_xe"],
                "Trạng thái": status,
                "Cửa hàng giao hôm đó": store_names,
                "Quãng đường (km)": round(route_km, 1),
                "Giờ làm ước tính": round(route_hours, 1),
            })

    schedule = pd.DataFrame(schedule_rows)
    return {"assignment": assignment, "schedule": schedule}
