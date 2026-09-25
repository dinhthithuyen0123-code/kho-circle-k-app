"""
route_merging.py
Tính toán đề xuất GHÉP TUYẾN (route consolidation): so sánh chi phí nếu giao riêng
từng cửa hàng (mỗi cửa hàng 1 chuyến xe đi-về) với chi phí nếu gộp nhiều cửa hàng
vào 1 chuyến duy nhất (dùng thuật toán TSP đã có — Nearest-Neighbor + 2-opt).

Toàn bộ số liệu tính từ dữ liệu THẬT của hệ thống (ma trận khoảng cách Haversine,
lịch sử xuất kho), không phải dữ liệu minh hoạ.
"""

import itertools
import numpy as np
import pandas as pd
from map_view import solve_optimal_route


def compute_daily_route_comparison(fact_warehouse_outbound: pd.DataFrame, dist: np.ndarray,
                                    cost_per_km: float = 1.2, fixed_dispatch_cost: float = 25.0,
                                    n_days: int = 7) -> pd.DataFrame:
    """
    Với mỗi ngày gần đây có phát sinh giao hàng: so sánh
      - TRƯỚC ghép tuyến: mỗi cửa hàng 1 chuyến xe đi-về riêng (depot -> cửa hàng -> depot)
      - SAU ghép tuyến: 1 chuyến duy nhất đi qua tất cả cửa hàng cần giao hôm đó (TSP tối ưu)
    Trả về DataFrame: ngày, chi phí/km trước, chi phí/km sau, quãng đường tiết kiệm (km).
    """
    daily_stores = fact_warehouse_outbound.groupby("day_idx")["store_id"].unique()
    recent_days = sorted(daily_stores.index)[-n_days:]

    rows = []
    for day in recent_days:
        stores_today = list(dict.fromkeys(daily_stores[day]))
        if len(stores_today) == 0:
            continue

        before_dist = sum(2 * dist[0, s] for s in stores_today)
        before_cost = before_dist * cost_per_km + len(stores_today) * fixed_dispatch_cost

        _, after_dist = solve_optimal_route(dist, stores_today, depot=0)
        after_cost = after_dist * cost_per_km + fixed_dispatch_cost

        rows.append({
            "day_idx": day, "so_cua_hang": len(stores_today),
            "chi_phi_km_truoc": before_cost / max(before_dist, 1e-6),
            "chi_phi_km_sau": after_cost / max(after_dist, 1e-6),
            "quang_duong_truoc_km": before_dist, "quang_duong_sau_km": after_dist,
            "quang_duong_tiet_kiem_km": before_dist - after_dist,
        })

    return pd.DataFrame(rows)


def clarke_wright_merge(dist: np.ndarray, store_ids: list, avg_speed_kmh: float = 30.0,
                         service_hours_per_stop: float = 0.5, max_route_hours: float = 4.0,
                         volume_per_store: dict = None, max_vehicle_volume_liters: float = None,
                         depot: int = 0) -> list:
    """
    Thuật toán CLARKE-WRIGHT SAVINGS — thuật toán kinh điển trong Operations Research
    (Clarke & Wright, 1964) để TỰ ĐỘNG gộp nhiều điểm giao hàng riêng lẻ thành các
    tuyến đa điểm tối ưu, KHÔNG cần người dùng tự chọn từng tổ hợp.

    Ý tưởng: bắt đầu với mỗi cửa hàng là 1 tuyến riêng (Kho-A-Kho, Kho-B-Kho...).
    Với mỗi cặp (i,j), "mức tiết kiệm" nếu ghép chung = dist(Kho,i) + dist(Kho,j) - dist(i,j).
    Ghép dần theo mức tiết kiệm giảm dần, miễn là không vượt giới hạn thời gian tuyến
    (max_route_hours) và 2 điểm đang ở ĐẦU/CUỐI tuyến của chúng (không phải điểm giữa).

    Trả về: danh sách các tuyến cuối cùng, mỗi tuyến là list store_id theo đúng thứ tự đi.
    """
    routes = {s: [s] for s in store_ids}
    store_to_route = {s: s for s in store_ids}

    def route_km(route):
        seq = [depot] + route + [depot]
        return sum(dist[seq[k], seq[k + 1]] for k in range(len(seq) - 1))

    def route_hours(route):
        return route_km(route) / avg_speed_kmh + len(route) * service_hours_per_stop

    def route_feasible(route):
        """Kiểm tra CẢ giới hạn thời gian VÀ thể tích — ghép tuyến chỉ hợp lệ khi thỏa cả hai."""
        if route_hours(route) > max_route_hours:
            return False
        if volume_per_store is not None and max_vehicle_volume_liters is not None:
            total_vol = sum(volume_per_store.get(s, 0.0) for s in route)
            if total_vol > max_vehicle_volume_liters:
                return False
        return True

    savings = sorted(
        ((dist[depot, i] + dist[depot, j] - dist[i, j], i, j)
         for idx, i in enumerate(store_ids) for j in store_ids[idx + 1:]),
        reverse=True,
    )

    for sav, i, j in savings:
        if sav <= 0:
            break  # ghép mà không tiết kiệm gì thì dừng, không ghép nữa
        ri_key, rj_key = store_to_route[i], store_to_route[j]
        if ri_key == rj_key:
            continue
        route_i, route_j = routes[ri_key], routes[rj_key]

        i_at_end, i_at_start = route_i[-1] == i, route_i[0] == i
        j_at_end, j_at_start = route_j[-1] == j, route_j[0] == j
        if not (i_at_end or i_at_start) or not (j_at_end or j_at_start):
            continue  # i hoặc j đang nằm giữa tuyến -> không ghép được nữa

        ordered_i = route_i if i_at_end else route_i[::-1]
        ordered_j = route_j if j_at_start else route_j[::-1]
        merged = ordered_i + ordered_j

        if not route_feasible(merged):
            continue  # vượt giới hạn thời gian HOẶC thể tích xe -> bỏ qua, giữ nguyên 2 tuyến riêng

        del routes[ri_key]
        del routes[rj_key]
        new_key = merged[0]
        routes[new_key] = merged
        for s in merged:
            store_to_route[s] = new_key

    return list(routes.values())


def auto_suggest_merged_routes(dist: np.ndarray, store_ids: list, dim_stores: pd.DataFrame,
                                cost_per_km: float = 1.2, fixed_dispatch_cost: float = 25.0,
                                avg_speed_kmh: float = 30.0, service_hours_per_stop: float = 0.5,
                                max_route_hours: float = 4.0,
                                demand_by_store: dict = None, catalog: "pd.DataFrame" = None,
                                dim_vehicles: "pd.DataFrame" = None) -> pd.DataFrame:
    """
    Chạy Clarke-Wright cho TOÀN BỘ store_ids, trả về bảng các tuyến đề xuất — kết hợp
    ĐỒNG THỜI quãng đường ngắn nhất VÀ thể tích tối ưu (nếu truyền demand_by_store +
    catalog + dim_vehicles): ghép tuyến sẽ tự giới hạn theo thể tích xe lớn nhất hiện có,
    và mỗi tuyến sẽ kèm luôn xe đề xuất + % thể tích trống — không cần xem thêm trang khác.
    """
    from product_catalog import compute_volume_needed, compute_volume_by_packaging_type, select_best_fit_vehicle

    store_name_map = dim_stores.set_index("store_id")["store_name"]

    volume_per_store, max_vehicle_volume = None, None
    use_volume = demand_by_store is not None and catalog is not None and dim_vehicles is not None
    if use_volume:
        volume_per_store = {s: compute_volume_needed(demand_by_store.get(s, {}), catalog) for s in store_ids}
        max_vehicle_volume = (dim_vehicles["the_tich_m3"] * 1000).max()

    routes = clarke_wright_merge(dist, store_ids, avg_speed_kmh, service_hours_per_stop, max_route_hours,
                                  volume_per_store=volume_per_store, max_vehicle_volume_liters=max_vehicle_volume)

    rows = []
    for route in routes:
        seq = [0] + route + [0]
        merged_km = sum(dist[seq[k], seq[k + 1]] for k in range(len(seq) - 1))
        merged_cost = merged_km * cost_per_km + fixed_dispatch_cost
        separate_km = sum(2 * dist[0, s] for s in route)
        separate_cost = separate_km * cost_per_km + len(route) * fixed_dispatch_cost
        savings_pct = (separate_cost - merged_cost) / separate_cost * 100 if separate_cost > 0 else 0
        route_hours = merged_km / avg_speed_kmh + len(route) * service_hours_per_stop
        chain = "Kho → " + " → ".join(store_name_map.get(s, str(s)) for s in route) + " → Kho"

        row = {
            "Tuyến đề xuất": chain,
            "Số cửa hàng": len(route),
            "Quãng đường (km)": round(merged_km, 1),
            "Chi phí ghép (VNĐ)": round(merged_cost, 0),
            "Tiết kiệm so với giao riêng (%)": round(savings_pct, 1),
            "Thời gian ước tính (giờ)": round(route_hours, 1),
        }

        if use_volume:
            combined_demand = {}
            for s in route:
                for sku_id, qty in demand_by_store.get(s, {}).items():
                    combined_demand[sku_id] = combined_demand.get(sku_id, 0) + qty
            vol_split = compute_volume_by_packaging_type(combined_demand, catalog)
            needs_cold = vol_split["lanh"] > 0
            veh_result = select_best_fit_vehicle(vol_split["tong"], dim_vehicles, requires_cold=needs_cold)

            row["Thể tích cần chở (lít)"] = round(vol_split["tong"], 1)
            if veh_result["loi"]:
                row["Xe đề xuất"] = "⚠ Không có xe phù hợp"
                row["% Thể tích trống"] = "—"
            else:
                row["Xe đề xuất"] = veh_result["vehicle"]["loai_xe"] + (" 🧊" if needs_cold else "")
                row["% Thể tích trống"] = f"{veh_result['empty_pct']}%" if veh_result["du_the_tich"] else f"⚠ thiếu {abs(veh_result['empty_pct'])}%"

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Tiết kiệm so với giao riêng (%)", ascending=False).reset_index(drop=True)


def suggest_route_merges(dist: np.ndarray, store_ids: list, dim_stores: pd.DataFrame,
                          cost_per_km: float = 1.2, fixed_dispatch_cost: float = 25.0,
                          avg_speed_kmh: float = 30.0, service_hours_per_stop: float = 0.5,
                          max_pairs: int = 20) -> pd.DataFrame:
    """
    Xét từng CẶP cửa hàng trong `store_ids` (thường là các cửa hàng cần giao trong
    cùng chu kỳ) — so sánh chi phí nếu giao riêng 2 chuyến vs gộp thành 1 chuyến qua cả 2.
    Trả về bảng xếp hạng theo % tiết kiệm chi phí, kèm mức độ ưu tiên ghép tuyến.
    """
    store_name_map = dim_stores.set_index("store_id")["store_name"]
    rows = []

    for i, j in itertools.combinations(store_ids, 2):
        separate_dist = 2 * dist[0, i] + 2 * dist[0, j]
        separate_cost = separate_dist * cost_per_km + 2 * fixed_dispatch_cost

        _, merged_dist = solve_optimal_route(dist, [i, j], depot=0)
        merged_cost = merged_dist * cost_per_km + fixed_dispatch_cost

        savings = separate_cost - merged_cost
        savings_pct = savings / max(separate_cost, 1e-6) * 100
        total_hours = merged_dist / avg_speed_kmh + 2 * service_hours_per_stop

        if savings_pct >= 15 and total_hours <= 3:
            priority = "🟢 Nên ghép ngay"
        elif savings_pct >= 5:
            priority = "🟡 Cân nhắc"
        else:
            priority = "🔴 Chưa ưu tiên"

        rows.append({
            "Cặp tuyến": f"{store_name_map.get(i, i)} + {store_name_map.get(j, j)}",
            "Tổng km (ghép)": round(merged_dist, 1),
            "Chi phí/km sau ghép": round(merged_cost / max(merged_dist, 1e-6), 0),
            "Tiết kiệm (%)": round(savings_pct, 1),
            "Tiết kiệm (VNĐ)": round(savings, 0),
            "Thời gian ước tính (giờ)": round(total_hours, 1),
            "Ưu tiên": priority,
        })

    df = pd.DataFrame(rows).sort_values("Tiết kiệm (%)", ascending=False)
    return df.head(max_pairs).reset_index(drop=True)
