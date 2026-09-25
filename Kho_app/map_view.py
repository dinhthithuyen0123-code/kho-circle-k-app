"""
map_view.py
Tạo bản đồ tương tác (Folium) hiển thị kho trung tâm + các cửa hàng theo đúng
tọa độ địa lý thật — có thể zoom, kéo, click vào từng điểm xem chi tiết.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Điểm mốc mặc định để "neo" bản đồ khi dùng dữ liệu MÔ PHỎNG (không có lat/lon thật)
# — chọn khu vực Gò Vấp/Thủ Đức cho nhất quán với dữ liệu thật đã có.
DEFAULT_ANCHOR_LAT, DEFAULT_ANCHOR_LON = 10.815, 106.68
KM_PER_DEGREE_LAT = 111.0


def ensure_latlon(dim_stores: pd.DataFrame, dim_warehouse: pd.DataFrame):
    """
    Đảm bảo dim_stores có cột lat/lon để vẽ bản đồ, dùng được cho CẢ 2 trường hợp:
    - Tọa độ thật (real_store_data.py): đã có sẵn lat/lon -> giữ nguyên.
    - Tọa độ mô phỏng (lưới 0-100, không có lat/lon thật): chuyển đổi bằng cách coi
      mỗi đơn vị lưới ~ 1km, neo quanh 1 điểm mốc cố định -> ra tọa độ HỢP LỆ để vẽ
      lên bản đồ thật, nhưng KHÔNG phải vị trí địa lý có thật (chỉ để minh hoạ).
    Trả về: (dim_stores_with_latlon, wh_lat, wh_lon, is_real)
    """
    if "lat" in dim_stores.columns and "lon" in dim_stores.columns:
        wh_lat, wh_lon = dim_warehouse["y"].iloc[0], dim_warehouse["x"].iloc[0]
        return dim_stores, wh_lat, wh_lon, True

    # ---- Mô phỏng: quy đổi lưới (x, y) 0-100 sang lat/lon giả định quanh 1 điểm mốc ----
    df = dim_stores.copy()
    wh_x, wh_y = dim_warehouse["x"].iloc[0], dim_warehouse["y"].iloc[0]

    df["lat"] = DEFAULT_ANCHOR_LAT + (df["y"] - wh_y) / KM_PER_DEGREE_LAT
    km_per_degree_lon = KM_PER_DEGREE_LAT * np.cos(np.radians(DEFAULT_ANCHOR_LAT))
    df["lon"] = DEFAULT_ANCHOR_LON + (df["x"] - wh_x) / km_per_degree_lon

    return df, DEFAULT_ANCHOR_LAT, DEFAULT_ANCHOR_LON, False


def _nearest_neighbor_route(dist: np.ndarray, points: list, depot: int = 0) -> list:
    if not points:
        return []
    unvisited = set(points)
    route, current = [depot], depot
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[current, j])
        route.append(nxt); unvisited.remove(nxt); current = nxt
    route.append(depot)
    return route


def _route_length(route: list, dist: np.ndarray) -> float:
    return sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1))


def _two_opt(route: list, dist: np.ndarray, max_iter: int = 200) -> list:
    if len(route) <= 3:
        return route
    best, best_len = route[:], _route_length(route, dist)
    improved, it = True, 0
    while improved and it < max_iter:
        improved, it = False, it + 1
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                new_len = _route_length(new_route, dist)
                if new_len < best_len - 1e-9:
                    best, best_len = new_route, new_len
                    improved = True
    return best


def solve_optimal_route(dist: np.ndarray, store_ids_to_visit: list, depot: int = 0):
    """
    Tính tuyến đường ngắn nhất (Nearest-Neighbor + cải tiến 2-opt) đi qua các cửa hàng
    cần giao hàng, xuất phát và quay về kho. `store_ids_to_visit` dùng trực tiếp làm chỉ số
    trong ma trận `dist` (đúng quy ước store_id = chỉ số hàng/cột trong dist, depot = 0).
    Trả về: (route [danh sách store_id theo đúng thứ tự đi], tổng quãng đường).
    """
    route = _nearest_neighbor_route(dist, store_ids_to_visit, depot)
    route = _two_opt(route, dist)
    return route, _route_length(route, dist)


def build_static_map(dim_stores: pd.DataFrame, dim_warehouse: pd.DataFrame, urgent_store_ids: set = None,
                      dist: np.ndarray = None, show_optimal_route: bool = True):
    """
    Bản đồ TĨNH (matplotlib) — không tải bất kỳ nội dung nào từ internet, luôn hiển thị
    được (không bị chặn bởi CSP/tường lửa/ad-blocker). Dùng làm mặc định cho độ ổn định
    khi demo. Trả về (fig, is_real) — is_real cho biết tọa độ có phải thật hay minh hoạ.
    """
    urgent_store_ids = urgent_store_ids or set()
    dim_stores, wh_lat, wh_lon, is_real = ensure_latlon(dim_stores, dim_warehouse)
    store_pos = dim_stores.set_index("store_id")[["lat", "lon"]]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    route, route_km = None, None

    if show_optimal_route and dist is not None and len(urgent_store_ids) > 0:
        route, route_km = solve_optimal_route(dist, list(urgent_store_ids), depot=0)
        route_latlon = [(wh_lat, wh_lon) if sid == 0 else
                         (store_pos.loc[sid, "lat"], store_pos.loc[sid, "lon"]) for sid in route]
        xs = [lon for lat, lon in route_latlon]
        ys = [lat for lat, lon in route_latlon]
        ax.plot(xs, ys, color="#e74c3c", linewidth=2, zorder=2, label=f"Tuyến giao hàng tối ưu ({route_km:.1f} km)")
        # Mũi tên chỉ hướng đi giữa mỗi chặng
        for i in range(len(xs) - 1):
            ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2), zorder=2)
    else:
        for sid in urgent_store_ids:
            if sid in store_pos.index:
                ax.plot([wh_lon, store_pos.loc[sid, "lon"]], [wh_lat, store_pos.loc[sid, "lat"]],
                        color="#e74c3c", linewidth=1, alpha=0.6, zorder=1)

    for _, store in dim_stores.iterrows():
        is_urgent = store["store_id"] in urgent_store_ids
        color = "#e74c3c" if is_urgent else "#3498db"
        ax.scatter(store["lon"], store["lat"], c=color, s=110, zorder=3, edgecolors="white", linewidths=1)
        ax.annotate(store["store_name"], (store["lon"], store["lat"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.scatter(wh_lon, wh_lat, c="black", marker="*", s=350, zorder=4, label="Kho trung tâm")
    ax.scatter([], [], c="#3498db", s=90, label="Cửa hàng bình thường")
    ax.scatter([], [], c="#e74c3c", s=90, label="Cần xử lý gấp (tồn thấp)")
    ax.set_xlabel("Kinh độ" if is_real else "Kinh độ (minh hoạ)")
    ax.set_ylabel("Vĩ độ" if is_real else "Vĩ độ (minh hoạ)")
    title = "Mạng lưới kho trung tâm & cửa hàng"
    if route_km is not None:
        title += f" — Tuyến tối ưu: {route_km:.1f} km"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig, is_real


def render_google_maps_route(api_key: str, dim_stores: pd.DataFrame, dim_warehouse: pd.DataFrame,
                              route_store_ids: list, height: int = 450):
    """
    Nhúng Google Maps Embed API — vẽ tuyến đường thật (theo đường phố thật, không phải
    đường thẳng chim bay) đi qua các cửa hàng theo ĐÚNG thứ tự route_store_ids đã tính
    (từ solve_optimal_route), xuất phát và quay về kho trung tâm.

    YÊU CẦU: api_key phải là Google Maps API key hợp lệ, đã bật "Maps Embed API",
    gắn với tài khoản Google Cloud có thẻ thanh toán (xem ghi chú ở nơi gọi hàm này).
    Dùng trong Streamlit bằng: components.iframe(url, height=...) — xem ví dụ cuối file.
    """
    dim_stores, wh_lat, wh_lon, _ = ensure_latlon(dim_stores, dim_warehouse)
    store_pos = dim_stores.set_index("store_id")[["lat", "lon"]]

    origin = f"{wh_lat},{wh_lon}"
    destination = origin  # quay vòng về kho
    waypoints = "|".join(f"{store_pos.loc[sid,'lat']},{store_pos.loc[sid,'lon']}" for sid in route_store_ids)

    url = (
        "https://www.google.com/maps/embed/v1/directions"
        f"?key={api_key}&origin={origin}&destination={destination}"
        f"&waypoints={waypoints}&mode=driving&avoid=tolls"
    )
    return url


def build_network_map(dim_stores: pd.DataFrame, dim_warehouse: pd.DataFrame,
                       urgent_store_ids: set = None, zoom_start: int = 13):
    """
    Vẽ bản đồ TƯƠNG TÁC (Folium) cho kho trung tâm + các cửa hàng — dùng được với
    cả dữ liệu tọa độ thật lẫn mô phỏng (xem ensure_latlon()).
    urgent_store_ids: tập hợp store_id đang có mặt hàng cần xử lý gấp (tô đỏ), có thể để None
    LƯU Ý: một số mạng/trình duyệt chặn nội dung nhúng từ openstreetmap.org (chính sách CSP
    của họ, không phải lỗi ở đây) — nếu gặp lỗi, dùng build_static_map() thay thế.
    """
    import folium  # import trễ — chỉ cần khi thực sự dùng bản đồ tương tác
    urgent_store_ids = urgent_store_ids or set()
    dim_stores, wh_lat, wh_lon, is_real = ensure_latlon(dim_stores, dim_warehouse)
    m = folium.Map(location=[wh_lat, wh_lon], zoom_start=zoom_start, tiles="CartoDB positron")

    note = "" if is_real else "<br><i>(Vị trí minh hoạ — dữ liệu mô phỏng, không phải toạ độ thật)</i>"
    folium.Marker(
        location=[wh_lat, wh_lon],
        popup=f"<b>Kho trung tâm</b>{note}",
        tooltip="Kho trung tâm",
        icon=folium.Icon(color="black", icon="warehouse", prefix="fa"),
    ).add_to(m)

    for _, store in dim_stores.iterrows():
        is_urgent = store["store_id"] in urgent_store_ids
        color = "red" if is_urgent else "blue"
        status_text = "⚠️ Có mặt hàng cần xử lý gấp" if is_urgent else "✅ Bình thường"
        folium.Marker(
            location=[store["lat"], store["lon"]],
            popup=f"<b>{store['store_name']}</b><br>{status_text}{note}",
            tooltip=store["store_name"],
            icon=folium.Icon(color=color, icon="shopping-cart", prefix="fa"),
        ).add_to(m)

        # Đường nối kho <-> cửa hàng (minh hoạ tuyến vận chuyển, đường thẳng đơn giản)
        folium.PolyLine(
            locations=[[wh_lat, wh_lon], [store["lat"], store["lon"]]],
            color="red" if is_urgent else "gray", weight=1.5, opacity=0.5,
        ).add_to(m)

    return m, is_real
