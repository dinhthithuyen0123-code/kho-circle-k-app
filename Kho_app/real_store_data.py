"""
real_store_data.py
Dữ liệu tọa độ THẬT cho 9 cửa hàng Circle K + 1 kho trung tâm (khu vực Gò Vấp/Thủ Đức, TP.HCM).
Thay thế cho dữ liệu tọa độ mô phỏng (generate_stores trong data_generator_v3.py)
khi bạn muốn dùng vị trí thực tế.

Khoảng cách được tính bằng công thức HAVERSINE (chuẩn cho tọa độ địa lý vĩ độ/kinh độ),
khác với khoảng cách Euclidean phẳng dùng trong dữ liệu mô phỏng trước đây.
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# DỮ LIỆU THẬT — sửa/thêm trực tiếp tại đây nếu có thêm cửa hàng
# ---------------------------------------------------------------------------

WAREHOUSE_LATLON = (10.84703, 106.77392)  # Kho trung tâm — điểm trung tâm (centroid) của 8 cửa hàng khu Thủ Đức,
                                            # dùng TẠM THỜI cho đến khi có toạ độ kho thật

STORE_LIST = [
    {"store_name": "CK001", "address": "45 Thống Nhất, Thủ Đức, TP.HCM",       "lat": 10.84623, "lon": 106.77020},
    {"store_name": "CK002", "address": "66C Hoàng Diệu 2, Thủ Đức, TP.HCM",    "lat": 10.85686, "lon": 106.76424},
    {"store_name": "CK003", "address": "223 Đặng Văn Bi, Thủ Đức, TP.HCM",     "lat": 10.84804, "lon": 106.75974},
    {"store_name": "CK004", "address": "364 Võ Văn Ngân, Thủ Đức, TP.HCM",     "lat": 10.84816, "lon": 106.77162},
    {"store_name": "CK005", "address": "240 Hoàng Diệu 2, Thủ Đức, TP.HCM",    "lat": 10.85394, "lon": 106.77104},
    {"store_name": "CK006", "address": "449 Lê Văn Việt, Thủ Đức, TP.HCM",     "lat": 10.84542, "lon": 106.79375},
    {"store_name": "CK007", "address": "18A15 Tăng Nhơn Phú, Thủ Đức, TP.HCM", "lat": 10.82941, "lon": 106.77350},
    {"store_name": "CK008", "address": "62 Man Thiện, Tăng Nhơn Phú, Thủ Đức, TP.HCM", "lat": 10.84816, "lon": 106.78723},
]


# ---------------------------------------------------------------------------
# Khoảng cách Haversine (km) — chuẩn cho tọa độ địa lý
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # bán kính Trái Đất (km)
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def build_distance_matrix(latlon_list):
    """latlon_list: list[(lat, lon)], index 0 PHẢI là kho trung tâm."""
    n = len(latlon_list)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = haversine_km(latlon_list[i][0], latlon_list[i][1],
                                       latlon_list[j][0], latlon_list[j][1])
    return dist


# ---------------------------------------------------------------------------
# Hàm chính — thay thế trực tiếp cho generate_stores() trong data_generator_v3.py
# ---------------------------------------------------------------------------

def generate_stores_real(n_clusters: int = 3, seed: int = 42):
    """
    Trả về đúng định dạng (dim_stores, coords, dist) như generate_stores() gốc,
    để dùng thay thế trực tiếp — không cần sửa các hàm khác (simulate_operations, v.v.)
    Lưu ý: coords ở đây là (x=lon, y=lat) để tương thích hiển thị bản đồ (scatter plot),
    còn khoảng cách thực tế luôn tính bằng Haversine (dist), KHÔNG dùng Euclidean trên coords.
    """
    n_stores = len(STORE_LIST)
    dim_stores = pd.DataFrame([
        {"store_id": i + 1, "store_name": s["store_name"], "address": s["address"],
         "lat": s["lat"], "lon": s["lon"], "x": s["lon"], "y": s["lat"]}
        for i, s in enumerate(STORE_LIST)
    ])

    # Phân cụm theo vị trí địa lý thật (KMeans trên lat/lon) — dùng cho hiệu ứng tương quan vùng
    km = KMeans(n_clusters=min(n_clusters, n_stores), random_state=seed, n_init=10)
    dim_stores["cluster_id"] = km.fit_predict(dim_stores[["lat", "lon"]])

    # coords: index 0 = kho trung tâm, index 1..n = cửa hàng (đúng quy ước của generate_stores gốc)
    latlon_list = [WAREHOUSE_LATLON] + list(zip(dim_stores["lat"], dim_stores["lon"]))
    coords = np.array([[lon, lat] for lat, lon in latlon_list])  # (x=lon, y=lat) để vẽ bản đồ đúng hướng

    dist = build_distance_matrix(latlon_list)  # đơn vị: KM THẬT (Haversine), không phải đơn vị ảo như trước

    return dim_stores, coords, dist


if __name__ == "__main__":
    dim_stores, coords, dist = generate_stores_real()
    print(dim_stores[["store_id", "store_name", "lat", "lon", "cluster_id"]])
    print("\nMa trận khoảng cách (km) — hàng/cột 0 = kho trung tâm:")
    print(np.round(dist, 2))
