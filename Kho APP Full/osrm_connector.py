"""
osrm_connector.py
Lấy khoảng cách ĐI ĐƯỜNG THẬT (không phải khoảng cách chim bay/Haversine) giữa kho
trung tâm và các cửa hàng, dùng máy chủ demo công khai, MIỄN PHÍ, KHÔNG CẦN API KEY
của OSRM (Open Source Routing Machine): https://router.project-osrm.org

Lưu ý sử dụng (theo đúng chính sách máy chủ demo):
- Chỉ dùng cho mục đích phi thương mại, hợp lý (đồ án/demo là phù hợp).
- Không gọi quá 1 request/giây.
- Không đảm bảo uptime — nếu máy chủ demo bận/lỗi, nên có phương án dự phòng
  (giữ lại khoảng cách Haversine cũ làm phương án thay thế).
"""

import time
import requests
import numpy as np
import pandas as pd

OSRM_BASE_URL = "https://router.project-osrm.org"


def get_real_road_distance_matrix(latlon_list: list, retry: int = 2) -> np.ndarray:
    """
    latlon_list: danh sách (lat, lon), index 0 PHẢI là kho trung tâm (đúng quy ước
    dùng xuyên suốt project — dist[0] = kho, dist[1:] = các cửa hàng).
    Trả về: ma trận khoảng cách (km) đi đường thật, cùng kích thước/thứ tự như
    ma trận Haversine cũ — dùng thay thế trực tiếp cho `dist` ở mọi nơi trong app.
    """
    coords_str = ";".join(f"{lon},{lat}" for lat, lon in latlon_list)  # OSRM dùng lon,lat (ngược với thường lệ)
    url = f"{OSRM_BASE_URL}/table/v1/driving/{coords_str}?annotations=distance"

    for attempt in range(retry + 1):
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "Ok":
                raise ValueError(f"OSRM trả về lỗi: {data.get('code')} - {data.get('message', '')}")
            distances_m = np.array(data["distances"])
            return distances_m / 1000.0  # đổi mét -> km
        except Exception as e:
            if attempt < retry:
                time.sleep(1.5)  # chờ rồi thử lại — tuân thủ giới hạn nhẹ nhàng với server demo
                continue
            raise RuntimeError(f"Không lấy được khoảng cách thật từ OSRM sau {retry+1} lần thử: {e}")


def get_real_distance_matrix_for_stores(dim_stores: pd.DataFrame, warehouse_latlon: tuple) -> np.ndarray:
    """
    Hàm tiện dùng: ghép toạ độ kho + toạ độ từng cửa hàng (theo đúng thứ tự store_id
    1..n) rồi gọi OSRM lấy ma trận khoảng cách thật.
    """
    latlon_list = [warehouse_latlon] + list(zip(dim_stores["lat"], dim_stores["lon"]))
    return get_real_road_distance_matrix(latlon_list)


if __name__ == "__main__":
    from real_store_data import generate_stores_real, WAREHOUSE_LATLON

    dim_stores, coords, dist_haversine = generate_stores_real()
    print("Đang gọi OSRM lấy khoảng cách đường thật (có thể mất vài giây)...")
    dist_real = get_real_distance_matrix_for_stores(dim_stores, WAREHOUSE_LATLON)

    labels = ["Kho"] + dim_stores["store_name"].tolist()
    print("\n=== So sánh: Haversine (chim bay) vs OSRM (đường thật) — hàng đầu tiên (từ Kho) ===")
    compare = pd.DataFrame({
        "Điểm đến": labels,
        "Chim bay (km)": np.round(dist_haversine[0], 2),
        "Đường thật (km)": np.round(dist_real[0], 2),
    })
    compare["Chênh lệch (%)"] = ((compare["Đường thật (km)"] - compare["Chim bay (km)"])
                                  / compare["Chim bay (km)"].replace(0, np.nan) * 100).round(1)
    print(compare.to_string(index=False))
