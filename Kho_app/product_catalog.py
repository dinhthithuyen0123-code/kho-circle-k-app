"""
product_catalog.py
Danh mục thông tin đóng gói của từng mặt hàng: kích thước thùng (dài x rộng x cao),
thể tích, loại đóng gói (Thường / Lạnh — hàng lạnh dùng thùng cách nhiệt nên thể tích
mỗi đơn vị lớn hơn hàng thường dù cùng kích cỡ sản phẩm).

Dùng để tính TOÁN THỂ TÍCH THẬT khi xếp hàng lên xe (thay cho ước tính trọng lượng
500g/đơn vị trước đây) — mục tiêu: chọn xe sao cho % thể tích trống thấp nhất,
không chỉ "đủ tải trọng".

GIẢ ĐỊNH: kích thước đóng gói theo nhóm hàng, ước tính hợp lý cho ngành FMCG/cửa hàng
tiện lợi — nên thay bằng số liệu thật của từng SKU cụ thể nếu doanh nghiệp có sẵn.
"""

import pandas as pd

# (dài cm, rộng cm, cao cm, số đơn vị sản phẩm mỗi thùng, loại đóng gói)
PACKAGING_SPECS = {
    "Đồ uống lạnh":        dict(dai=40, rong=27, cao=25, don_vi_moi_thung=24, loai="Lạnh"),
    "Thực phẩm chế biến":  dict(dai=35, rong=25, cao=20, don_vi_moi_thung=12, loai="Lạnh"),
    "Bánh kẹo":            dict(dai=45, rong=30, cao=25, don_vi_moi_thung=30, loai="Thường"),
    "Mì ly/ăn liền":       dict(dai=40, rong=30, cao=28, don_vi_moi_thung=24, loai="Thường"),
    "Sữa/Sản phẩm lạnh":   dict(dai=35, rong=25, cao=22, don_vi_moi_thung=20, loai="Lạnh"),
    "Đồ gia dụng":         dict(dai=50, rong=40, cao=35, don_vi_moi_thung=6,  loai="Thường"),
    "Hàng giá trị cao":    dict(dai=30, rong=20, cao=15, don_vi_moi_thung=10, loai="Thường (dễ vỡ)"),
}


def build_product_catalog(dim_skus: pd.DataFrame) -> pd.DataFrame:
    """Trả về danh mục đóng gói đầy đủ cho từng SKU: kích thước thùng, thể tích thùng,
    thể tích mỗi ĐƠN VỊ sản phẩm (dùng để nhân với số lượng dự báo -> tổng thể tích cần chở)."""
    rows = []
    for _, sku in dim_skus.iterrows():
        spec = PACKAGING_SPECS.get(sku["category"], PACKAGING_SPECS["Bánh kẹo"])
        the_tich_thung_cm3 = spec["dai"] * spec["rong"] * spec["cao"]
        the_tich_don_vi_cm3 = the_tich_thung_cm3 / spec["don_vi_moi_thung"]
        rows.append({
            "sku_id": sku["sku_id"], "sku_name": sku["sku_name"], "category": sku["category"],
            "Loại đóng gói": spec["loai"],
            "Kích thước thùng (D x R x C, cm)": f"{spec['dai']} x {spec['rong']} x {spec['cao']}",
            "Thể tích thùng (lít)": round(the_tich_thung_cm3 / 1000, 2),
            "Số đơn vị/thùng": spec["don_vi_moi_thung"],
            "the_tich_don_vi_lit": round(the_tich_don_vi_cm3 / 1000, 4),  # dùng để tính toán (không hiển thị thô)
        })
    return pd.DataFrame(rows)


def compute_volume_needed(demand_by_sku: dict, catalog: pd.DataFrame) -> float:
    """demand_by_sku: {sku_id: số lượng dự báo}. Trả về tổng thể tích cần chở (lít)."""
    vol_map = catalog.set_index("sku_id")["the_tich_don_vi_lit"]
    return sum(qty * vol_map.get(sku_id, 0.0) for sku_id, qty in demand_by_sku.items())


def select_best_fit_vehicle(volume_needed_liters: float, dim_vehicles: pd.DataFrame,
                             requires_cold: bool = False) -> dict:
    """
    Chọn xe sao cho % THỂ TÍCH TRỐNG thấp nhất (best-fit) — không chỉ "đủ chở" mà còn
    hạn chế lãng phí thể tích thừa. Nếu không xe nào đủ chứa, trả về xe lớn nhất kèm cảnh báo.

    requires_cold=True: BẮT BUỘC chỉ chọn trong các xe có ngăn lạnh (co_ngan_lanh=True) —
    dùng khi lô hàng có chứa mặt hàng cần bảo quản lạnh, tránh xếp nhầm lên xe không có
    ngăn lạnh làm hỏng hàng.
    """
    vehicles = dim_vehicles.copy()
    vehicles["suc_chua_lit"] = vehicles["the_tich_m3"] * 1000

    if requires_cold:
        vehicles = vehicles[vehicles["co_ngan_lanh"] == True]  # noqa: E712
        if len(vehicles) == 0:
            return {"vehicle": None, "empty_pct": None, "du_the_tich": False,
                    "loi": "⚠ Không có xe nào có ngăn lạnh trong đội xe hiện tại — không thể giao hàng lạnh."}

    suitable = vehicles[vehicles["suc_chua_lit"] >= volume_needed_liters].copy()
    if len(suitable) > 0:
        # Trong các xe đủ chỗ, chọn xe có sức chứa GẦN NHẤT với nhu cầu (ít trống nhất)
        suitable["do_lech"] = suitable["suc_chua_lit"] - volume_needed_liters
        best = suitable.sort_values("do_lech").iloc[0]
        empty_pct = (best["suc_chua_lit"] - volume_needed_liters) / best["suc_chua_lit"] * 100
        return {"vehicle": best, "empty_pct": round(empty_pct, 1), "du_the_tich": True, "loi": None}
    else:
        best = vehicles.sort_values("suc_chua_lit", ascending=False).iloc[0]
        thieu_pct = (volume_needed_liters - best["suc_chua_lit"]) / best["suc_chua_lit"] * 100
        return {"vehicle": best, "empty_pct": -round(thieu_pct, 1), "du_the_tich": False, "loi": None}


def compute_volume_by_packaging_type(demand_by_sku: dict, catalog: pd.DataFrame) -> dict:
    """Tách tổng thể tích cần chở thành 2 phần: hàng LẠNH và hàng THƯỜNG — để biết
    lô hàng có cần xe có ngăn lạnh hay không, và cần bao nhiêu thể tích mỗi loại."""
    cat = catalog.set_index("sku_id")
    vol_lanh, vol_thuong = 0.0, 0.0
    for sku_id, qty in demand_by_sku.items():
        if sku_id not in cat.index:
            continue
        v = qty * cat.loc[sku_id, "the_tich_don_vi_lit"]
        if cat.loc[sku_id, "Loại đóng gói"] == "Lạnh":
            vol_lanh += v
        else:
            vol_thuong += v
    return {"lanh": vol_lanh, "thuong": vol_thuong, "tong": vol_lanh + vol_thuong}
