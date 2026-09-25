"""
kiotviet_connector.py
Kết nối THẬT với KiotViet Public API (https://public.kiotapi.com) để lấy dữ liệu
hàng hóa + tồn kho, chuyển về đúng định dạng mà Tầng 1 (layer1_ingestion.py) cần.

YÊU CẦU: cần có tài khoản KiotViet (dùng thử miễn phí được) + ClientId/Client Secret
lấy từ "Thiết lập cửa hàng" -> "Thiết lập kết nối API".

Cách dùng:
    connector = KiotVietConnector(retailer="ten_gian_hang", client_id="...", client_secret="...")
    connector.authenticate()
    df = connector.get_products_as_sales_format()
"""

import requests
import pandas as pd
from datetime import datetime

TOKEN_URL = "https://id.kiotviet.vn/connect/token"
API_BASE = "https://public.kiotapi.com"


class KiotVietConnector:
    def __init__(self, retailer: str, client_id: str, client_secret: str):
        self.retailer = retailer
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None

    def authenticate(self):
        """Lấy Access Token theo cơ chế OAuth2 client_credentials."""
        resp = requests.post(TOKEN_URL, data={
            "scopes": "PublicApi.Access",
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }, headers={"Content-Type": "application/x-www-form-urlencoded"})
        resp.raise_for_status()
        self.access_token = resp.json()["access_token"]
        return self.access_token

    def _headers(self):
        if not self.access_token:
            self.authenticate()
        return {"Retailer": self.retailer, "Authorization": f"Bearer {self.access_token}"}

    def get_products_with_inventory(self, page_size: int = 100) -> pd.DataFrame:
        """Lấy toàn bộ hàng hóa kèm tồn kho thật từ KiotViet."""
        all_items = []
        current_item = 0
        while True:
            resp = requests.get(f"{API_BASE}/products", headers=self._headers(), params={
                "pageSize": page_size, "currentItem": current_item, "includeInventory": True,
            })
            resp.raise_for_status()
            data = resp.json()
            all_items.extend(data.get("data", []))
            if len(data.get("data", [])) < page_size:
                break
            current_item += page_size
        return pd.DataFrame(all_items)

    def get_products_as_sales_format(self) -> pd.DataFrame:
        """
        Chuyển dữ liệu KiotViet về đúng định dạng file mẫu của Tầng 1
        (Ngày, Cửa hàng, Mặt hàng, Số lượng bán, Số lượng nhập).
        LƯU Ý: bản này mới lấy được TỒN KHO hiện tại (onHand) từ /products —
        để lấy đúng SỐ LƯỢNG BÁN mỗi ngày cần gọi thêm API /invoices (hóa đơn),
        endpoint này cần thêm thời gian kiểm tra cấu trúc response thực tế trên
        tài khoản của bạn trước khi viết phần chuyển đổi tương ứng.
        """
        products_df = self.get_products_with_inventory()
        rows = []
        today = datetime.now().strftime("%Y-%m-%d")
        for _, row in products_df.iterrows():
            for inv in row.get("inventories", []):
                rows.append({
                    "Ngày": today,
                    "Cửa hàng": inv.get("branchName", "N/A"),
                    "Mặt hàng": row.get("fullName", row.get("name")),
                    "Số lượng bán": 0,  # placeholder — cần API /invoices để có số liệu thật
                    "Số lượng nhập": 0,  # placeholder — cần API /purchaseorders để có số liệu thật
                    "Tồn kho hiện tại (từ KiotViet)": inv.get("onHand", 0),
                })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Ví dụ sử dụng — thay bằng thông tin tài khoản thật của bạn
    connector = KiotVietConnector(
        retailer="ten_gian_hang_cua_ban",
        client_id="dan_client_id_vao_day",
        client_secret="dan_client_secret_vao_day",
    )
    df = connector.get_products_as_sales_format()
    print(df.head(20))
    df.to_csv("du_lieu_tu_kiotviet.csv", index=False, encoding="utf-8-sig")
