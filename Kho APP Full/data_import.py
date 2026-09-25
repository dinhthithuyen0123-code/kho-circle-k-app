"""
data_import.py
Cho phép người quản lý kho TẢI LÊN dữ liệu bán hàng/tồn kho thật (CSV hoặc Excel)
để thay thế phần dữ liệu mô phỏng của fact_store_daily — dùng ngay cho dự báo AI
và các báo cáo, không cần sửa code.

Định dạng file yêu cầu (xem hàm get_template_csv() để tải file mẫu):
    Ngày, Cửa hàng, Mặt hàng, Số lượng bán, Số lượng nhập

Lưu ý hạn chế (giống mọi hệ thống POS thực tế): file thường KHÔNG có dữ liệu
"nhu cầu bị mất" (khách muốn mua nhưng hết hàng) và không có sẵn thông tin
thời tiết/lễ Tết — các cột này sẽ được điền giá trị mặc định/suy luận từ ngày,
người dùng có thể bổ sung thêm cột nếu muốn chính xác hơn.
"""

import io
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Ngày", "Cửa hàng", "Mặt hàng", "Số lượng bán"]
OPTIONAL_COLUMNS = ["Số lượng nhập", "Khuyến mãi", "Nhiệt độ (C)"]


def get_template_csv() -> bytes:
    """Tạo file CSV mẫu để người dùng biết đúng định dạng cần điền."""
    sample = pd.DataFrame({
        "Ngày": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "Cửa hàng": ["CK001", "CK002", "CK001"],
        "Mặt hàng": ["Đồ uống lạnh #1", "Đồ uống lạnh #1", "Đồ uống lạnh #1"],
        "Số lượng bán": [32, 28, 30],
        "Số lượng nhập": [50, 0, 0],
        "Khuyến mãi": [0, 0, 1],
        "Nhiệt độ (C)": [29, 29, 31],
    })
    return sample.to_csv(index=False).encode("utf-8-sig")


def validate_columns(df: pd.DataFrame) -> list:
    """Trả về danh sách lỗi (rỗng nếu file hợp lệ)."""
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Thiếu cột bắt buộc: {', '.join(missing)}. "
                       f"Các cột bắt buộc: {', '.join(REQUIRED_COLUMNS)}")
    return errors


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Đọc file CSV hoặc Excel do người dùng tải lên (Streamlit UploadedFile)."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Chỉ hỗ trợ file .csv, .xlsx hoặc .xls")
    return df


def build_fact_store_daily_from_upload(
    raw_df: pd.DataFrame, dim_stores: pd.DataFrame, dim_skus: pd.DataFrame,
) -> tuple:
    """
    Chuyển dữ liệu thô người dùng tải lên thành đúng định dạng fact_store_daily
    mà mô hình dự báo (forecasting_v3.py) và các trang báo cáo đang dùng.

    Trả về: (fact_store_daily, warnings) — warnings là danh sách cảnh báo
    (ví dụ tên cửa hàng/mặt hàng không khớp danh mục hiện có) để hiển thị cho người dùng.
    """
    warnings = []
    df = raw_df.copy()
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors="coerce")
    if df["Ngày"].isna().any():
        warnings.append(f"{df['Ngày'].isna().sum()} dòng có định dạng ngày không đọc được — đã loại bỏ.")
        df = df.dropna(subset=["Ngày"])

    # Khớp tên cửa hàng / mặt hàng với danh mục hiện có (không phân biệt hoa thường, khoảng trắng thừa)
    store_map = {s.strip().lower(): sid for s, sid in zip(dim_stores["store_name"], dim_stores["store_id"])}
    sku_map = {s.strip().lower(): sid for s, sid in zip(dim_skus["sku_name"], dim_skus["sku_id"])}

    df["store_id"] = df["Cửa hàng"].astype(str).str.strip().str.lower().map(store_map)
    df["sku_id"] = df["Mặt hàng"].astype(str).str.strip().str.lower().map(sku_map)

    unmatched_stores = df[df["store_id"].isna()]["Cửa hàng"].unique().tolist()
    unmatched_skus = df[df["sku_id"].isna()]["Mặt hàng"].unique().tolist()
    if unmatched_stores:
        warnings.append(f"Không khớp được {len(unmatched_stores)} tên cửa hàng với danh mục hiện có "
                         f"(ví dụ: {unmatched_stores[:3]}) — các dòng này bị loại bỏ. "
                         f"Kiểm tra lại chính tả hoặc thêm cửa hàng mới vào hệ thống trước.")
    if unmatched_skus:
        warnings.append(f"Không khớp được {len(unmatched_skus)} tên mặt hàng với danh mục hiện có "
                         f"(ví dụ: {unmatched_skus[:3]}) — các dòng này bị loại bỏ.")

    df = df.dropna(subset=["store_id", "sku_id"]).copy()
    df["store_id"] = df["store_id"].astype(int)
    df["sku_id"] = df["sku_id"].astype(int)

    if len(df) == 0:
        return pd.DataFrame(), warnings + ["Không còn dòng dữ liệu hợp lệ nào sau khi kiểm tra — vui lòng kiểm tra lại file."]

    df["units_sold"] = pd.to_numeric(df["Số lượng bán"], errors="coerce").fillna(0).clip(lower=0)
    df["units_received"] = pd.to_numeric(
        df["Số lượng nhập"] if "Số lượng nhập" in df.columns else pd.Series(0, index=df.index), errors="coerce"
    ).fillna(0).clip(lower=0)
    df["promo"] = pd.to_numeric(
        df["Khuyến mãi"] if "Khuyến mãi" in df.columns else pd.Series(0, index=df.index), errors="coerce"
    ).fillna(0).astype(int)
    df["temperature_c"] = pd.to_numeric(
        df["Nhiệt độ (C)"] if "Nhiệt độ (C)" in df.columns else pd.Series(np.nan, index=df.index), errors="coerce"
    )
    if df["temperature_c"].isna().any():
        warnings.append("Thiếu dữ liệu nhiệt độ ở một số dòng — đã điền giá trị trung bình 28°C thay thế "
                         "(khuyến nghị bổ sung cột 'Nhiệt độ (C)' để dự báo chính xác hơn với mặt hàng nhạy thời tiết).")
        df["temperature_c"] = df["temperature_c"].fillna(28.0)

    df["day_of_week"] = df["Ngày"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_tet_period"] = 0  # người dùng có thể bổ sung cột riêng nếu muốn khai báo giai đoạn Tết
    df["lost_sales"] = 0.0   # HẠN CHẾ: dữ liệu POS thực tế thường không ghi nhận nhu cầu bị mất do hết hàng
    warnings.append("Lưu ý: dữ liệu tải lên không có thông tin 'nhu cầu bị mất khi hết hàng' — "
                     "mô hình dự báo sẽ coi số bán ra = nhu cầu thực tế, có thể đánh giá thấp nhu cầu "
                     "tại các thời điểm từng bị hết hàng.")

    # Tính tồn kho lũy kế theo (store, sku): tồn cuối = tồn đầu + nhập - bán, cộng dồn theo ngày
    df = df.sort_values(["store_id", "sku_id", "Ngày"]).reset_index(drop=True)
    df["net_change"] = df["units_received"] - df["units_sold"]
    df["cum_change"] = df.groupby(["store_id", "sku_id"])["net_change"].cumsum()

    # File không có tồn kho đầu kỳ thật -> suy ra mức tồn đầu kỳ tối thiểu để tồn kho không bao giờ âm
    min_cum = df.groupby(["store_id", "sku_id"])["cum_change"].transform("min")
    implied_opening = np.where(min_cum < 0, -min_cum, 0)
    if (implied_opening > 0).any():
        warnings.append("Không có dữ liệu tồn kho đầu kỳ trong file — hệ thống đã tự suy ra mức tồn đầu kỳ "
                         "tối thiểu hợp lý (đủ để tồn kho không bị âm) cho từng cửa hàng/mặt hàng. "
                         "Nếu có số tồn kho đầu kỳ thật, nên bổ sung để kết quả chính xác hơn.")
    df["closing_stock"] = implied_opening + df["cum_change"]
    df["opening_stock"] = df["closing_stock"] + df["units_sold"] - df["units_received"]

    min_date = df["Ngày"].min()
    df["day_idx"] = (df["Ngày"] - min_date).dt.days
    df["date"] = df["Ngày"]

    fact_store_daily = df[[
        "date", "day_idx", "store_id", "sku_id", "day_of_week", "is_weekend",
        "temperature_c", "is_tet_period", "opening_stock", "units_sold",
        "lost_sales", "closing_stock", "units_received", "promo",
    ]].copy()

    return fact_store_daily, warnings
