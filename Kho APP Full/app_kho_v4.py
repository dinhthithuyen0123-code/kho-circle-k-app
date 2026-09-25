"""
app_kho_v4.py
Dashboard quản lý kho trung tâm — phiên bản đầy đủ (v4):
  🏠 Tổng quan | 📈 Dự báo nhu cầu | ⏰ Cảnh báo cận date
  🏭 Kho trung tâm & Kệ | 🚚 Đội xe & Tài xế | 🏬 Chi tiết cửa hàng

Chạy: streamlit run app_kho_v4.py
YÊU CẦU CÙNG THƯ MỤC: data_generator_v3.py, data_generator_v4.py, forecasting_v3.py
"""

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_generator_v4 import build_full_dataset
from data_import import get_template_csv, validate_columns, read_uploaded_file, build_fact_store_daily_from_upload
from layer1_ingestion import receive_daily_pos_data, read_raw_log
from layer2_weekly_batch import run_weekly_batch
from layer3_forecast_warehouse import run_forecast_and_plan_warehouse, load_full_history
import streamlit.components.v1 as components
from map_view import build_static_map, build_network_map, ensure_latlon, solve_optimal_route, render_google_maps_route
from route_merging import compute_daily_route_comparison, suggest_route_merges, auto_suggest_merged_routes
from distribution_planning import plan_future_distribution, build_weekly_driver_schedule
from product_catalog import build_product_catalog
import driver_db as ddb
from forecasting_v3 import (
    build_features_v3, train_quantile_models_v3, predict_quantiles_v3,
    get_feature_importance, explain_row, FEATURE_LABELS_VI,
)

st.set_page_config(page_title="Kho Trung Tâm — Dashboard", page_icon="📦", layout="wide")

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("📦 Kho Trung Tâm")
st.sidebar.caption("Bảng điều khiển vận hành")

# ---------------------------------------------------------------------------
# Cấu hình cố định (không hiện trên giao diện) — dùng dữ liệu Circle K thật
# làm mặc định. Đổi trực tiếp các giá trị dưới đây trong code nếu cần điều
# chỉnh, không cần thanh trượt cho người dùng cuối.
# ---------------------------------------------------------------------------
use_real_stores = True
n_stores, n_skus, n_clusters = 9, 12, 3
n_days, n_vehicles, n_drivers = 200, 3, 3
replenish_cycle, seed = 7, 42
DEFAULT_DRIVER_NAMES = ["Nguyễn Văn An", "Trần Văn Bình", "Lê Văn Cường"]  # dùng chung cho TOÀN BỘ app

expiry_threshold = st.sidebar.slider("Ngưỡng cảnh báo cận date (ngày)", 1, 14, 5)

with st.sidebar.expander("🛣️ Khoảng cách: chim bay hay đường thật?", expanded=False):
    st.caption("Mặc định dùng khoảng cách chim bay (Haversine). Có thể lấy khoảng cách "
               "đi đường thật (OSRM, miễn phí) — cần internet, chỉ dùng được với tọa độ thật.")
    if st.button("Lấy khoảng cách đường thật (OSRM)"):
        if not use_real_stores:
            st.error("Cần bật 'Dùng tọa độ cửa hàng THẬT' trước.")
        else:
            try:
                from osrm_connector import get_real_distance_matrix_for_stores
                from real_store_data import WAREHOUSE_LATLON
                with st.spinner("Đang gọi OSRM lấy khoảng cách đường thật..."):
                    real_dist = get_real_distance_matrix_for_stores(
                        st.session_state["v4_data"]["dim_stores"], WAREHOUSE_LATLON
                    )
                st.session_state["v4_data"]["dist"] = real_dist
                st.session_state["using_real_road_dist"] = True
                st.success("Đã cập nhật — toàn bộ app giờ dùng khoảng cách đường thật.")
            except Exception as e:
                st.error(f"Không lấy được từ OSRM ({e}) — vẫn đang dùng khoảng cách chim bay.")
    if st.session_state.get("using_real_road_dist"):
        st.caption("✅ Đang dùng khoảng cách đường thật (OSRM)")
    else:
        st.caption("Đang dùng khoảng cách chim bay (mặc định)")

run_btn = st.sidebar.button("🔄 Tải / Làm mới dữ liệu", type="primary", use_container_width=True)

st.sidebar.markdown("---")
with st.sidebar.expander("📤 Tải dữ liệu bán hàng THẬT (thay dữ liệu mô phỏng)", expanded=False):
    st.caption("Dữ liệu kho trung tâm & đội xe vẫn dùng mô phỏng. Chỉ phần bán hàng/tồn kho cửa hàng được thay thế.")
    st.download_button("📄 Tải file mẫu (CSV)", get_template_csv(), "mau_du_lieu_ban_hang.csv", "text/csv")
    uploaded_file = st.file_uploader("Chọn file CSV hoặc Excel", type=["csv", "xlsx", "xls"])
    apply_upload_btn = st.button("✅ Áp dụng dữ liệu này", use_container_width=True, disabled=uploaded_file is None)

page = st.sidebar.radio("Điều hướng", [
    "🏠 Tổng quan", "📈 Dự báo nhu cầu", "⏰ Cảnh báo cận date",
    "🏭 Kho trung tâm & Kệ", "🚚 Đội xe & Tài xế", "🏬 Chi tiết cửa hàng",
    "🔄 Quy trình 3 tầng", "🔗 Ghép tuyến", "📦 Kế hoạch phân phối", "📐 Thông tin sản phẩm",
])
st.sidebar.caption(f"Cập nhật đến: {pd.Timestamp.now().strftime('%Y-%m-%d')}")


# =============================================================================
# TẢI / SINH DỮ LIỆU (cache)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_all(n_stores, n_skus, n_clusters, n_days, n_vehicles, n_drivers, replenish_cycle, seed, use_real_stores):
    return build_full_dataset(
        n_stores=n_stores, n_skus=n_skus, n_clusters=n_clusters, n_days=n_days,
        n_vehicles=n_vehicles, n_drivers=n_drivers, replenish_cycle=replenish_cycle, seed=seed,
        use_real_stores=use_real_stores, driver_names=DEFAULT_DRIVER_NAMES,
    )


@st.cache_data(show_spinner=False)
def run_forecast_v3(fact_store_daily, test_days=30):
    feat_df = build_features_v3(fact_store_daily)
    split_day = feat_df["day_idx"].max() - test_days
    train_df = feat_df[feat_df["day_idx"] < split_day]
    test_df = feat_df[feat_df["day_idx"] >= split_day]
    models = train_quantile_models_v3(train_df)
    pred_df = predict_quantiles_v3(models, test_df)
    importance_df = get_feature_importance(models)
    return pred_df, importance_df


if run_btn or "v4_data" not in st.session_state or "v4_forecast" not in st.session_state:
    with st.spinner("Đang sinh dữ liệu và mô phỏng vận hành (kho, kệ, đội xe)..."):
        st.session_state["v4_data"] = load_all(
            n_stores, n_skus, n_clusters, n_days, n_vehicles, n_drivers, replenish_cycle, seed, use_real_stores,
        )
    with st.spinner("Đang huấn luyện mô hình dự báo AI..."):
        st.session_state["v4_forecast"] = run_forecast_v3(st.session_state["v4_data"]["fact_store_daily"])
    st.toast("Đã cập nhật dữ liệu!", icon="✅")

data = st.session_state["v4_data"]
pred_df, importance_df = st.session_state["v4_forecast"]

# ---------------- Xử lý khi người dùng tải file dữ liệu thật lên ----------------
if apply_upload_btn and uploaded_file is not None:
    try:
        raw_df = read_uploaded_file(uploaded_file)
        col_errors = validate_columns(raw_df)
        if col_errors:
            for e in col_errors:
                st.sidebar.error(e)
        else:
            new_fact_store_daily, upload_warnings = build_fact_store_daily_from_upload(
                raw_df, data["dim_stores"], data["dim_skus"],
            )
            if len(new_fact_store_daily) == 0:
                st.sidebar.error("Không có dòng dữ liệu hợp lệ nào trong file.")
            else:
                data["fact_store_daily"] = new_fact_store_daily
                st.session_state["v4_data"] = data
                with st.spinner("Đang huấn luyện lại mô hình dự báo trên dữ liệu thật..."):
                    st.session_state["v4_forecast"] = run_forecast_v3(new_fact_store_daily, test_days=min(30, new_fact_store_daily["day_idx"].max() // 3 or 1))
                pred_df, importance_df = st.session_state["v4_forecast"]
                st.sidebar.success(f"Đã áp dụng {len(new_fact_store_daily)} dòng dữ liệu thật!")
                for w in upload_warnings:
                    st.sidebar.warning(w)
    except Exception as e:
        st.sidebar.error(f"Lỗi khi đọc file: {e}")

dim_stores, dim_skus = data["dim_stores"], data["dim_skus"]
dim_warehouse, dim_shelves = data["dim_warehouse"], data["dim_shelves"]
dim_vehicles, dim_drivers = data["dim_vehicles"], data["dim_drivers"]
fact_store_daily = data["fact_store_daily"]
fact_fleet_daily = data["fact_fleet_daily"]
fact_warehouse_inbound = data["fact_warehouse_inbound"]
fact_warehouse_outbound = data["fact_warehouse_outbound"]
fact_shelf_stock_daily = data["fact_shelf_stock_daily"]
fact_warehouse_daily = data["fact_warehouse_daily"]
batch_snapshot = data["batch_snapshot"]
coords = data["coords"]
dist_matrix = data["dist"]
N_DAYS = int(fact_store_daily["day_idx"].max()) + 1


# =============================================================================
# TRANG: TỔNG QUAN
# =============================================================================
if page == "🏠 Tổng quan":
    st.title("Kho Circle K — Tổng quan vận hành")

    last_7 = fact_store_daily[fact_store_daily["day_idx"] >= N_DAYS - 7]
    total_sold_7d = int(last_7["units_sold"].sum())
    lost_by_sku = last_7.groupby("sku_id")["lost_sales"].sum()
    n_sku_thieu = int((lost_by_sku > 0).sum())
    latest_stock = fact_store_daily[fact_store_daily["day_idx"] == N_DAYS - 1]
    avg_daily = fact_store_daily[fact_store_daily["day_idx"] >= N_DAYS - 30].groupby(["store_id", "sku_id"])["units_sold"].mean()
    merged = latest_stock.set_index(["store_id", "sku_id"])["closing_stock"]
    excess = (merged / (avg_daily * 7 + 1e-6)) > 2.0
    n_du_thua = int(excess.sum())
    n_can_date = int((batch_snapshot["days_to_expiry"] <= expiry_threshold).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sản lượng bán ra (7 ngày)", f"{total_sold_7d:,}", f"{n_stores} cửa hàng × {n_skus} SKU")
    c2.metric("SKU đang thiếu hàng", n_sku_thieu, "so với dự báo nhu cầu", delta_color="inverse")
    c3.metric("SKU đang thừa hàng", n_du_thua, "ứ đọng, rủi ro cận date", delta_color="inverse")
    c4.metric("Lô hàng cận date", n_can_date, f"cần xử lý trong {expiry_threshold} ngày", delta_color="inverse")

    c5, c6 = st.columns(2)
    today_fleet = fact_fleet_daily[fact_fleet_daily["day_idx"] == N_DAYS - 1]
    n_dang_giao = (today_fleet["trang_thai"] == "Đang giao hàng").sum()
    n_ranh = today_fleet["trang_thai"].str.contains("Rảnh", na=False).sum()
    c5.metric("Tài xế đang giao hàng (hôm nay)", int(n_dang_giao))
    c6.metric("Tài xế đang rảnh (hôm nay)", int(n_ranh))

    st.markdown("#### Nhu cầu bán ra toàn hệ thống (thực tế + dự báo)")
    daily_total = fact_store_daily.groupby("day_idx")["units_sold"].sum().reset_index()
    daily_total = daily_total[daily_total["day_idx"] >= N_DAYS - 30]
    fc_total = pred_df.groupby("day_idx")["q50"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily_total["day_idx"], daily_total["units_sold"], color="#2c3e50", linewidth=1.8, label="Thực tế")
    ax.plot(fc_total["day_idx"], fc_total["q50"], "--", color="#e67e22", linewidth=1.8, label="Dự báo (AI)")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Sản lượng bán")
    ax.legend()
    st.pyplot(fig)

    st.markdown("#### ⚠️ Cần xử lý gấp — Tồn kho thấp so với nhu cầu")
    urgent = last_7.groupby(["store_id", "sku_id"]).agg(
        ton_hien_tai=("closing_stock", "last"),
        nhu_cau_tb_ngay=("units_sold", "mean"),
        thieu_hut=("lost_sales", "sum"),
    ).reset_index()
    # Số ngày tồn kho hiện tại còn đủ dùng, dựa trên nhu cầu bán trung bình gần đây
    # (dùng cách này thay vì chỉ dựa vào "thiếu hụt" vì dữ liệu POS thật thường không
    # ghi nhận được nhu cầu bị mất khi hết hàng — thieu_hut khi đó luôn bằng 0)
    urgent["so_ngay_con_du"] = urgent["ton_hien_tai"] / urgent["nhu_cau_tb_ngay"].replace(0, np.nan)
    urgent = urgent[urgent["so_ngay_con_du"] <= 2].merge(dim_stores[["store_id", "store_name"]], on="store_id") \
        .merge(dim_skus[["sku_id", "sku_name"]], on="sku_id") \
        .sort_values("so_ngay_con_du").head(8)

    if len(urgent) == 0:
        st.success("Không có mặt hàng nào tồn kho thấp đáng lo ngại.")
    else:
        display_urgent = urgent[["store_name", "sku_name", "ton_hien_tai", "nhu_cau_tb_ngay", "so_ngay_con_du", "thieu_hut"]].copy()
        display_urgent.columns = ["Cửa hàng", "Mặt hàng", "Tồn hiện tại", "Nhu cầu TB/ngày", "Số ngày còn đủ dùng", "Thiếu hụt đã ghi nhận"]
        st.dataframe(display_urgent.round(1), use_container_width=True, hide_index=True)
        st.caption("'Thiếu hụt đã ghi nhận' chỉ có giá trị khi dùng dữ liệu mô phỏng hoặc hệ thống POS "
                   "có ghi nhận nhu cầu bị mất — với dữ liệu thật thường sẽ luôn là 0, đây là hạn chế "
                   "cố hữu của dữ liệu POS, không phải lỗi hệ thống.")


# =============================================================================
# TRANG: DỰ BÁO NHU CẦU
# =============================================================================
elif page == "📈 Dự báo nhu cầu":
    st.title("Dự báo nhu cầu AI theo cửa hàng × SKU")
    st.caption("Mô hình học từ: ngày trong tuần, mùa vụ, thời tiết, lễ Tết")

    col1, col2 = st.columns(2)
    sel_store = col1.selectbox("Cửa hàng", sorted(dim_stores["store_name"]))
    sel_sku = col2.selectbox("Mặt hàng", sorted(dim_skus["sku_name"]))
    store_id = dim_stores[dim_stores["store_name"] == sel_store]["store_id"].iloc[0]
    sku_id = dim_skus[dim_skus["sku_name"] == sel_sku]["sku_id"].iloc[0]
    sel_category = dim_skus[dim_skus["sku_name"] == sel_sku]["category"].iloc[0]

    sp = pred_df[(pred_df["store_id"] == store_id) & (pred_df["sku_id"] == sku_id)].sort_values("day_idx").reset_index(drop=True)
    sp["date"] = pd.to_datetime(sp["date"])

    if len(sp) == 0:
        st.warning("Không đủ dữ liệu để dự báo cho lựa chọn này.")
    else:
        fig, ax = plt.subplots(figsize=(11, 4.3))
        ax.fill_between(sp["date"], sp["q10"], sp["q90"], alpha=0.25, label="Khoảng tin cậy 80%")
        ax.plot(sp["date"], sp["q50"], linewidth=1.8, label="Dự báo trung vị")
        ax.plot(sp["date"], sp["true_demand"], "--", alpha=0.7, label="Nhu cầu thực tế")
        ax.set_title(f"{sel_store} — {sel_sku}")
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)

        sp["tuan"] = (sp.index // 7) + 1
        weekly_list = []
        for tuan, g in sp.groupby("tuan"):
            peak_idx = g["q50"].idxmax()
            weekly_list.append({
                "tuan": tuan, "ngay_bat_dau": g["date"].iloc[0], "ngay_ket_thuc": g["date"].iloc[-1],
                "tong_du_bao": g["q50"].sum(), "toi_da_an_toan": g["q90"].sum(),
                "ngay_cao_diem": g.loc[peak_idx, "date"], "luong_cao_diem": g.loc[peak_idx, "q50"],
                "peak_row": g.loc[peak_idx],
            })
        weekly = pd.DataFrame(weekly_list)

        # ---------------- Tồn kho hiện tại & lượng cần nhập kho tuần tới ----------------
        current_stock_rows = fact_store_daily[
            (fact_store_daily["store_id"] == store_id) & (fact_store_daily["sku_id"] == sku_id)
        ].sort_values("day_idx")
        current_stock = current_stock_rows["closing_stock"].iloc[-1] if len(current_stock_rows) > 0 else 0

        tuan_toi = weekly.iloc[0]
        can_nhap = max(0.0, tuan_toi["tong_du_bao"] - current_stock)

        st.markdown("##### 📦 Lượng cần nhập kho tuần tới")
        c1, c2, c3 = st.columns(3)
        c1.metric("Dự báo nhu cầu tuần tới", f"{tuan_toi['tong_du_bao']:.0f} sản phẩm")
        c2.metric("Tồn kho hiện tại", f"{current_stock:.0f} sản phẩm")
        c3.metric("➜ Cần nhập thêm", f"{can_nhap:.0f} sản phẩm")
        st.caption("Cần nhập thêm = Dự báo nhu cầu tuần tới − Tồn kho hiện tại (làm tròn về 0 nếu tồn kho đã đủ).")

        st.markdown("##### 📋 Nhu cầu dự kiến theo tuần")
        for _, row in weekly.iterrows():
            st.markdown(
                f"**Tuần {int(row['tuan'])}** "
                f"({row['ngay_bat_dau'].strftime('%d/%m')} → {row['ngay_ket_thuc'].strftime('%d/%m')}): "
                f"cần chuẩn bị khoảng **{row['tong_du_bao']:.0f} sản phẩm** "
                f"(mức an toàn tối đa: {row['toi_da_an_toan']:.0f}). "
                f"Ngày cao điểm dự kiến: {row['ngay_cao_diem'].strftime('%d/%m')} "
                f"(~{row['luong_cao_diem']:.0f} sản phẩm)."
            )
            with st.expander(f"🔍 Vì sao tuần {int(row['tuan'])} dự báo như vậy?"):
                for reason in explain_row(row["peak_row"], category=sel_category):
                    st.markdown(f"- {reason}")

        # ---------------- Dự báo chi tiết theo TỪNG NGÀY ----------------
        st.markdown("##### 📅 Dự báo chi tiết theo từng ngày")
        DOW_VI = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]

        daily_display = sp.copy()
        daily_display["Thứ"] = daily_display["date"].dt.dayofweek.map(lambda d: DOW_VI[d])
        daily_display["Ngày"] = daily_display["date"].dt.strftime("%d/%m/%Y")
        daily_display["Dự báo thấp (q10)"] = daily_display["q10"].round(0)
        daily_display["Dự báo trung vị (q50)"] = daily_display["q50"].round(0)
        daily_display["Dự báo cao (q90)"] = daily_display["q90"].round(0)
        daily_display["Nhu cầu thực tế"] = daily_display["true_demand"].round(0)
        daily_display["Chênh lệch (thực tế − dự báo)"] = (
            daily_display["true_demand"] - daily_display["q50"]
        ).round(0)

        table_cols = ["Ngày", "Thứ", "Dự báo thấp (q10)", "Dự báo trung vị (q50)",
                      "Dự báo cao (q90)", "Nhu cầu thực tế", "Chênh lệch (thực tế − dự báo)"]

        def highlight_weekend(row):
            is_we = row["Thứ"] in ("Thứ 7", "Chủ Nhật")
            return ["background-color: #eef6ff" if is_we else ""] * len(row)

        st.dataframe(
            daily_display[table_cols].style.apply(highlight_weekend, axis=1),
            use_container_width=True, hide_index=True,
        )
        st.caption("Mỗi dòng là dự báo cho **một ngày cụ thể**. q10/q90 là khoảng dao động (tin cậy 80%); "
                   "'Nhu cầu thực tế' chỉ có với những ngày đã qua (dữ liệu lịch sử dùng để kiểm tra mô hình).")

        csv_daily = daily_display[table_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Tải dự báo theo ngày (CSV)", csv_daily,
                            f"du_bao_ngay_{sel_store}_{sel_sku}.csv", "text/csv")

        with st.expander("🔍 Xem giải thích dự báo cho một ngày cụ thể"):
            chosen_date = st.selectbox(
                "Chọn ngày muốn xem giải thích",
                options=list(sp["date"]),
                format_func=lambda d: d.strftime("%d/%m/%Y (%A)"),
                key="daily_explain_date",
            )
            chosen_row = sp[sp["date"] == chosen_date].iloc[0]
            st.markdown(
                f"**{sel_store} — {sel_sku}**, ngày **{chosen_date.strftime('%d/%m/%Y')}**: "
                f"dự báo trung vị **{chosen_row['q50']:.0f} sản phẩm** "
                f"(khoảng {chosen_row['q10']:.0f} – {chosen_row['q90']:.0f})."
            )
            for reason in explain_row(chosen_row, category=sel_category):
                st.markdown(f"- {reason}")

        st.markdown("##### 💡 Nhận xét")
        if len(weekly) >= 2:
            last_w, prev_w = weekly.iloc[-1], weekly.iloc[-2]
            pct = (last_w["tong_du_bao"] - prev_w["tong_du_bao"]) / max(1e-6, prev_w["tong_du_bao"]) * 100
            if pct > 10:
                st.info(f"📈 Nhu cầu tuần {int(last_w['tuan'])} tăng {pct:.0f}% so với tuần trước — nên tăng lượng nhập/giao hàng.")
            elif pct < -10:
                st.warning(f"📉 Nhu cầu tuần {int(last_w['tuan'])} giảm {abs(pct):.0f}% so với tuần trước — cân nhắc giảm lượng nhập.")
            else:
                st.success(f"➡️ Nhu cầu tuần {int(last_w['tuan'])} tương đối ổn định (thay đổi {pct:+.0f}%).")

        st.markdown("##### 🔍 Yếu tố nào ảnh hưởng nhiều nhất đến mô hình (tính chung)")
        imp_display = importance_df.copy()
        imp_display["feature_vi"] = imp_display["feature"].map(FEATURE_LABELS_VI).fillna(imp_display["feature"])
        imp_display = imp_display.groupby("feature_vi")["importance"].sum().sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.barh(imp_display.index, imp_display.values, color="#3498db")
        ax2.set_xlabel("Mức độ quan trọng")
        st.pyplot(fig2)
        st.caption("Đây là mức ảnh hưởng chung của toàn mô hình. Giải thích riêng cho từng tuần xem ở mục "
                   "'🔍 Vì sao tuần X dự báo như vậy?' phía trên.")


# =============================================================================
# TRANG: CẢNH BÁO CẬN DATE
# =============================================================================
elif page == "⏰ Cảnh báo cận date":
    st.title("Cảnh báo lô hàng cận date")

    near = batch_snapshot[batch_snapshot["days_to_expiry"] <= expiry_threshold].copy()
    near = near.merge(dim_stores[["store_id", "store_name"]], on="store_id") \
        .merge(dim_skus[["sku_id", "sku_name", "category"]], on="sku_id") \
        .sort_values("days_to_expiry")

    st.metric("Tổng số lô cần xử lý", len(near))

    if len(near) == 0:
        st.success("Không có lô hàng nào cận date trong ngưỡng hiện tại.")
    else:
        display = near[["store_name", "sku_name", "category", "qty_remaining", "days_to_expiry"]].copy()
        display.columns = ["Cửa hàng", "Mặt hàng", "Nhóm hàng", "Số lượng còn", "Số ngày còn lại"]

        def status_icon(days):
            if days <= 0:
                return "🔴 Đã quá hạn"
            elif days <= 2:
                return "🟡 Ưu tiên xử lý"
            return "🟢 Còn thời gian"
        display.insert(0, "Trạng thái", display["Số ngày còn lại"].apply(status_icon))

        def highlight(row):
            color = "background-color: #ffcccc" if row["Số ngày còn lại"] <= 0 else (
                "background-color: #fff3cd" if row["Số ngày còn lại"] <= 2 else "")
            return [color] * len(row)

        st.dataframe(display.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)
        st.caption("🔴 Đã quá hạn — cần thu hồi ngay | 🟡 Còn ≤ 2 ngày — ưu tiên xử lý | 🟢 Còn thời gian")
        csv = display.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Tải danh sách cận date (CSV)", csv, "canh_bao_can_date.csv", "text/csv")


# =============================================================================
# TRANG: KHO TRUNG TÂM & KỆ
# =============================================================================
elif page == "🏭 Kho trung tâm & Kệ":
    st.title("Quản lý kho trung tâm")

    st.markdown(
        f"📍 **Tọa độ kho:** ({dim_warehouse['x'].iloc[0]:.1f}, {dim_warehouse['y'].iloc[0]:.1f}) &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Tổng sức chứa:** {dim_warehouse['tong_suc_chua'].iloc[0]:,.0f} đơn vị"
    )

    st.markdown("#### 🗺️ Bản đồ mạng lưới")
    _recent = fact_store_daily[fact_store_daily["day_idx"] >= N_DAYS - 7]
    _avg = fact_store_daily[fact_store_daily["day_idx"] >= N_DAYS - 30].groupby(["store_id", "sku_id"])["units_sold"].mean()
    _latest = fact_store_daily[fact_store_daily["day_idx"] == N_DAYS - 1].set_index(["store_id", "sku_id"])["closing_stock"]
    _days_left = (_latest / _avg.replace(0, np.nan)).reset_index(name="so_ngay_du")
    urgent_store_ids = set(_days_left[_days_left["so_ngay_du"] <= 2]["store_id"])

    map_choice = st.radio(
        "Loại bản đồ", ["Bản đồ tĩnh (mặc định, luôn hoạt động)", "OpenStreetMap tương tác", "Google Maps (cần API key riêng)"],
        horizontal=True,
    )

    if map_choice == "Google Maps (cần API key riêng)":
        google_api_key = st.text_input("Google Maps API key", type="password",
                                        help="Cần tài khoản Google Cloud có gắn thẻ thanh toán, đã bật 'Maps Embed API'.")
        if google_api_key and len(urgent_store_ids) > 0:
            route, route_km = solve_optimal_route(dist_matrix, list(urgent_store_ids), depot=0)
            route_stops = [sid for sid in route if sid != 0]  # bỏ depot khỏi danh sách waypoint
            url = render_google_maps_route(google_api_key, dim_stores, dim_warehouse, route_stops)
            components.iframe(url, height=450)
            st.caption(f"Tuyến đường thật qua {len(route_stops)} cửa hàng cần xử lý gấp — tổng {route_km:.1f} km (đường chim bay, Google có thể chọn đường thật dài/ngắn hơn).")
        elif not google_api_key:
            st.info("Nhập API key để hiển thị bản đồ.")
        else:
            st.info("Không có cửa hàng nào đang cần xử lý gấp để vẽ tuyến — không có gì để hiển thị.")
        is_real_coords = ensure_latlon(dim_stores, dim_warehouse)[3]

    elif map_choice == "OpenStreetMap tương tác":
        try:
            from streamlit_folium import st_folium
            m, is_real_coords = build_network_map(dim_stores, dim_warehouse, urgent_store_ids=urgent_store_ids)
            st_folium(m, width=None, height=450, use_container_width=True)
        except Exception as e:
            st.warning(f"Không tải được bản đồ tương tác ({e}) — hiển thị bản đồ tĩnh thay thế.")
            fig, is_real_coords = build_static_map(dim_stores, dim_warehouse, urgent_store_ids=urgent_store_ids, dist=dist_matrix)
            st.pyplot(fig)
    else:
        fig, is_real_coords = build_static_map(dim_stores, dim_warehouse, urgent_store_ids=urgent_store_ids, dist=dist_matrix)
        st.pyplot(fig)

    st.caption("🔴 Đường/điểm đỏ: cửa hàng có mặt hàng tồn kho thấp (≤ 2 ngày) cần ưu tiên giao hàng | 🔵 Bình thường")
    if not is_real_coords:
        st.caption("⚠️ Đang dùng dữ liệu mô phỏng — vị trí trên bản đồ chỉ mang tính minh hoạ, "
                   "không phải toạ độ thật. Bật 'Dùng tọa độ cửa hàng THẬT' để xem đúng vị trí thật.")

    st.markdown("#### 📏 Bảng khoảng cách giữa các cửa hàng")
    dist_label = "đường thật (OSRM)" if st.session_state.get("using_real_road_dist") else "chim bay (Haversine)"
    st.caption(f"Đơn vị: km — đang tính theo khoảng cách {dist_label}.")
    dist_labels = ["Kho trung tâm"] + list(dim_stores["store_name"])
    dist_table = pd.DataFrame(dist_matrix.round(2), index=dist_labels, columns=dist_labels)
    st.dataframe(dist_table, use_container_width=True)
    csv_dist = dist_table.to_csv().encode("utf-8-sig")
    st.download_button("📥 Tải bảng khoảng cách (CSV)", csv_dist, "khoang_cach_cua_hang.csv", "text/csv")

    st.markdown("#### Sức chứa & tồn kho theo từng kệ")
    shelf_cap = dim_shelves.set_index("shelf_id")["suc_chua"]
    latest_shelf_stock = fact_shelf_stock_daily[fact_shelf_stock_daily["day_idx"] == N_DAYS - 1] \
        .set_index("warehouse_shelf_id")["ton_kho_ke"]
    shelf_view = dim_shelves.copy()
    shelf_view["ton_hien_tai"] = shelf_view["shelf_id"].map(latest_shelf_stock).fillna(0)
    shelf_view["ty_le_su_dung_%"] = (shelf_view["ton_hien_tai"] / shelf_view["suc_chua"] * 100).round(1)
    st.dataframe(
        shelf_view[["ten_ke", "cong_nang", "suc_chua", "ton_hien_tai", "ty_le_su_dung_%"]]
        .rename(columns={"ten_ke": "Kệ", "cong_nang": "Công năng", "suc_chua": "Sức chứa",
                          "ton_hien_tai": "Tồn hiện tại", "ty_le_su_dung_%": "Tỷ lệ sử dụng (%)"}),
        use_container_width=True, hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    for shelf_id, g in fact_shelf_stock_daily.groupby("warehouse_shelf_id"):
        ln, = ax.plot(g["day_idx"], g["ton_kho_ke"], label=f"Kệ {int(shelf_id)}")
        ax.axhline(shelf_cap.get(shelf_id, np.nan), linestyle=":", alpha=0.4, color=ln.get_color())
    ax.set_title("Tồn kho theo từng kệ theo thời gian (đường chấm = sức chứa tối đa)")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Tồn kho")
    ax.legend()
    st.pyplot(fig)

    st.markdown("#### Danh mục mặt hàng theo kệ")
    sku_by_shelf = dim_skus.merge(dim_shelves[["shelf_id", "ten_ke"]], left_on="warehouse_shelf_id", right_on="shelf_id")
    for shelf_id, g in sku_by_shelf.groupby("ten_ke"):
        st.markdown(f"**{shelf_id}:** {', '.join(g['sku_name'])}")

    st.markdown("#### Nhập / Xuất kho theo SKU")
    wh_io = fact_warehouse_inbound.groupby("sku_id")["so_luong_nhap"].sum().to_frame()
    wh_io["tong_xuat"] = fact_warehouse_outbound.groupby("sku_id")["so_luong_xuat"].sum()
    wh_io = wh_io.merge(dim_skus[["sku_id", "sku_name"]], on="sku_id")

    recent = fact_store_daily[fact_store_daily["day_idx"] >= N_DAYS - 14]
    fc7 = recent.groupby("sku_id")["units_sold"].mean().reset_index()
    fc7["du_bao_7_ngay"] = fc7["units_sold"] * n_stores * 7
    wh_io = wh_io.merge(fc7[["sku_id", "du_bao_7_ngay"]], on="sku_id")

    last_stock = fact_warehouse_daily.dropna(subset=["closing_stock_snapshot"]).sort_values("day_idx") \
        .groupby("sku_id")["closing_stock_snapshot"].last().reset_index() \
        .rename(columns={"closing_stock_snapshot": "ton_hien_tai"})
    wh_io = wh_io.merge(last_stock, on="sku_id")
    wh_io["chenh_lech_%"] = ((wh_io["ton_hien_tai"] - wh_io["du_bao_7_ngay"]) / wh_io["du_bao_7_ngay"] * 100).round(1)

    display_wh = wh_io[["sku_name", "so_luong_nhap", "tong_xuat", "ton_hien_tai", "du_bao_7_ngay", "chenh_lech_%"]]
    display_wh.columns = ["Mặt hàng", "Tổng nhập", "Tổng xuất", "Tồn hiện tại", "Dự báo 7 ngày", "Chênh lệch (%)"]
    st.dataframe(display_wh.round(1), use_container_width=True, hide_index=True)

    csv_wh = display_wh.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Tải báo cáo kho (CSV)", csv_wh, "bao_cao_kho.csv", "text/csv")


# =============================================================================
# TRANG: ĐỘI XE & TÀI XẾ
# =============================================================================
elif page == "🚚 Đội xe & Tài xế":
    st.title("Đội xe & Tài xế")

    tab_overview, tab_schedule, tab_leave = st.tabs(
        ["📊 Tổng quan đội xe", "🗓️ Lịch trình tuần", "👤 Nghỉ phép & Hiệu suất"])

    # ---------------- TAB 1: Tổng quan (như cũ) ----------------
    with tab_overview:
        today = fact_fleet_daily[fact_fleet_daily["day_idx"] == N_DAYS - 1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng số xe", len(dim_vehicles))
        c2.metric("Tổng số tài xế", len(dim_drivers))
        c3.metric("Đang giao hàng (hôm nay)", int((today["trang_thai"] == "Đang giao hàng").sum()))
        c4.metric("Nghỉ phép (hôm nay)", int((today["trang_thai"] == "Nghỉ phép").sum()))

        st.markdown("#### Trạng thái hôm nay")
        today_display = today.merge(dim_drivers[["driver_id", "ten_tai_xe"]], on="driver_id", how="left") \
            .merge(dim_vehicles[["vehicle_id", "bien_so", "loai_xe"]], on="vehicle_id", how="left")
        st.dataframe(
            today_display[["ten_tai_xe", "bien_so", "loai_xe", "so_gio_da_lam", "trang_thai"]]
            .rename(columns={"ten_tai_xe": "Tài xế", "bien_so": "Biển số xe", "loai_xe": "Loại xe",
                              "so_gio_da_lam": "Giờ đã làm", "trang_thai": "Trạng thái"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### Danh mục xe")
        vehicles_display = dim_vehicles.rename(columns={
            "bien_so": "Biển số", "loai_xe": "Loại xe", "tai_trong_kg": "Tải trọng (kg)",
            "the_tich_m3": "Thể tích (m³)", "co_ngan_lanh": "Có ngăn lạnh",
        }).drop(columns=["vehicle_id"])
        vehicles_display["Có ngăn lạnh"] = vehicles_display["Có ngăn lạnh"].map({True: "✅ Có", False: "❌ Không"})
        st.dataframe(vehicles_display, use_container_width=True, hide_index=True)

        st.markdown("#### Thống kê vận hành (toàn bộ giai đoạn dữ liệu)")
        col1, col2 = st.columns(2)
        with col1:
            status_counts = fact_fleet_daily["trang_thai"].value_counts(normalize=True) * 100
            fig, ax = plt.subplots(figsize=(5.5, 4))
            status_counts.plot(kind="bar", ax=ax, color="#3498db")
            ax.set_title("Tỷ lệ trạng thái (%)")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)
        with col2:
            avg_hours = fact_fleet_daily.dropna(subset=["driver_id"]).groupby("driver_id")["so_gio_da_lam"].mean()
            avg_hours = avg_hours.to_frame("gio_lam_tb").merge(dim_drivers.set_index("driver_id"), left_index=True, right_index=True)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.bar(avg_hours["ten_tai_xe"], avg_hours["gio_lam_tb"], color="#2ecc71")
            ax.axhline(avg_hours["gio_lam_toi_da_ngay"].mean(), linestyle="--", color="red", alpha=0.6, label="TB giờ tối đa")
            ax.set_title("Giờ làm việc trung bình/ngày")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            st.pyplot(fig)

    # ---------------- TAB 2: Lịch trình tuần (3 tài xế cố định) ----------------
    with tab_schedule:
        st.caption("Ghép cố định mỗi tài xế với 1 xe và 1 khu vực giao hàng riêng cho cả tuần. "
                   "Lịch làm/nghỉ từng ngày dựa trên nhịp giao hàng thật quan sát được 7 ngày gần nhất.")

        names_input = st.text_area("Danh sách tên tài xế (mỗi dòng 1 tên, số dòng = số xe sẽ dùng)",
                                    value="\n".join(DEFAULT_DRIVER_NAMES), height=100)
        driver_names_new = [n.strip() for n in names_input.split("\n") if n.strip()]

        if st.button("🗓️ Lập lịch trình", type="primary"):
            with st.spinner("Đang tính toán lịch trình..."):
                n_needed = len(driver_names_new)
                data_small = build_full_dataset(
                    n_stores=n_stores, n_skus=n_skus, n_clusters=n_clusters, n_days=n_days,
                    n_vehicles=n_needed, n_drivers=n_needed, replenish_cycle=replenish_cycle, seed=seed,
                    use_real_stores=use_real_stores, driver_names=driver_names_new,
                )
                fwo_small = data_small["fact_store_daily"]
                fwo_small = fwo_small[fwo_small["units_received"] > 0][["day_idx", "date", "store_id"]]
                result = build_weekly_driver_schedule(
                    data_small["dim_stores"], data_small["dim_vehicles"], data_small["dim_drivers"],
                    fwo_small, data_small["dist"],
                )
            st.session_state["driver_schedule"] = result

        if "driver_schedule" in st.session_state:
            result = st.session_state["driver_schedule"]
            st.markdown("#### 👤 Ghép tài xế — xe — khu vực phụ trách")
            st.dataframe(result["assignment"], use_container_width=True, hide_index=True)

            st.markdown("#### 📅 Lịch làm việc cả tuần")
            sched = result["schedule"]
            pivot = sched.pivot(index="Tài xế", columns="Ngày", values="Trạng thái")
            day_order = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
            pivot = pivot[[d for d in day_order if d in pivot.columns]]
            st.dataframe(pivot, use_container_width=True)

            st.markdown("#### Chi tiết từng ngày (cửa hàng, quãng đường, giờ làm)")
            st.dataframe(sched, use_container_width=True, hide_index=True)

            total_hours = sched.groupby("Tài xế")["Giờ làm ước tính"].sum().reset_index()
            total_hours.columns = ["Tài xế", "Tổng giờ làm cả tuần"]
            st.markdown("#### Tổng giờ làm cả tuần / tài xế")
            st.dataframe(total_hours.round(1), use_container_width=True, hide_index=True)

            csv = sched.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Tải lịch trình (CSV)", csv, "lich_trinh_tai_xe.csv", "text/csv")
        else:
            st.info("Nhập tên tài xế rồi nhấn 'Lập lịch trình' để bắt đầu.")

    # ---------------- TAB 3: Nghỉ phép & Hiệu suất (dữ liệu dùng chung với app Tài xế sau này) ----------------
    with tab_leave:
        st.caption("Dữ liệu này lưu trong database dùng chung (`outputs/driver_shared.db`) — khi app Tài xế "
                   "hoàn thiện, tài xế sẽ tự gửi yêu cầu và ghi nhận hiệu suất thật vào đây thay vì dữ liệu mẫu.")

        driver_id_name_map = dict(zip(dim_drivers["driver_id"], dim_drivers["ten_tai_xe"]))
        ddb.seed_sample_data(driver_id_name_map, n_days=30)
        driver_names_all = list(dim_drivers["ten_tai_xe"])

        st.markdown("#### ⏳ Yêu cầu xin nghỉ đang chờ duyệt")
        pending = ddb.get_pending_requests()
        pending = pending[pending["driver_name"].isin(driver_names_all)]
        if len(pending) == 0:
            st.success("Không có yêu cầu nào đang chờ duyệt.")
        else:
            for _, req in pending.iterrows():
                c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
                c1.write(f"**{req['driver_name']}**")
                c2.write(req["ngay"])
                c3.write(f"Lý do: {req['ly_do'] or '—'}" + (f" (📎 {req['minh_chung']})" if req["minh_chung"] else ""))
                with c4:
                    bcol1, bcol2 = st.columns(2)
                    if bcol1.button("✅ Duyệt", key=f"approve_{req['id']}"):
                        ddb.approve_request(int(req["id"]), approved=True)
                        st.rerun()
                    if bcol2.button("❌ Từ chối", key=f"reject_{req['id']}"):
                        ddb.approve_request(int(req["id"]), approved=False)
                        st.rerun()

        st.markdown("#### 📊 Bảng tổng hợp tháng theo tài xế")
        month_choice = st.date_input("Chọn 1 ngày trong tháng cần xem", value=datetime.date.today())
        month_start = month_choice.replace(day=1)
        next_month = (month_start.replace(day=28) + datetime.timedelta(days=4)).replace(day=1)
        month_end = next_month - datetime.timedelta(days=1)

        summary = ddb.get_monthly_summary(driver_names_all, month_start.isoformat(), month_end.isoformat())
        vehicle_by_driver = dim_drivers.merge(
            fact_fleet_daily.dropna(subset=["vehicle_id"]).drop_duplicates("driver_id")[["driver_id", "vehicle_id"]],
            on="driver_id", how="left",
        ).merge(dim_vehicles[["vehicle_id", "loai_xe"]], on="vehicle_id", how="left")
        summary = summary.merge(vehicle_by_driver[["ten_tai_xe", "loai_xe"]],
                                 left_on="Tên tài xế", right_on="ten_tai_xe", how="left") \
            .rename(columns={"loai_xe": "Loại xe"}).drop(columns=["ten_tai_xe"])
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("#### 📅 Lịch trình tuần (Làm việc / Xin nghỉ)")
        week_choice = st.date_input("Chọn ngày bắt đầu tuần cần xem", value=datetime.date.today(), key="week_leave_view")
        week_dates = [week_choice + datetime.timedelta(days=i) for i in range(7)]
        all_reqs = ddb.get_all_requests(driver_names_all)
        all_reqs["ngay"] = pd.to_datetime(all_reqs["ngay"]).dt.date

        grid = pd.DataFrame(index=driver_names_all, columns=[d.strftime("%d/%m (%a)") for d in week_dates])
        for name in driver_names_all:
            for d in week_dates:
                col = d.strftime("%d/%m (%a)")
                match = all_reqs[(all_reqs["driver_name"] == name) & (all_reqs["ngay"] == d)]
                if len(match) == 0:
                    grid.loc[name, col] = "—"
                else:
                    r = match.iloc[-1]
                    grid.loc[name, col] = f"{r['loai']} ({r['trang_thai']})" if r["loai"] == "Xin nghỉ" else r["loai"]
        st.dataframe(grid, use_container_width=True)


# =============================================================================
# TRANG: CHI TIẾT CỬA HÀNG
# =============================================================================
elif page == "🏬 Chi tiết cửa hàng":
    st.title("Chi tiết theo cửa hàng")

    sel_store = st.selectbox("Chọn cửa hàng", sorted(dim_stores["store_name"]))
    store_id = dim_stores[dim_stores["store_name"] == sel_store]["store_id"].iloc[0]

    store_data = fact_store_daily[fact_store_daily["store_id"] == store_id]
    last_day_data = store_data[store_data["day_idx"] == N_DAYS - 1].merge(
        dim_skus[["sku_id", "sku_name", "category"]], on="sku_id")

    st.markdown(f"##### Tồn kho hiện tại — {sel_store}")
    st.dataframe(
        last_day_data[["sku_name", "category", "closing_stock"]]
        .rename(columns={"sku_name": "Mặt hàng", "category": "Nhóm", "closing_stock": "Tồn hiện tại"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("##### Lịch sử bán hàng 30 ngày gần nhất (tổng tất cả SKU)")
    hist = store_data[store_data["day_idx"] >= N_DAYS - 30].groupby("day_idx")["units_sold"].sum()
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(hist.index, hist.values, color="#3498db")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Sản lượng bán")
    st.pyplot(fig)


# =============================================================================
# TRANG: QUY TRÌNH 3 TẦNG (Thu thập -> Xử lý cuối tuần -> Dự báo & Kế hoạch kho)
# =============================================================================
elif page == "🔄 Quy trình 3 tầng":
    st.title("Quy trình xử lý dữ liệu — 3 tầng")
    st.caption("Tầng 1: nhận dữ liệu hàng ngày từ POS  →  Tầng 2: xử lý cuối tuần  →  Tầng 3: dự báo & lập kế hoạch nhập kho")

    tab1, tab2, tab3 = st.tabs(["1️⃣ Thu thập dữ liệu", "2️⃣ Xử lý cuối tuần", "3️⃣ Dự báo & Kế hoạch kho"])

    # ---------------- TẦNG 1 ----------------
    with tab1:
        st.subheader("Tầng 1 — Nhận dữ liệu bán hàng hôm nay từ POS")
        st.caption("Mỗi ngày, cửa hàng xuất dữ liệu bán hàng ra file rồi tải lên đây. "
                   "Dữ liệu được cộng dồn liên tục cho đến khi chạy xử lý cuối tuần (Tầng 2).")

        current_log = read_raw_log()
        st.metric("Số dòng đã tích luỹ trong tuần này (chưa xử lý)", len(current_log))

        st.download_button("📄 Tải file mẫu (CSV)", get_template_csv(), "mau_du_lieu_ban_hang.csv",
                            "text/csv", key="template_layer1")
        daily_file = st.file_uploader("Chọn file dữ liệu hôm nay (CSV/Excel)", type=["csv", "xlsx", "xls"], key="layer1_upload")
        if st.button("📥 Nhận dữ liệu vào hệ thống", disabled=daily_file is None, key="layer1_btn"):
            try:
                raw_df = read_uploaded_file(daily_file)
                result = receive_daily_pos_data(raw_df, dim_stores, dim_skus)
                if result["success"]:
                    st.success(f"Đã nhận {result['n_rows_added']} dòng dữ liệu.")
                else:
                    st.error("Không nhận được dữ liệu.")
                for e in result["errors"]:
                    st.error(e)
                for w in result["warnings"]:
                    st.warning(w)
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {e}")

        if len(current_log) > 0:
            with st.expander("Xem dữ liệu thô đã tích luỹ"):
                st.dataframe(current_log, use_container_width=True, hide_index=True)

    # ---------------- TẦNG 2 ----------------
    with tab2:
        st.subheader("Tầng 2 — Xử lý cuối tuần: tổng hợp bán ra + tồn kho")
        st.caption("Chạy vào cuối mỗi tuần. Tổng hợp toàn bộ dữ liệu Tầng 1 đã tích luỹ, xuất báo cáo, "
                   "lưu vào lịch sử dài hạn cho Tầng 3, rồi xoá dữ liệu tạm để bắt đầu tuần mới.")

        current_log = read_raw_log()
        if len(current_log) == 0:
            st.info("Chưa có dữ liệu nào từ Tầng 1 để xử lý.")
        else:
            st.write(f"Sẵn sàng xử lý **{len(current_log)} dòng** dữ liệu đã thu thập.")
            if st.button("⚙️ Chạy xử lý cuối tuần", key="layer2_btn"):
                result = run_weekly_batch(dim_stores, dim_skus, reset_log=True)
                if result["success"]:
                    st.success(result["message"])
                    st.session_state["last_weekly_report"] = result["report_df"]

                    # Đồng bộ dữ liệu này sang TOÀN BỘ app (Tổng quan, Kho trung tâm, Chi tiết cửa hàng...)
                    # thay vì chỉ hiển thị riêng trong trang này.
                    full_history = load_full_history()
                    if len(full_history) > 0:
                        data["fact_store_daily"] = full_history
                        st.session_state["v4_data"] = data
                        st.info("Đã đồng bộ dữ liệu thật này sang toàn bộ app — các trang khác "
                                "(Tổng quan, Kho trung tâm, Chi tiết cửa hàng...) sẽ dùng dữ liệu này "
                                "thay vì dữ liệu mô phỏng.")
                else:
                    st.error(result["message"])

        if "last_weekly_report" in st.session_state:
            st.markdown("##### Báo cáo tuần gần nhất")
            st.dataframe(st.session_state["last_weekly_report"], use_container_width=True, hide_index=True)
            csv = st.session_state["last_weekly_report"].to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Tải báo cáo tuần (CSV)", csv, "bao_cao_tuan.csv", "text/csv")

    # ---------------- TẦNG 3 ----------------
    with tab3:
        st.subheader("Tầng 3 — Dự báo nhu cầu & Kế hoạch nhập kho trung tâm")
        st.caption("Dùng toàn bộ lịch sử đã tích luỹ qua các tuần (từ Tầng 2) để huấn luyện AI, "
                   "dự báo nhu cầu tuần tới và tính tổng lượng cần nhập cho KHO TRUNG TÂM (không phải từng cửa hàng lẻ).")

        if st.button("🤖 Chạy dự báo & lập kế hoạch", key="layer3_btn"):
            with st.spinner("Đang huấn luyện mô hình trên toàn bộ lịch sử..."):
                result = run_forecast_and_plan_warehouse(dim_stores, dim_skus)
            if result["success"]:
                st.success(result["message"])
                st.session_state["last_warehouse_plan"] = result["warehouse_plan"]

                # Đồng bộ mô hình dự báo này sang trang "📈 Dự báo nhu cầu" — dùng chung 1 mô hình,
                # không để trang đó tiếp tục hiển thị dự báo huấn luyện trên dữ liệu mô phỏng cũ.
                forecast_detail = result["forecast_detail"].copy()
                forecast_detail["true_demand"] = forecast_detail.get("true_demand", forecast_detail["q50"])
                st.session_state["v4_forecast"] = (forecast_detail, importance_df)
                st.info("Đã cập nhật mô hình dự báo này cho toàn bộ app — trang '📈 Dự báo nhu cầu' "
                        "giờ sẽ dùng đúng mô hình vừa huấn luyện trên dữ liệu thật này.")
            else:
                st.warning(result["message"])

        if "last_warehouse_plan" in st.session_state:
            st.markdown("##### Kế hoạch nhập kho trung tâm — tuần tới")
            st.dataframe(st.session_state["last_warehouse_plan"], use_container_width=True, hide_index=True)
            csv = st.session_state["last_warehouse_plan"].to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Tải kế hoạch nhập kho (CSV)", csv, "ke_hoach_nhap_kho.csv", "text/csv")


# =============================================================================
# TRANG: GHÉP TUYẾN (Route Consolidation)
# =============================================================================
elif page == "🔗 Ghép tuyến":
    st.title("Đề xuất ghép tuyến giao hàng")
    st.caption("So sánh chi phí nếu giao riêng từng cửa hàng so với gộp thành 1 chuyến — "
               "tính từ khoảng cách thật (Haversine) và thuật toán tối ưu tuyến (Nearest-Neighbor + 2-opt).")

    COST_PER_KM = 1.2
    FIXED_DISPATCH_COST = 25.0

    daily_cmp = compute_daily_route_comparison(fact_warehouse_outbound, dist_matrix,
                                                cost_per_km=COST_PER_KM, fixed_dispatch_cost=FIXED_DISPATCH_COST)

    all_store_ids = list(dim_stores["store_id"])
    suggestions = suggest_route_merges(dist_matrix, all_store_ids, dim_stores,
                                        cost_per_km=COST_PER_KM, fixed_dispatch_cost=FIXED_DISPATCH_COST)

    if len(daily_cmp) == 0:
        st.info("Chưa có đủ lịch sử xuất kho để tính so sánh ghép tuyến.")
    else:
        avg_before = daily_cmp["chi_phi_km_truoc"].mean()
        avg_after = daily_cmp["chi_phi_km_sau"].mean()
        pct_change = (avg_after - avg_before) / avg_before * 100
        total_saved_km = daily_cmp["quang_duong_tiet_kiem_km"].sum()
        n_recommend = (suggestions["Ưu tiên"] == "🟢 Nên ghép ngay").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chi phí trung bình/km (sau ghép)", f"{avg_after:,.0f}đ", f"{pct_change:+.0f}% so với trước")
        c2.metric("Số cặp tuyến nên ghép ngay", f"{n_recommend}/{len(suggestions)}")
        c3.metric("Quãng đường tiết kiệm", f"{total_saved_km:,.1f} km", f"trong {len(daily_cmp)} ngày gần nhất")
        c4.metric("Tiết kiệm chi phí ước tính", f"{total_saved_km*COST_PER_KM:,.0f}đ")

        st.markdown("#### Chi phí/km theo ngày — trước vs sau ghép tuyến")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily_cmp["day_idx"], daily_cmp["chi_phi_km_truoc"], "o-", color="#e74c3c", label="Trước ghép tuyến")
        ax.plot(daily_cmp["day_idx"], daily_cmp["chi_phi_km_sau"], "o-", color="#2ecc71", label="Sau ghép tuyến")
        ax.set_xlabel("Ngày"); ax.set_ylabel("Chi phí/km (VNĐ)")
        ax.legend()
        st.pyplot(fig)

    st.markdown("#### Đề xuất ghép tuyến — xếp theo mức tiết kiệm")
    st.dataframe(suggestions, use_container_width=True, hide_index=True)
    st.caption("🟢 Nên ghép ngay: tiết kiệm ≥15% và thời gian chuyến ≤3 giờ | 🟡 Cân nhắc: tiết kiệm 5-15% | 🔴 Chưa ưu tiên: tiết kiệm <5%")
    csv = suggestions.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Tải đề xuất ghép tuyến (CSV)", csv, "de_xuat_ghep_tuyen.csv", "text/csv")

    st.markdown("---")
    st.markdown("#### 🤖 Đề xuất ghép tuyến TỰ ĐỘNG (toàn bộ mạng lưới — không cần tự chọn)")
    st.caption("Dùng thuật toán Clarke-Wright Savings (kinh điển trong Operations Research) để tự động "
               "tìm cách gộp TẤT CẢ cửa hàng thành các tuyến đa điểm tối ưu — không cần dò tay từng tổ hợp.")

    max_hours = st.slider("Giới hạn thời gian tối đa mỗi tuyến (giờ)", 1.0, 8.0, 4.0, step=0.5)
    all_store_ids_auto = list(dim_stores["store_id"])

    # Nhu cầu tuần tới theo từng cửa hàng (từ dự báo Tầng 3) -> để xét ràng buộc thể tích xe khi ghép tuyến
    week1_start_gt = pred_df["day_idx"].min()
    pred_week1_gt = pred_df[pred_df["day_idx"] < week1_start_gt + 7]
    weekly_demand_gt = pred_week1_gt.groupby(["store_id", "sku_id"])["q50"].sum().reset_index()
    demand_by_store_gt = {
        sid: dict(zip(g["sku_id"], g["q50"])) for sid, g in weekly_demand_gt.groupby("store_id")
    }
    catalog_gt = build_product_catalog(dim_skus)

    auto_routes = auto_suggest_merged_routes(dist_matrix, all_store_ids_auto, dim_stores,
                                              cost_per_km=COST_PER_KM, fixed_dispatch_cost=FIXED_DISPATCH_COST,
                                              max_route_hours=max_hours,
                                              demand_by_store=demand_by_store_gt, catalog=catalog_gt,
                                              dim_vehicles=dim_vehicles)
    st.dataframe(auto_routes, use_container_width=True, hide_index=True)
    st.caption("✅ Đã xét ĐỒNG THỜI quãng đường ngắn nhất VÀ thể tích xe (dựa trên dự báo nhu cầu tuần tới) — "
               "🧊 = tuyến có hàng lạnh, bắt buộc xe có ngăn lạnh.")
    csv_auto = auto_routes.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Tải đề xuất tự động (CSV)", csv_auto, "de_xuat_tu_dong.csv", "text/csv")

    st.markdown("---")
    st.markdown("#### 🔗 Ghép tuyến tự chọn (nhiều cửa hàng cùng lúc)")
    st.caption("Chọn các cửa hàng muốn ghép chung 1 chuyến — hệ thống tự tính thứ tự đi tối ưu "
               "(Kho → ... → Kho) và so sánh với việc giao riêng từng cửa hàng.")

    chosen_names = st.multiselect("Chọn cửa hàng cần ghép tuyến", dim_stores["store_name"].tolist(),
                                   default=list(dim_stores["store_name"].iloc[:3]))

    if len(chosen_names) < 2:
        st.info("Chọn ít nhất 2 cửa hàng để ghép tuyến.")
    else:
        chosen_ids = dim_stores[dim_stores["store_name"].isin(chosen_names)]["store_id"].tolist()
        route, merged_km = solve_optimal_route(dist_matrix, chosen_ids, depot=0)

        route_names = ["Kho"] + [dim_stores.set_index("store_id").loc[sid, "store_name"] for sid in route[1:-1]] + ["Kho"]
        st.markdown("##### Thứ tự tuyến tối ưu")
        st.markdown("### " + " → ".join(route_names))

        separate_km = sum(2 * dist_matrix[0, sid] for sid in chosen_ids)
        separate_cost = separate_km * COST_PER_KM + len(chosen_ids) * FIXED_DISPATCH_COST
        merged_cost = merged_km * COST_PER_KM + FIXED_DISPATCH_COST
        savings_pct = (separate_cost - merged_cost) / separate_cost * 100 if separate_cost > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Quãng đường ghép chung", f"{merged_km:.1f} km", f"{merged_km - separate_km:+.1f} km so với giao riêng")
        c2.metric("Chi phí ghép chung", f"{merged_cost:,.0f}đ", f"{merged_cost - separate_cost:+,.0f}đ")
        c3.metric("Tiết kiệm", f"{savings_pct:.1f}%")

        with st.expander("Xem chi tiết nếu giao RIÊNG từng cửa hàng (không ghép)"):
            for sid, name in zip(chosen_ids, chosen_names):
                st.write(f"- Kho → {name} → Kho: {2*dist_matrix[0, sid]:.2f} km")
            st.write(f"**Tổng nếu giao riêng: {separate_km:.1f} km, chi phí {separate_cost:,.0f}đ**")


# =============================================================================
# TRANG: KẾ HOẠCH PHÂN PHỐI TƯƠNG LAI (Cửa xuất + Xe + Tài xế)
# =============================================================================
elif page == "📦 Kế hoạch phân phối":
    st.title("Kế hoạch phân phối tuần tới")
    st.caption("Dựa trên dự báo nhu cầu (Tầng 3) → xác định mặt hàng xuất qua cửa nào, "
               "tuyến nào cần xe/tài xế gì.")

    col1, col2 = st.columns(2)
    n_gates = col1.slider("Số cửa xuất kho (theo vị trí kệ)", 1, 4, 2)
    max_route_hours_plan = col2.slider("Giới hạn thời gian tối đa mỗi tuyến (giờ)", 1.0, 8.0, 4.0, step=0.5)
    st.caption("Cửa xuất = hàng ở kệ nào xuất qua cửa đó (cố định, không đổi theo cửa hàng đích đến). "
               "Tuyến giao hàng được gộp TỰ ĐỘNG bằng thuật toán Clarke-Wright (giống hệt trang 'Ghép tuyến') "
               "— không chia đều theo cụm địa lý cố định nữa, đảm bảo 2 trang cho kết quả nhất quán.")
    st.caption("✅ Chọn xe giờ dựa trên **thể tích thật** (xem trang '📦 Thông tin sản phẩm' để biết kích thước "
               "đóng gói từng mặt hàng) — không còn ước tính trọng lượng chung chung như trước.")

    # Chỉ lấy đúng 7 ngày đầu của dự báo (tuần tới), tránh cộng dồn cả giai đoạn test dài hơn
    week1_start = pred_df["day_idx"].min()
    pred_week1 = pred_df[pred_df["day_idx"] < week1_start + 7]

    _dates_w1 = pd.to_datetime(pred_week1["date"])
    st.info(
        f"📅 **Kế hoạch này áp dụng cho tuần: {_dates_w1.min().strftime('%d/%m/%Y')} "
        f"→ {_dates_w1.max().strftime('%d/%m/%Y')}** "
        f"(7 ngày đầu tiên của giai đoạn dự báo hiện tại — xem cùng khoảng ngày này ở trang "
        f"'📈 Dự báo nhu cầu', mục '📅 Dự báo chi tiết theo từng ngày')."
    )

    if st.button("📋 Lập kế hoạch tuần tới", type="primary"):
        with st.spinner("Đang tính toán..."):
            result = plan_future_distribution(
                pred_week1, dim_stores, dim_skus, dim_shelves, dim_vehicles, dim_drivers, dist_matrix,
                n_gates=n_gates, max_route_hours=max_route_hours_plan,
            )
        st.session_state["dist_plan"] = result

    if "dist_plan" in st.session_state:
        result = st.session_state["dist_plan"]

        st.markdown("#### 🚪 Kệ nào xuất qua cửa nào (cố định)")
        st.dataframe(result["shelf_gates"][["ten_ke", "cong_nang", "cua_xuat"]].rename(
            columns={"ten_ke": "Kệ", "cong_nang": "Công năng", "cua_xuat": "Cửa xuất"}),
            use_container_width=True, hide_index=True)

        st.markdown("#### 📦 Mặt hàng nào xuất qua cửa nào")
        gate_filter = st.selectbox("Lọc theo cửa xuất", ["Tất cả"] + sorted(result["sku_gate_plan"]["Cửa xuất"].unique()))
        display_sku = result["sku_gate_plan"] if gate_filter == "Tất cả" else \
            result["sku_gate_plan"][result["sku_gate_plan"]["Cửa xuất"] == gate_filter]
        st.dataframe(display_sku, use_container_width=True, hide_index=True)

        st.markdown("#### 🚚 Tuyến / Xe / Tài xế theo từng khu vực giao hàng")
        st.dataframe(result["route_plan"], use_container_width=True, hide_index=True)
        st.caption("'Cần lấy hàng từ' cho biết tuyến đó cần lấy hàng ở những cửa xuất nào trước khi khởi hành "
                   "(vì các cửa hàng trên cùng 1 tuyến có thể cần mặt hàng nằm ở kệ/cửa khác nhau).")
        if result["route_plan"]["Loại xe đề xuất"].str.contains("⚠").any():
            st.warning("Có tuyến vượt tải xe lớn nhất hiện có — cần chia thành nhiều chuyến trong tuần, "
                       "hoặc cân nhắc bổ sung thêm xe tải trọng lớn hơn.")
        if (result["route_plan"]["Tài xế đề xuất"] == "⚠ Không đủ tài xế rảnh").any():
            st.warning("Có tuyến chưa đủ tài xế rảnh trong ngày — cần sắp xếp lại lịch làm việc hoặc thuê thêm.")

        csv = result["sku_gate_plan"].to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Tải kế hoạch mặt hàng theo cửa xuất (CSV)", csv, "ke_hoach_cua_xuat.csv", "text/csv")
    else:
        st.info("Nhấn 'Lập kế hoạch tuần tới' để bắt đầu.")


# =============================================================================
# TRANG: THÔNG TIN SẢN PHẨM & ĐÓNG GÓI
# =============================================================================
elif page == "📐 Thông tin sản phẩm":
    st.title("Thông tin sản phẩm & Đóng gói")
    st.caption("Kích thước thùng đóng gói từng mặt hàng — dùng để tính thể tích thật khi xếp hàng lên xe "
               "(trang 'Kế hoạch phân phối' dùng đúng số liệu này để chọn xe tối ưu thể tích).")

    catalog = build_product_catalog(dim_skus)
    display_catalog = catalog[["sku_name", "category", "Loại đóng gói",
                                "Kích thước thùng (D x R x C, cm)", "Thể tích thùng (lít)", "Số đơn vị/thùng"]]
    display_catalog.columns = ["Mặt hàng", "Nhóm hàng", "Loại đóng gói",
                                "Kích thước thùng (D x R x C, cm)", "Thể tích thùng (lít)", "Số đơn vị/thùng"]

    loai_filter = st.multiselect("Lọc theo loại đóng gói", display_catalog["Loại đóng gói"].unique().tolist(),
                                  default=display_catalog["Loại đóng gói"].unique().tolist())
    st.dataframe(display_catalog[display_catalog["Loại đóng gói"].isin(loai_filter)],
                 use_container_width=True, hide_index=True)
    st.caption("⚠️ Kích thước đang là số liệu ƯỚC TÍNH theo nhóm hàng (chưa có dữ liệu thật từng SKU cụ thể) "
               "— nên thay bằng số đo thật nếu doanh nghiệp có sẵn, để kết quả xếp xe chính xác hơn.")

    csv_catalog = display_catalog.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Tải danh mục sản phẩm (CSV)", csv_catalog, "danh_muc_san_pham.csv", "text/csv")

    st.markdown("---")
    st.markdown("#### 🧮 Thử tính thể tích & chọn xe")
    st.caption("Nhập số lượng cần chở cho vài mặt hàng, xem tổng thể tích và xe đề xuất (tối ưu % trống thấp nhất).")

    sel_skus = st.multiselect("Chọn mặt hàng", dim_skus["sku_name"].tolist(), default=dim_skus["sku_name"].tolist()[:3])
    demand_input = {}
    for name in sel_skus:
        sid = dim_skus[dim_skus["sku_name"] == name]["sku_id"].iloc[0]
        demand_input[sid] = st.number_input(f"Số lượng — {name}", min_value=0, value=50, step=10, key=f"demo_qty_{sid}")

    if sel_skus:
        from product_catalog import compute_volume_by_packaging_type, select_best_fit_vehicle
        vol_split = compute_volume_by_packaging_type(demand_input, catalog)
        needs_cold = vol_split["lanh"] > 0

        c1, c2 = st.columns(2)
        c1.metric("Thể tích hàng LẠNH", f"{vol_split['lanh']:.1f} lít")
        c2.metric("Thể tích hàng THƯỜNG", f"{vol_split['thuong']:.1f} lít")

        result = select_best_fit_vehicle(vol_split["tong"], dim_vehicles, requires_cold=needs_cold)
        c3, c4, c5 = st.columns(3)
        c3.metric("Tổng thể tích cần chở", f"{vol_split['tong']:.1f} lít")
        if result["loi"]:
            c4.metric("Xe đề xuất", "Không có xe phù hợp")
            st.error(result["loi"])
        else:
            c4.metric("Xe đề xuất", result["vehicle"]["loai_xe"])
            if result["du_the_tich"]:
                c5.metric("% thể tích trống", f"{result['empty_pct']}%")
            else:
                c5.metric("⚠ Vượt thể tích", f"{abs(result['empty_pct'])}% thiếu chỗ")
            if needs_cold:
                st.info("🧊 Lô hàng có mặt hàng cần bảo quản lạnh — chỉ chọn trong các xe có ngăn lạnh.")

