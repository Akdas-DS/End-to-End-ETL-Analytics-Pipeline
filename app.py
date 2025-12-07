# ============================================================
# Global Automotive Parts Pricing – Streamlit App
# ============================================================
# This app lets us explore automotive parts listing data,
# understand market prices, and get a smart price recommendation
# for a new listing based on historical data.
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# ------------------------------------------------------------
# 1. Basic page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Global Automotive Parts Pricing",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Global Automotive Parts Pricing Dashboard")
st.caption(
    "End-to-end ETL + analytics on automotive parts listings, "
    "with market insights and price recommendations."
)

# ------------------------------------------------------------
# 2. Data loading helper
# ------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_apps_data(csv_path: str = "apps_pricing_clean.csv") -> pd.DataFrame:
    """
    Load the final, cleaned applications dataset.

    Assumptions:
    - You exported a single CSV from your Jupyter notebook:
        apps.to_csv('apps_pricing_clean.csv', index=False)
    - The file contains at least the columns you showed:
        ['app_id', 'headline', 'price_gel', 'price_usd', 'app_register_date',
         'status_id', 'category_id', 'vehicle_type_id', 'seller_id',
         'item_condition', 'insert_date', 'status_id_dim', 'status_name',
         'category_id_dim', 'category_name', 'parent_category_id',
         'seller_id_dim', 'seller_name', 'address', 'mobile_number',
         'vehicle_type_id_dim', 'type_name', 'compatible_models',
         'compatible_brands', 'min_year', 'max_year', 'price_usd_filled']
    """

    path = Path(csv_path)
    if not path.exists():
        # If the CSV is missing, show a clear error and stop the app.
        st.error(
            f"❌ Could not find '{csv_path}'. "
            "Please export your final DataFrame from Jupyter as this file "
            "and place it in the same folder as this Streamlit app."
        )
        st.stop()

    df = pd.read_csv(path)

    # Parse date columns if present (this is safe even if some are missing)
    for col in ["app_register_date", "insert_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Make sure we have a consistent vehicle type column name: 'type_name'
    if "vehicle_type_name" in df.columns and "type_name" not in df.columns:
        df = df.rename(columns={"vehicle_type_name": "type_name"})

    # Ensure we have a 'price_usd_filled' column to work with
    if "price_usd_filled" not in df.columns:
        # Start with price_usd; fall back to median if missing
        if "price_usd" in df.columns:
            df["price_usd_filled"] = df["price_usd"].copy()
        else:
            st.error(
                "Dataset does not contain 'price_usd_filled' or 'price_usd'. "
                "Please add at least one of these columns in your notebook."
            )
            st.stop()

        median_price = df["price_usd_filled"].median()
        df["price_usd_filled"] = df["price_usd_filled"].fillna(median_price)

    # Drop obviously bad / non-positive prices
    df = df[df["price_usd_filled"] > 0]

    return df


apps = load_apps_data()

# ------------------------------------------------------------
# 3. Pre-compute market-level statistics
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_market_stats(apps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate market statistics by:
    - product category
    - vehicle type
    - item condition (New / Used / etc.)

    This fixes your KeyError by using 'type_name' instead of 'vehicle_type_name'.
    """

    # Columns we want to group by
    group_cols = ["category_name", "type_name", "item_condition"]

    # Only keep rows where all grouping keys are available
    df = apps_df.dropna(subset=group_cols + ["price_usd_filled"])

    market_stats = (
        df.groupby(group_cols)
        .agg(
            market_median_price=("price_usd_filled", "median"),
            market_mean_price=("price_usd_filled", "mean"),
            market_min_price=("price_usd_filled", "min"),
            market_max_price=("price_usd_filled", "max"),
            listing_count=("app_id", "count"),
        )
        .reset_index()
    )

    return market_stats


market_stats = compute_market_stats(apps)

# ------------------------------------------------------------
# 4. Sidebar – filters and price range
# ------------------------------------------------------------
st.sidebar.header("🔎 Filter Listings")

# Safe helpers to handle missing columns
def safe_unique(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().unique())

all_categories = safe_unique(apps, "category_name")
all_types = safe_unique(apps, "type_name")
all_conditions = safe_unique(apps, "item_condition")
all_statuses = safe_unique(apps, "status_name")

selected_categories = st.sidebar.multiselect(
    "Product category",
    options=all_categories,
    default=all_categories,
)

selected_types = st.sidebar.multiselect(
    "Vehicle type",
    options=all_types,
    default=all_types,
)

selected_conditions = st.sidebar.multiselect(
    "Item condition",
    options=all_conditions,
    default=all_conditions,
)

selected_statuses = st.sidebar.multiselect(
    "Application status",
    options=all_statuses,
    default=all_statuses,
)

# Price range filter
min_price = float(apps["price_usd_filled"].min())
max_price = float(apps["price_usd_filled"].max())
q05 = float(apps["price_usd_filled"].quantile(0.05))
q95 = float(apps["price_usd_filled"].quantile(0.95))

price_range = st.sidebar.slider(
    "Price range (USD)",
    min_value=round(min_price, 2),
    max_value=round(max_price, 2),
    value=(round(q05, 2), round(q95, 2)),
)

# ------------------------------------------------------------
# 5. Apply filters to the main dataset
# ------------------------------------------------------------
filtered = apps.copy()

if selected_categories:
    filtered = filtered[filtered["category_name"].isin(selected_categories)]
if selected_types:
    filtered = filtered[filtered["type_name"].isin(selected_types)]
if selected_conditions:
    filtered = filtered[filtered["item_condition"].isin(selected_conditions)]
if selected_statuses and "status_name" in filtered.columns:
    filtered = filtered[filtered["status_name"].isin(selected_statuses)]

filtered = filtered[
    (filtered["price_usd_filled"] >= price_range[0])
    & (filtered["price_usd_filled"] <= price_range[1])
]

st.sidebar.markdown("---")
st.sidebar.write(f"📊 Listings after filters: **{len(filtered):,}**")

# ------------------------------------------------------------
# 6. High-level KPIs
# ------------------------------------------------------------
st.subheader("📌 Market Overview (Filtered Data)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Listings", f"{len(filtered):,}")

with col2:
    median_price = filtered["price_usd_filled"].median()
    st.metric("Median Price (USD)", f"${median_price:,.2f}")

with col3:
    avg_price = filtered["price_usd_filled"].mean()
    st.metric("Average Price (USD)", f"${avg_price:,.2f}")

with col4:
    unique_sellers = filtered["seller_id"].nunique() if "seller_id" in filtered.columns else np.nan
    st.metric("Active Sellers", f"{unique_sellers:,}" if not np.isnan(unique_sellers) else "N/A")

st.markdown("---")

# ------------------------------------------------------------
# 7. Data preview
# ------------------------------------------------------------
with st.expander("👀 View sample of raw listings data"):
    st.write(
        "This is a quick peek at the raw rows that passed your filters. "
        "Useful to sanity-check the ETL pipeline and column meanings."
    )
    st.dataframe(filtered.head(50))

# ------------------------------------------------------------
# 8. Visual Analytics
# ------------------------------------------------------------
st.subheader("📈 Visual Pricing Insights")

tab1, tab2, tab3 = st.tabs(
    ["Price distribution", "Category comparison", "Category × Type heatmap"]
)

# --- Tab 1: Price distribution by condition ---
with tab1:
    st.write(
        "How are prices distributed for different item conditions "
        "(e.g., New vs Used) under the current filters?"
    )
    if len(filtered) == 0:
        st.warning("No data available for the current filter combination.")
    else:
        fig_box = px.box(
            filtered,
            x="item_condition",
            y="price_usd_filled",
            points="outliers",
            labels={
                "item_condition": "Item condition",
                "price_usd_filled": "Price (USD)",
            },
            title="Price distribution by item condition",
        )
        st.plotly_chart(fig_box, use_container_width=True)

# --- Tab 2: Median price by product category ---
with tab2:
    st.write(
        "Compare median prices across product categories "
        "to see which types of parts are more expensive."
    )
    if len(filtered) == 0:
        st.warning("No data available for the current filter combination.")
    else:
        cat_stats = (
            filtered.groupby("category_name", as_index=False)["price_usd_filled"]
            .median()
            .rename(columns={"price_usd_filled": "median_price_usd"})
        )

        fig_bar = px.bar(
            cat_stats.sort_values("median_price_usd", ascending=False),
            x="category_name",
            y="median_price_usd",
            title="Median price by product category",
            labels={"category_name": "Product category", "median_price_usd": "Median price (USD)"},
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

# --- Tab 3: Heatmap of mean price by category × type_name ---
with tab3:
    st.write(
        "Heatmap of mean prices by product category and vehicle type."
    )
    if len(filtered) == 0:
        st.warning("No data available for the current filter combination.")
    else:
        heat_df = (
            filtered.groupby(["category_name", "type_name"], as_index=False)["price_usd_filled"]
            .mean()
            .rename(columns={"price_usd_filled": "mean_price_usd"})
        )

        fig_heat = px.density_heatmap(
            heat_df,
            x="category_name",
            y="type_name",
            z="mean_price_usd",
            color_continuous_scale="Blues",
            title="Mean price heatmap (category × vehicle type)",
            labels={
                "category_name": "Product category",
                "type_name": "Vehicle type",
                "mean_price_usd": "Mean price (USD)",
            },
        )
        fig_heat.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# 9. Smart Price Recommendation
# ------------------------------------------------------------
st.subheader("🤖 Smart Price Recommendation Engine")

st.write(
    "Use the controls below to simulate a **new listing**. "
    "The model looks at historical listings with similar characteristics "
    "and suggests a competitive price range."
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    rec_category = st.selectbox(
        "Product category",
        options=all_categories,
        index=0 if all_categories else None,
    )

with col_b:
    rec_type = st.selectbox(
        "Vehicle type",
        options=all_types,
        index=0 if all_types else None,
    )

with col_c:
    rec_condition = st.selectbox(
        "Item condition",
        options=all_conditions,
        index=0 if all_conditions else None,
    )

col_d, col_e = st.columns(2)

with col_d:
    rec_min_year = st.number_input(
        "Compatible from year (optional)",
        min_value=1900,
        max_value=2100,
        value=2005,
        step=1,
    )

with col_e:
    rec_max_year = st.number_input(
        "Compatible up to year (optional)",
        min_value=1900,
        max_value=2100,
        value=2024,
        step=1,
    )

if st.button("💡 Recommend price"):
    # Filter to similar historical listings
    subset = apps.copy()
    subset = subset[
        (subset["category_name"] == rec_category)
        & (subset["type_name"] == rec_type)
        & (subset["item_condition"] == rec_condition)
    ]

    # Use year overlap if min_year / max_year exist
    if "min_year" in subset.columns and "max_year" in subset.columns:
        subset = subset[
            (subset["max_year"] >= rec_min_year)
            & (subset["min_year"] <= rec_max_year)
        ]

    count = len(subset)

    if count < 5:
        st.warning(
            f"Not enough similar listings to give a robust recommendation "
            f"(found only {count} matching rows). "
            "Try using broader conditions or check data quality."
        )
    else:
        p25 = subset["price_usd_filled"].quantile(0.25)
        p50 = subset["price_usd_filled"].quantile(0.50)
        p75 = subset["price_usd_filled"].quantile(0.75)
        p10 = subset["price_usd_filled"].quantile(0.10)
        p90 = subset["price_usd_filled"].quantile(0.90)

        st.success(
            f"Based on **{count:,}** similar listings, here is the suggested price range "
            f"for a **{rec_condition}** {rec_category} (vehicle type: {rec_type}):"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Conservative lower bound (P25)", f"${p25:,.2f}")
        with col2:
            st.metric("Recommended price (median)", f"${p50:,.2f}")
        with col3:
            st.metric("Aggressive upper bound (P75)", f"${p75:,.2f}")

        st.caption(
            f"For reference, 80% of similar listings fall between "
            f"${p10:,.2f} and ${p90:,.2f}."
        )

        with st.expander("See distribution of similar listings"):
            fig_rec = px.histogram(
                subset,
                x="price_usd_filled",
                nbins=30,
                title="Price distribution for similar listings",
                labels={"price_usd_filled": "Price (USD)"},
            )
            st.plotly_chart(fig_rec, use_container_width=True)

# ------------------------------------------------------------
# 10. Market stats table (from the grouped DataFrame)
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📚 Market Statistics by Category × Type × Condition")

st.write(
    "This table is built from the grouped `market_stats` DataFrame "
    "using the corrected column name `'type_name'`. "
    "You can download it for use in Power BI or further analysis."
)

st.dataframe(market_stats.head(100))

csv_export = market_stats.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download market_stats as CSV",
    data=csv_export,
    file_name="market_stats_by_category_type_condition.csv",
    mime="text/csv",
)
