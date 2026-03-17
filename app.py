
import streamlit as st
import streamlit as st

st.write("🚨 THIS IS THE UPDATED FILE 🚨")
# UI-only mapping (does not affect ML model)
CATEGORY_MAP = {
    0: "Snacks",
    1: "Beverages",
    2: "Desserts",
    3: "Meals"
}
# Item price mapping (dummy but realistic)
PRICE_MAP = {
    "Snacks": 100,
    "Beverages": 80,
    "Desserts": 120,
    "Meals": 200
} 
import pandas as pd
import lightgbm as lgb
import os

import lightgbm as lgb

# Load trained model
ranker = lgb.Booster(model_file="csao_lgbm_ranker.txt")
st.set_page_config(
    page_title="CSAO Recommendation System",
    layout="wide"
)

st.title("🛒 Cart Super Add-On (CSAO) Recommendation System")

st.write(
    """
    This prototype demonstrates a cart-based recommendation system
    that dynamically suggests add-on items as the cart changes.
    """
)

# ---- Load dataset ----
DATA_PATH = "final_training_dataset.csv"
FEATURE_COLS = [
    "historical_orders",
    "is_cold_start",
    "dessert_affinity",
    "beverage_affinity",
    "budget_sensitivity",
    "hour",
    "weekend",
    "meal_slot_encoded",
    "cart_size",
    "cart_total_value",
    "budget_utilization",
    "remaining_budget",
    "step_number",
    "last_item_price",
    "item_price",
    "item_category",
    "dessert_affinity_x_item",
    "beverage_affinity_x_item",
    "price_gap_from_last"
]


MODEL_PATH = "csao_lgbm_ranker.txt"

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found.")
else:
    df = pd.read_csv(DATA_PATH)
    st.success("Dataset loaded successfully.")

# ---- Load model ----
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
else:
    model = lgb.Booster(model_file=MODEL_PATH)
    st.success("Model loaded successfully.")

# ---- Show basic info ----
if "df" in locals():
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
# ---------------- CART UI ----------------

st.subheader("🛍️ Build Your Cart")

# Initialize cart in session state
if "cart_items" not in st.session_state:
    st.session_state.cart_items = []

# Create a mock menu from dataset
menu_df = df[["item_category", "item_price"]].drop_duplicates().head(20)

# Available item categories (from dataset)
options = sorted(df["item_category"].unique())
# Select item
selected_category = st.selectbox(
    "Select an item to add to cart:",
    options,
    format_func=lambda x: CATEGORY_MAP.get(x, f"Item {x}")
)

category_name = CATEGORY_MAP[selected_category]
selected_price = PRICE_MAP[category_name]

# Add button
if st.button("Add to Cart"):
    st.session_state.cart_items.append({
        "item_category": selected_category,   # numeric → model
        "item_price": selected_price           # numeric → price
    })
    st.success(f"{category_name} added to cart!")

# Display cart
st.subheader("🛒 Current Cart")

if st.session_state.cart_items:
    cart_df = pd.DataFrame(st.session_state.cart_items)
    st.dataframe(cart_df)
    st.write(f"**Cart Size:** {len(cart_df)}")
    st.write(f"**Total Value:** ₹{cart_df['item_price'].sum():.2f}")
else:
    st.info("Your cart is empty.")

# =============================
# 🔮 Recommendations Section
# =============================

st.markdown("## ⭐ Recommended Add-Ons")

if len(st.session_state.cart_items) > 0:
    last_item = st.session_state.cart_items[-1]

    # Build candidate items (sample from dataset)
    candidate_df = df.sample(30, random_state=42).copy()

    # Update dynamic cart features
    candidate_df["cart_size"] = len(st.session_state.cart_items)
    candidate_df["cart_total_value"] = sum(
        item["item_price"] for item in st.session_state.cart_items
    )
    candidate_df["last_item_category"] = last_item["item_category"]
    candidate_df["last_item_price"] = last_item["item_price"]

    # Drop label if present
    if "label" in candidate_df.columns:
        candidate_df = candidate_df.drop(columns=["label"])

    # Predict relevance score
    candidate_df["score"] = ranker.predict(candidate_df[FEATURE_COLS])

    # Top-5 recommendations
    top_recos = candidate_df.sort_values("score", ascending=False).head(5)

    for _, row in top_recos.iterrows():
        st.write(
            f"🍽️ Category: {CATEGORY_MAP.get(row['item_category'], 'Item')} "
            f"| Price: ₹{row['item_price']:.2f}"
        )
else:
    st.info("Add items to cart to see recommendations.")
# ---- Force realistic context defaults ----
candidate_df["hour"] = 20                 # dinner time
candidate_df["weekend"] = 0
candidate_df["meal_slot_encoded"] = 2     # dinner
candidate_df["step_number"] = len(st.session_state.cart_items)

# Budget logic
candidate_df["budget_utilization"] = (
    candidate_df["cart_total_value"] / 500
)
candidate_df["remaining_budget"] = 500 - candidate_df["cart_total_value"]