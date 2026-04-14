import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import shap

# ---------------------------
# DARK THEME (custom CSS)
# ---------------------------
st.set_page_config(page_title="🏠 House Price AI", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 AI House Price Predictor")
st.markdown("### Smart real estate valuation powered by Machine Learning")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

X, y = load_data()

# ---------------------------
# MODEL
# ---------------------------
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=150)
    model.fit(X, y)
    return model

model = train_model()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("⚙️ Customize House")

def user_input():
    data = {}
    for col in X.columns:
        avg = float(X[col].mean())

        data[col] = st.sidebar.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            avg
        )

        st.sidebar.caption(f"Avg: {avg:.2f}")  # <-- добавили среднее

    return pd.DataFrame(data, index=[0])

df = user_input()

prediction = model.predict(df)[0]
price = prediction * 100000

# ---------------------------
# METRICS
# ---------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Predicted Price", f"${price:,.0f}")
col2.metric("📊 Dataset Avg", f"${y.mean()*100000:,.0f}")
col3.metric("📈 Difference", f"${price - y.mean()*100000:,.0f}")

st.divider()

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📥 Input", "📊 Analytics", "🗺️ Map", "🧠 Model"])

# ---------------------------
# TAB 1
# ---------------------------
with tab1:
    st.subheader("Your House Parameters")
    st.dataframe(df.style.background_gradient(cmap="Blues"))

# ---------------------------
# TAB 2
# ---------------------------
with tab2:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        ax.hist(y, bins=30)
        ax.axvline(prediction, linestyle="--")
        st.pyplot(fig)

    with colB:
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig2, ax2 = plt.subplots()
        ax2.barh(importance["Feature"], importance["Importance"])
        st.pyplot(fig2)

    # ---------------------------
    # SHAP
    # ---------------------------
    st.subheader("🔍 Why this price?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    fig_shap = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        df,
        matplotlib=True
    )

    st.pyplot(fig_shap)

# ---------------------------
# TAB 3
# ---------------------------
with tab3:
    st.subheader("Geographic Visualization")

    map_data = X.copy()
    map_data["price"] = y

    st.map(map_data.rename(columns={
        "Latitude": "lat",
        "Longitude": "lon"
    }))

# ---------------------------
# TAB 4
# ---------------------------
with tab4:
    st.subheader("Model Explanation")

    st.write("""
    This model uses **Random Forest Regression**.

    ✔ Combines multiple decision trees  
    ✔ Captures complex relationships  
    ✔ Works well for real estate data  
    """)

    st.subheader("Correlation Matrix")

    corr = X.corr()
    fig3, ax3 = plt.subplots()
    cax = ax3.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig3.colorbar(cax)

    st.pyplot(fig3)

# ---------------------------
# BEAUTIFUL COMPARISON
# ---------------------------
st.divider()
st.subheader("📊 Compare Your House to Dataset")

comparison = pd.DataFrame({
    "Your House": df.iloc[0],
    "Average": X.mean()
})

# нормализация для красоты
comparison_norm = comparison / comparison.max()

st.line_chart(comparison_norm.T)
st.dataframe(comparison.style.background_gradient(cmap="coolwarm"))
