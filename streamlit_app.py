import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# Настройка страницы
st.set_page_config(page_title="Pro House AI", layout="wide")

# ---------------------------
# ЗАГРУЗКА И МОДЕЛЬ (Максимальное ускорение)
# ---------------------------
@st.cache_data
def get_data():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="PRICE")
    return X, y

@st.cache_resource
def train_fast_model(_X, _y):
    # n_jobs=-1 использует все ядра процессора для мгновенного обучения
    model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(_X, _y)
    return model

X, y = get_data()
model = train_fast_model(X, y)

# ---------------------------
# БОКОВАЯ ПАНЕЛЬ
# ---------------------------
st.sidebar.header("⚙️ Параметры дома")
def get_user_input():
    inputs = {}
    stats = X.describe()
    for col in X.columns:
        inputs[col] = st.sidebar.slider(col, float(stats.loc['min', col]), 
                                        float(stats.loc['max', col]), float(stats.loc['mean', col]))
    return pd.DataFrame(inputs, index=[0])

user_df = get_user_input()
pred_raw = model.predict(user_df)[0]
price = pred_raw * 100000
avg_price = y.mean() * 100000

# ---------------------------
# ГЛАВНЫЙ ИНТЕРФЕЙС
# ---------------------------
st.title("🏠 AI Real Estate Pro")

# 1. СПИДОМЕТР (Gauge Chart) вместо скучных цифр
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = price,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Прогноз стоимости ($)"},
    delta = {'reference': avg_price, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge = {
        'axis': {'range': [None, y.max()*100000]},
        'bar': {'color': "#1f77b4"},
        'steps': [
            {'range': [0, avg_price], 'color': "lightgray"},
            {'range': [avg_price, y.max()*100000], 'color': "gray"}],
        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': price}
    }
))
fig_gauge.update_layout(height=350, margin=dict(t=0, b=0))
st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["📍 Карта цен", "🔍 Анализ факторов", "📊 Сравнение"])

# 2. ИНТЕРАКТИВНАЯ КАРТА (Цвет = Цена)
with tab1:
    st.subheader("Где находятся дорогие дома?")
    map_df = X.sample(1500).copy()
    map_df['Price'] = y[map_df.index] * 100000
    # Используем plotly для цветных точек
    fig_map = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude", color="Price",
                                size="Price", color_continuous_scale=px.colors.cyclical.IceFire, 
                                size_max=10, zoom=5, mapbox_style="carto-positron")
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# 3. ОБЪЯСНЕНИЕ ЦЕНЫ (Почему такая цена?)
with tab2:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.subheader("Логика ИИ")
        # Простая имитация SHAP: объясняем влияние главного фактора
        main_feat = "MedInc" # Доход населения - главный фактор в этом датасете
        val = user_df[main_feat][0]
        if val > X[main_feat].mean():
            st.success(f"✅ Цена выше средней, так как доход в районе ({val:.2f}) выше нормы.")
        else:
            st.warning(f"⚠️ Цена ниже, так как показатель дохода ({val:.2f}) невысок.")
    
    with col_b:
        st.subheader("Важность параметров")
        feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance')
        st.bar_chart(feat_imp.set_index('Feature'))

# 4. БЫСТРОЕ СРАВНЕНИЕ
with tab3:
    st.subheader("Ваш выбор vs Среднее по рынку")
    comparison = pd.concat([user_df.T, X.mean().to_frame()], axis=1)
    comparison.columns = ['Ваш дом', 'Среднее']
    st.table(comparison.style.highlight_max(axis=1, color='lightgreen'))

st.caption("Данные: California Housing Dataset. Обучено мгновенно с использованием RandomForest(n_jobs=-1).")
