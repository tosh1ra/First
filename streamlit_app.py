import streamlit as st  # импорт библиотеки Streamlit для создания веб-приложения
import pandas as pd  # импорт pandas для работы с таблицами
import numpy as np  # импорт numpy для численных операций
import matplotlib.pyplot as plt  # импорт matplotlib для построения графиков
from sklearn.datasets import fetch_california_housing  # загрузка датасета недвижимости Калифорнии
from sklearn.ensemble import RandomForestRegressor  # модель случайного леса для регрессии

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="🏠 House Price AI", layout="wide")  # настройка страницы приложения

st.title("🏠 AI House Price Predictor")  # заголовок приложения
st.markdown("### Smart real estate valuation powered by Machine Learning")  # подзаголовок

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data  # кэширование данных для ускорения
def load_data():
    housing = fetch_california_housing()  # загрузка датасета
    X = pd.DataFrame(housing.data, columns=housing.feature_names)  # признаки в DataFrame
    y = pd.Series(housing.target, name="PRICE")  # целевая переменная (цена)
    return X, y  # возврат данных

X, y = load_data()  # получение данных

# ---------------------------
# MODEL
# ---------------------------
@st.cache_resource  # кэширование модели
def train_model():
    model = RandomForestRegressor(n_estimators=150)  # создание модели с 150 деревьями
    model.fit(X, y)  # обучение модели
    return model  # возврат модели

model = train_model()  # обучение модели

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("⚙️ Customize House")  # заголовок боковой панели

def user_input():
    data = {}  # словарь для хранения пользовательских данных
    for col in X.columns:  # цикл по всем признакам
        data[col] = st.sidebar.slider(  # создание слайдера для каждого признака
            col,  # название признака
            float(X[col].min()),  # минимальное значение
            float(X[col].max()),  # максимальное значение
            float(X[col].mean())  # значение по умолчанию
        )
    return pd.DataFrame(data, index=[0])  # преобразование в DataFrame

df = user_input()  # получение пользовательского ввода

prediction = model.predict(df)[0]  # предсказание модели
price = prediction * 100000  # масштабирование цены

# ---------------------------
# TOP METRICS
# ---------------------------
col1, col2, col3 = st.columns(3)  # создание 3 колонок

col1.metric("💰 Predicted Price", f"${price:,.0f}")  # отображение предсказанной цены
col2.metric("📊 Dataset Avg", f"${y.mean()*100000:,.0f}")  # средняя цена по датасету
col3.metric("📈 Difference", f"${price - y.mean()*100000:,.0f}")  # разница

st.divider()  # разделительная линия

# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📥 Input", "📊 Analytics", "🗺️ Map", "🧠 Model"])  # создание вкладок

# ---------------------------
# TAB 1 INPUT
# ---------------------------
with tab1:
    st.subheader("Your House Parameters")  # заголовок
    st.dataframe(df)  # отображение введённых данных

# ---------------------------
# TAB 2 ANALYTICS
# ---------------------------
with tab2:
    colA, colB = st.columns(2)  # две колонки

    with colA:
        st.subheader("Price Distribution")  # заголовок графика
        fig, ax = plt.subplots()  # создание фигуры
        ax.hist(y, bins=30)  # гистограмма цен
        ax.axvline(prediction, linestyle="--")  # линия предсказания
        st.pyplot(fig)  # вывод графика

    with colB:
        st.subheader("Feature Importance")  # заголовок

        importance = pd.DataFrame({  # создание DataFrame важности признаков
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)  # сортировка

        fig2, ax2 = plt.subplots()  # создание фигуры
        ax2.barh(importance["Feature"], importance["Importance"])  # горизонтальный барчарт
        st.pyplot(fig2)  # вывод графика

# ---------------------------
# TAB 3 MAP
# ---------------------------
with tab3:
    st.subheader("Geographic Visualization")  # заголовок

    map_data = X.copy()  # копия данных
    map_data["price"] = y  # добавление цены

    st.map(map_data.rename(columns={  # отображение карты
        "Latitude": "lat",  # переименование широты
        "Longitude": "lon"  # переименование долготы
    }))

# ---------------------------
# TAB 4 MODEL INFO
# ---------------------------
with tab4:
    st.subheader("Model Explanation")  # заголовок

    st.write("""  # текстовое описание модели
    This model uses **Random Forest Regression**.

    ✔ Combines multiple decision trees  
    ✔ Captures complex relationships  
    ✔ Works well for real estate data  

    ### Key Factors:
    - Median Income (most important)
    - Location (Latitude/Longitude)
    - Average Rooms
    """)

    st.subheader("Correlation Matrix")  # заголовок корреляции

    corr = X.corr()  # вычисление корреляционной матрицы
    fig3, ax3 = plt.subplots()  # создание фигуры
    cax = ax3.matshow(corr)  # визуализация матрицы
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)  # подписи оси X
    plt.yticks(range(len(corr.columns)), corr.columns)  # подписи оси Y
    fig3.colorbar(cax)  # цветовая шкала

    st.pyplot(fig3)  # вывод графика

# ---------------------------
# BONUS: COMPARE WITH DATASET
# ---------------------------
st.divider()  # разделитель
st.subheader("📊 Compare Your House to Dataset")  # заголовок

comparison = pd.DataFrame({  # создание DataFrame сравнения
    "Your House": df.iloc[0],  # данные пользователя
    "Average": X.mean()  # средние значения
})

st.bar_chart(comparison)  # отображение столбчатой диаграммы
