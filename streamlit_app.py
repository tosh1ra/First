import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing

st.title('Hello World!')

st.markdown('### AI')
@st.cashe_data

data = fetch_california_housing()
