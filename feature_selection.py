import streamlit as st
from sklearn.datasets import load_diabetes
import altair as alt
from sklearn.feature_selection import GenericUnivariateSelect, r_regression, f_regression, mutual_info_regression
import pandas as pd

X = load_diabetes(as_frame=True)
score_func = st.sidebar.selectbox("Score function",
                     ['r_regression', 'f_regression', 'mutual_info_regression'])
mode = st.sidebar.selectbox("Mode", ['k_best', 'percentile'])
if mode == 'k_best':
    param = st.sidebar.slider("Number of features", 1, 10, 5)
if mode == 'percentile':
    param = st.sidebar.slider("Percentile", 1, 100, 50)
if mode in ['fpr', 'fdr', 'fwe']:
    param = st.sidebar.slider("Alpha", 0.01, 0.1, 0.05)
selector = GenericUnivariateSelect(score_func=eval(score_func), mode=mode, param=param)
selector.fit(X.data, X.target)
df = pd.DataFrame({'feature': X.data.columns, 'score': selector.scores_})
chart = alt.Chart(df).mark_bar().encode(
    alt.X('feature', sort='-y'),
    alt.Y('score')
)
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(chart)
with col2:
    st.write("Selected features:", selector.get_feature_names_out())