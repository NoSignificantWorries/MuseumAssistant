import streamlit as st

st.title('📊 Дашборд метрик выставочных стендов')

st.sidebar.header('Фильтры')
selected_stand = st.sidebar.selectbox('Выберите стенд', ["id: 1", "id: 2", "id: 3"])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Кол-во посещений", 35, delta="+155%")
with col2:
    st.metric("Среднее время удержания", "5.2 мин.")
with col3:
    st.metric("Средний возраст", "25 лет")

