import requests

import streamlit as st
import pandas as pd


API = "http://0.0.0.0:8000"
ALL_VISITS = "/api/visits/all"
STANDS_NAMES = "/api/stands/names"
STAND_DATES = (lambda stand_id: f"/api/stands/{stand_id}/dates")
STAND_DATE_RANGES = (lambda stand_id: f"/api/stands/{stand_id}/date_range")
STAND_STATS = (lambda stand_id: f"/api/stands/{stand_id}/stats")


class Shared:
    stands: list = []
    dates_from_stand: pd.Series = pd.Series()
    date_range: tuple = ()
    stats: dict = {}


@st.cache_data(ttl=60)
def get_data(api_val):
    try:
        response = requests.get(f"{API}{api_val}")
        response.raise_for_status()
        return response.json()
    except Exception as err:
        st.error(f"API error: {err}")
        return []


def get_stand_stats(stand_name, start_date, end_date):
    try:
        response = requests.get(
            API + STAND_STATS(stand_name),
            params={
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            }
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as err:
        st.error(f"Stats error: {err}")
        return {}


def stand_callback():
    if "stand_select" not in st.session_state:
        return
        
    selected_stand = st.session_state.stand_select
    print(f"Selected stand: {selected_stand}")
    
    with st.spinner("Loading dates..."):
        dates_data = get_data(STAND_DATES(selected_stand))
        Shared.dates_from_stand = pd.to_datetime([obj["timestamp"] for obj in dates_data])
        
        if len(Shared.dates_from_stand) > 0:
            Shared.date_range = (
                Shared.dates_from_stand.min().to_pydatetime(), 
                Shared.dates_from_stand.max().to_pydatetime()
            )
            st.session_state.date_slider = (
                Shared.dates_from_stand[0], 
                Shared.dates_from_stand[-1]
            )
            Shared.stats = get_stand_stats(selected_stand, Shared.date_range[0], Shared.date_range[1])


def date_callback():
    if "stand_select" not in st.session_state or "date_slider" not in st.session_state:
        return
        
    selected_stand = st.session_state.stand_select
    start_time = st.session_state.date_slider[1]
    end_time = st.session_state.date_slider[0]
    print(f"Date range: {start_time} to {end_time}")
    
    Shared.stats = get_stand_stats(selected_stand, start_time, end_time)


def main():
    st.set_page_config(page_title="Museum dashboard", layout="wide")

    if not Shared.stands:
        with st.spinner("Loading initial data..."):
            get_data(ALL_VISITS)
            stands_data = get_data(STANDS_NAMES)
            Shared.stands = [obj["name"] for obj in stands_data]
            
            if Shared.stands:
                first_stand = Shared.stands[0]
                dates_data = get_data(STAND_DATES(first_stand))
                Shared.dates_from_stand = pd.to_datetime([obj["timestamp"] for obj in dates_data])
                
                if len(Shared.dates_from_stand) > 0:
                    Shared.date_range = (
                        Shared.dates_from_stand.min().to_pydatetime(), 
                        Shared.dates_from_stand.max().to_pydatetime()
                    )
                    Shared.stats = get_stand_stats(first_stand, Shared.date_range[0], Shared.date_range[1])

    if not Shared.stands:
        st.error("Stands not found")
        st.stop()

    st.title("Museum stats")
    st.markdown("---")

    if "stand_select" not in st.session_state and Shared.stands:
        st.session_state.stand_select = Shared.stands[0]
    
    if ("date_slider" not in st.session_state or 
        len(Shared.dates_from_stand) == 0 or 
        st.session_state.date_slider[0] not in Shared.dates_from_stand):
        
        if len(Shared.dates_from_stand) > 1:
            st.session_state.date_slider = (
                Shared.dates_from_stand[0], 
                Shared.dates_from_stand[-1]
            )

    with st.sidebar:
        st.header("Menu")
        
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

        current_stand_index = Shared.stands.index(st.session_state.stand_select) if Shared.stands else 0
        st.selectbox(
            "Stand", 
            options=Shared.stands, 
            key="stand_select", 
            index=current_stand_index,
            on_change=stand_callback
        )

        if len(Shared.dates_from_stand) > 1:
            start_time, end_time = st.select_slider(
                "Date range",
                options=Shared.dates_from_stand,
                key="date_slider",
                value=st.session_state.date_slider,
                on_change=date_callback,
            )
        else:
            st.info("Недостаточно данных для выбора диапазона")

    st.header("Key statistics")
    cols = st.columns(5)
    
    stats = Shared.stats or {}
    default_stats = {
        "avg_age": 0, "avg_time_elapsed": 0, "total_visits": 0,
        "most_common_gender": "N/A", "most_common_age_group": "N/A"
    }
    stats = {**default_stats, **stats}
    
    with cols[0]:
        st.metric("Age", f"{stats['avg_age']:.1f} years")
    with cols[1]:
        st.metric("Time", f"{stats['avg_time_elapsed']:.1f} min")
    with cols[2]:
        st.metric("Visits", stats["total_visits"])
    with cols[3]:
        st.metric("Gender", stats["most_common_gender"])
    with cols[4]:
        st.metric("Age Group", stats["most_common_age_group"])

    all_visits = get_data(ALL_VISITS)
    if all_visits:
        st.dataframe(pd.DataFrame(all_visits))
    else:
        st.info("Нет данных о посещениях")


if __name__ == "__main__":
    main()

