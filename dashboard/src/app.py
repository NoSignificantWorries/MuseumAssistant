import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Museum dashboard",
                       page_icon="",
                       layout="wide")

    st.title("Museum stats")
    st.markdown("---")

    st.sidebar.header("󰍜 Menu")
    st.sidebar.selectbox(label="Stand", options=["1", "2", "3"])


    st.header("Key statistics", anchor="center")
    mean_edge_clmn, mean_time_clmn, count_of_visits_clmn = st.columns(3)

    with mean_edge_clmn:
        st.metric(label="Edge", value="30.2 years", delta="-10%", border=True)

    with mean_time_clmn:
        st.metric(label="Time", value="6.4 min", delta="+0.5%", border=True)

    with count_of_visits_clmn:
        st.metric(label="Visits", value="142", delta="+32", border=True)



if __name__ == "__main__":
    main()

