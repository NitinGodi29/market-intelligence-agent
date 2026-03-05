import streamlit as st
import time
import requests

container1, container2, container3 = st.container(), st.container(), st.container()
with container1:
    st.set_page_config(page_title="Market Intelligence Agent", layout="wide")
    st.title("Market Intelligence Agent")
    st.markdown(
        """
    Welcome to the Market Intelligence Agent! \n\nThis application provides insights and analysis on stock market trends based on ticker information.
    """
    )


def trigger_analysis(ticker):
    with container3:
        with st.spinner("Analyzing ticker..."):
            url = f"http://localhost:8009/analyze/{ticker}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                st.subheader(f"Analysis for {data['ticker']}")
                st.markdown(f"**Executive Summary:** {data['executive_summary']}")
                st.markdown(f"**Recommendation:** {data['recommendation']}")
                st.markdown(f"**Confidence Score:** {data['confidence_score']}")
                st.markdown(f"**Data status:** {data['data_status']}")
                st.markdown("**Ticker Details:**")
                st.dataframe(data["details"])
            else:
                st.error(f"Error fetching analysis")

with container2:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        selected_ticker = st.selectbox(
            "Select a stock ticker",
            options=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
            placeholder="Choose a ticker",
            accept_new_options=True,
            index=None
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button('Fetch Details', on_click=trigger_analysis, args=(selected_ticker,))
