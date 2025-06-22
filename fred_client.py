from fredapi import Fred
import streamlit as st

fred = Fred(api_key=st.secrets["FRED_API_KEY"])