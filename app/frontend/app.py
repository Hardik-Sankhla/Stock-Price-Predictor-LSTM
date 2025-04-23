# app/frontend/app.py
import streamlit as st
import requests

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

st.title("📈 Stock Price Predictor (LSTM)")
st.markdown("Enter a stock ticker to see the predicted vs actual closing price.")

# Input for ticker
ticker = st.text_input("Stock Ticker Symbol", value="AAPL")

# API request
if st.button("Predict"):
    with st.spinner("Predicting..."):
        try:
            response = requests.get("http://localhost:8000/predict", params={"ticker": ticker})
            if response.status_code == 200:
                data = response.json()
                predicted = round(data['predicted_price'], 2)
                actual = round(data['last_actual_price'], 2)

                st.success(f"✅ Predicted Price: **${predicted}**")
                st.info(f"ℹ️ Last Actual Price: **${actual}**")

                # Line chart
                st.line_chart({
                    "Predicted Price": [None] * 58 + [actual, predicted],
                    "Actual Price": [None] * 59 + [actual]
                })

            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"❌ Failed to connect to API: {str(e)}")
