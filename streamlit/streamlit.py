import streamlit as st
import yfinance as yf
import datetime
import plotly.graph_objs as go
import requests
import json

API_URL = "http://127.0.0.1:8000/LSTM_Predict"

min_date = datetime.date(2020, 1, 1)
max_date = datetime.date(2022, 12, 31)

stock_name = st.selectbox('Please choose stock name', ('AAPL','TSLA','AMZN','MSFT'))

start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

if start_date <= end_date:
    st.success("Start date: `{}`\n\nEnd date:`{}`".format(start_date, end_date))
else:
    st.error("Error: End date must be after start date.")

stock_data = yf.download(stock_name, start=start_date, end=end_date)
stock_data.reset_index(inplace=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close'))
fig.update_layout(title=f"{stock_name} Stock Price")
st.plotly_chart(fig)

stock_data.to_csv(f'{stock_name}_data.csv',index=False)


if st.button("Predict"):
    payload = {"stock_name": stock_name}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        predictions = response.json()
        predicted_prices = predictions["prediction"]

        actual_prices = stock_data['Close'].tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=actual_prices, name='Actual'))
        fig.add_trace(go.Scatter(x=stock_data.index[-len(predicted_prices):], y=predicted_prices, name='Predicted'))
        fig.update_layout(title=f"{stock_name} Stock Price")
        st.plotly_chart(fig)

    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred while making the request: {e}")


