import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()


class StockRequest(BaseModel):
    stock_name: str


STOCK_FILE_PATHS = {
    "TSLA": "../data/TSLA_data.csv",
    "AAPL": "../data/AAPL_data.csv",
    "AMZN": "../data/AMZN_data.csv",
    "MSFT": "../data/MSFT_data.csv",
}


@app.post("/LSTM_Predict")
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name
    try:
        file_path = STOCK_FILE_PATHS[stock_name]
        df = pd.read_csv(file_path)
    except KeyError:
        raise HTTPException(status_code=422, detail="Invalid stock name")

    data = df.filter(["Close"])

    dataset = data.values

    train_len = int(np.ceil(len(dataset) * 0.8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:train_len, :]

    seq_len = 60
    x_train, y_train = [], []

    for i in range(seq_len, len(train_data)):
        x_train.append(train_data[i - seq_len : i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"]
    )
    model.fit(x_train, y_train, batch_size=32, epochs=5)

    if train_len == 0:
        train_len = len(scaled_data) - len(train_data)

    test_data = scaled_data[train_len - seq_len :, :]
    x_test = []
    y_test = dataset[train_data:, :]

    for i in range(seq_len, len(test_data)):
        x_test.append(test_data[i - seq_len : i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    predict_prices = [price[0] for price in predictions.tolist()]

    return {"prediction": predict_prices}
