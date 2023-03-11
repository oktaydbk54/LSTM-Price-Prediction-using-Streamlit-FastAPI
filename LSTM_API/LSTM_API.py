import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class StockRequest(BaseModel):
    stock_name: str

STOCK_FILE_PATHS = {
    'TSLA': '../data/TSLA_data.csv',
    'AAPL': '../data/AAPL_data.csv',
    'AMZN': '../data/AMZN_data.csv',
    'MSFT': '../data/MSFT_data.csv'
}

@app.post('/LSTM_Predict')
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name
    try:
        file_path = STOCK_FILE_PATHS[stock_name]
        df = pd.read_csv(file_path)
    except KeyError:
        raise HTTPException(status_code=422, detail='Invalid stock name')

    data = df.filter(['Close'])

    dataset = data.values

    training_data_len = int(np.ceil(len(dataset) * .80))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            pass

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    history = model.fit(x_train, y_train, batch_size=32, epochs=5)

    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    predict_price = list()

    for price in predictions.tolist():
        predict_price.append(price[0])

    return {'prediction': predict_price}
