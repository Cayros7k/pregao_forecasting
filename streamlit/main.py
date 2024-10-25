import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.models import load_model
from sklearn.model_selection import train_test_split
from datetime import date
from plotly import graph_objs as go
from matplotlib import pyplot as plt

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Aplicativo de Previsão de Cotação")

stocks = ("POSI3.SA", "TOTS3.SA", "INTB3.SA", "DOTZ3.SA", "LWSA3.SA")

selected_stock = st.selectbox("Selecione o Dataset para a previsão", stocks)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)  # Mantém apenas o primeiro nível

    return data

tf.compat.v1.reset_default_graph()
model = load_model(f"../model/{selected_stock}_model.keras")

data_load_state = st.text("Carregando dados...")
data = load_data(selected_stock)
data_load_state.text("Dados carregados com sucesso!")

st.subheader('DADO BRUTO')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="SÉRIE TEMPORAL DOS DADOS", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

n_days = st.slider("Meses para prever:", 1, 3)
period = n_days * 30

df = data.set_index('Date', inplace=True)
values = data['Close'].values

train_values, test_values = train_test_split(values, test_size=0.2, shuffle=False)

def create_sequences(df, sequence_length):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[i:i + sequence_length])
        y.append(df[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 30
X_train, y_train = create_sequences(train_values, sequence_length)
X_test, y_test = create_sequences(test_values, sequence_length)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

last_sequence = train_values[-sequence_length:]
future_predictions = []

current_sequence = last_sequence.reshape((1, sequence_length, 1))

for _ in range(period):
    next_prediction = model.predict(current_sequence)[0]
    future_predictions.append(next_prediction)
    current_sequence = np.append(current_sequence[:, 1:, :], [[next_prediction]], axis=1)

future_predictions = np.array(future_predictions).flatten()

last_date = data.index[-1]
last_date = pd.to_datetime(last_date)
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=period, freq='B')

future_df = pd.DataFrame({'Date': future_dates, 'LSTM Prediction': future_predictions})

fig1 = plt.figure(figsize=(12, 4))

plt.plot(data.index, data['Close'], label='Dados Históricos (2024)', color='blue')
plt.plot(future_df['Date'], future_df['LSTM Prediction'], label='Previsões Futuras', color='red')

plt.xlim(pd.Timestamp('2024-01-01'), future_df['Date'].max())

plt.title('Previsão dos Próximos Valores com Datas')
plt.xlabel('Data')
plt.ylabel('Preço Último Negócio')
plt.axvline(x=last_date, color='green', linestyle='--', label='Última Data do Treino')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)