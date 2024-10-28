import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.dates as mdates


from keras.models import load_model
from sklearn.model_selection import train_test_split
from datetime import date
from plotly import graph_objs as go
from matplotlib import pyplot as plt

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("EconoVision")

stocks = ("POSI3.SA", "TOTS3.SA", "INTB3.SA", "DOTZ3.SA", "LWSA3.SA")

selected_stock = st.selectbox("Selecione o Dataset para a previsão", stocks)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

tf.compat.v1.reset_default_graph()
model = load_model(f"../model/{selected_stock}_model.keras")

data_load_state = st.text("Carregando dados...")
data = load_data(selected_stock)
data_load_state.text("Dados carregados com sucesso!")

st.subheader('DADO BRUTO')
st.write(data.tail())

sample_data = data.to_string()

def calcular_rsi(data, janela=14):
    delta = data['Close'].diff(1)
    ganho = delta.clip(lower=0).rolling(window=janela).mean()
    perda = -delta.clip(upper=0).rolling(window=janela).mean()
    rsi = 100 - (100 / (1 + ganho / perda))
    return rsi.iloc[-1]

def calcular_variacao(data):
    # Cálculo da variação percentual
    variacao_diaria = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
    variacao_semanal = (data['Close'].iloc[-1] - data['Close'].iloc[-8]) / data['Close'].iloc[-8] * 100
    variacao_mensal = (data['Close'].iloc[-1] - data['Close'].iloc[-22]) / data['Close'].iloc[-22] * 100

    return variacao_diaria, variacao_semanal, variacao_mensal

def interpretar_dados(data):
    preco_inicial = data['Close'].iloc[0]
    preco_final = data['Close'].iloc[-1]
    variacao_total = (preco_final - preco_inicial) / preco_inicial * 100

    volatilidade = data['Close'].std()
    data['media_movel_7d'] = data['Close'].rolling(window=7).mean()
    media_movel_final = data['media_movel_7d'].iloc[-1]

    preco_max = data['Close'].max()
    preco_min = data['Close'].min()
    data_max = data.loc[data['Close'] == preco_max, 'Date'].values[0]
    data_min = data.loc[data['Close'] == preco_min, 'Date'].values[0]

    rsi_final = calcular_rsi(data)
    variacao_diaria, variacao_semanal, variacao_mensal = calcular_variacao(data)

    if rsi_final > 70:
        comentario_rsi = f"O RSI atual é {rsi_final:.2f}. A ação pode estar sobrecomprada."
    elif rsi_final < 30:
        comentario_rsi = f"O RSI atual é {rsi_final:.2f}. A ação pode estar sobrevendida."
    else:
        comentario_rsi = f"O RSI atual é {rsi_final:.2f}. A ação está em uma zona neutra."

    if variacao_total > 0:
        tendencia = f"A ação valorizou {variacao_total:.2f}% no período analisado."
        resumo_tendencia = "A ação está em alta, o que é um bom sinal."
    elif variacao_total < 0:
        tendencia = f"A ação desvalorizou {abs(variacao_total):.2f}% no período analisado."
        resumo_tendencia = "A ação está em queda, o que pode ser um alerta."
    else:
        tendencia = "O preço da ação permaneceu estável no período analisado."
        resumo_tendencia = "A ação manteve seu valor estável."

    if volatilidade > 2:
        comentario_volatilidade = "O ativo apresentou alta volatilidade."
        resumo_volatilidade = "O preço tem oscilado bastante, o que aumenta o risco."
    else:
        comentario_volatilidade = "O ativo teve comportamento estável."
        resumo_volatilidade = "O preço está estável, o que indica menor risco."

    if media_movel_final > preco_final:
        comentario_media = "A média móvel sugere uma tendência de queda no curto prazo."
        resumo_media = "Pode haver uma leve queda nos próximos dias."
    else:
        comentario_media = "A média móvel sugere uma tendência de alta no curto prazo."
        resumo_media = "Pode haver um crescimento nos próximos dias."

    comentario_picos = (
        f"O preço máximo foi R$ {preco_max:.2f} em {data_max}, "
        f"e o mínimo foi R$ {preco_min:.2f} em {data_min}."
    )

    if variacao_total > 10 and volatilidade < 2:
        recomendacao = "A tendência é positiva e o risco é baixo. Pode ser um bom momento para comprar."
    elif variacao_total < -10:
        recomendacao = "A ação está em queda acentuada. Avalie se é melhor aguardar antes de comprar."
    else:
        recomendacao = "O mercado está oscilando. Pode ser prudente esperar mais sinais."

    analise_completa = (
        f"{tendencia}\n"
        f"{comentario_volatilidade}\n"
        f"{comentario_picos}\n"
        f"{comentario_media}\n"
        f"{comentario_rsi}\n"
        f"Recomendação: {recomendacao}\n"
        f"\nVariação diária: {variacao_diaria:.2f}%\n"
        f"Variação semanal: {variacao_semanal:.2f}%\n"
        f"Variação mensal: {variacao_mensal:.2f}%"
    )

    resumo_simplificado = (
        f"Resumo para você:\n"
        f"- {resumo_tendencia}\n"
        f"- {resumo_volatilidade}\n"
        f"- {resumo_media}\n"
        f"- O preço variou entre R$ {preco_min:.2f} e R$ {preco_max:.2f}.\n"
        f"- RSI: {rsi_final:.2f}\n"
        f"- {recomendacao}\n"
        f"- Variação diária: {variacao_diaria:.2f}%\n"
        f"- Variação semanal: {variacao_semanal:.2f}%\n"
        f"- Variação mensal: {variacao_mensal:.2f}%"
    )

    return analise_completa, resumo_simplificado

st.subheader("Análise dos dados")
if st.button("Gerar Análise"):
    analise, resumo = interpretar_dados(data)
    st.subheader("Análise Completa:")
    st.text(analise)

    st.subheader("Resumo Simplificado:")
    st.text(resumo)

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

# Gráfico: Preço Histórico + Previsões
fig1, ax = plt.subplots(figsize=(12, 4))

# Plotando dados históricos e previsões
ax.plot(pd.to_datetime(data.index), data['Close'], label='Dados Históricos (2024)', color='blue')
ax.plot(future_df['Date'], future_df['LSTM Prediction'], label='Previsões Futuras', color='red')

# Configurações do eixo X para datas corretamente formatadas
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig1.autofmt_xdate()

# Definindo limites e rótulos
ax.set_xlim(pd.Timestamp('2024-01-01'), future_df['Date'].max())
ax.set_title('Previsão dos Próximos Valores com Datas')
ax.set_xlabel('Data')
ax.set_ylabel('Preço Último Negócio')

# Linha vertical para última data de treino
ax.axvline(x=last_date, color='green', linestyle='--', label='Última Data do Treino')

# Exibir a legenda e ajustar layout
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Exibindo o gráfico no Streamlit
st.pyplot(fig1)