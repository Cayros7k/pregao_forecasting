# Aplicativo de Previsão de Cotação

Este é um aplicativo desenvolvido com Streamlit para prever cotações de ações utilizando um modelo LSTM (Long Short-Term Memory). O aplicativo permite a seleção de diferentes ações e fornece análises detalhadas sobre a performance das ações selecionadas.

## Funcionalidades

- **Seleção de Ações**: O usuário pode escolher entre diferentes ações ("POSI3.SA", "TOTS3.SA", "INTB3.SA", "DOTZ3.SA", "LWSA3.SA").
- **Carregamento de Dados**: Os dados são carregados usando a biblioteca `yfinance`, com um intervalo definido de 2018 até a data atual.
- **Cálculo do RSI**: O aplicativo calcula o Índice de Força Relativa (RSI) para avaliar condições de sobrecompra ou sobrevenda.
- **Variação Percentual**: Exibe a variação percentual diária, semanal e mensal das ações.
- **Análise Completa e Resumida**: Fornece uma análise detalhada da performance da ação, incluindo tendência, volatilidade, média móvel, e recomendações.
- **Previsão de Preços Futuros**: Utiliza um modelo LSTM para prever os preços futuros da ação selecionada.
- **Visualização**: Plota gráficos de preços históricos e previsões futuras.

## Tecnologias Utilizadas

- Python
- Streamlit
- TensorFlow/Keras
- yfinance
- pandas
- NumPy
- Plotly
- Matplotlib

## Requisitos

Para executar este projeto, você precisará das seguintes bibliotecas Python:

```bash
pip install streamlit yfinance pandas numpy tensorflow plotly matplotlib
