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
```

## Treinamento do Modelo LSTM

O modelo LSTM (Long Short-Term Memory) é utilizado para prever o preço do último negócio de ações com base em dados históricos. O processo de treinamento inclui as seguintes etapas:

1. **Preparação dos Dados**:
   - Os dados são carregados de um arquivo CSV e processados para extrair as colunas relevantes: `data_pregao` e `preco_ultimo_negocio`.
   - As datas são convertidas em objetos de data do pandas e definidas como índice do DataFrame.

2. **Divisão do Conjunto de Dados**:
   - Os dados são divididos em conjuntos de treino e teste, com 80% dos dados utilizados para treino e 20% para teste.

3. **Criação de Sequências**:
   - A função `create_sequences` cria sequências de dados com um comprimento definido (neste caso, 30), preparando os dados para o modelo LSTM.

4. **Otimização de Hiperparâmetros**:
   - O Optuna é utilizado para otimizar os hiperparâmetros do modelo. Os parâmetros otimizados incluem o número de unidades LSTM, a taxa de aprendizado e o otimizador a ser utilizado (Adam ou RMSprop).

5. **Treinamento do Modelo**:
   - Após encontrar os melhores hiperparâmetros, o modelo LSTM é treinado por 100 épocas, utilizando o conjunto de treino.
   - O modelo é avaliado no conjunto de teste e métricas como MSE (Mean Squared Error), RMSE (Root Mean Squared Error) e MAPE (Mean Absolute Percentage Error) são calculadas.

6. **Previsão e Visualização**:
   - O modelo é utilizado para prever os preços nos últimos 30, 60 e 90 dias a partir dos dados de treino. As previsões são comparadas com os valores reais e visualizadas em gráficos.

### Resultados
Os resultados da previsão são apresentados em gráficos, comparando os valores reais com as previsões do modelo LSTM. Essas visualizações são essenciais para avaliar a performance do modelo e identificar áreas de melhoria.

### Métricas:
MSE: Mean Squared Error
RMSE: Root Mean Squared Error
MAPE: Mean Absolute Percentage Error
Essas métricas ajudam a entender a precisão das previsões e a ajustar o modelo conforme necessário.
