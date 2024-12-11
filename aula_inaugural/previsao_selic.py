import pandas as pd
import numpy as np
import statsmodels
import matplotlib.pyplot as plt

# Carregar os dados históricos
# data_url = "https://www.bcb.gov.br/controleinflacao/historicotaxasjuros"
df = pd.read_csv('./file.csv', sep=";")  # Este código assume que a tabela de dados está na primeira posição da lista retornada

# Ajustar os dados para serem usados no modelo ARIMA
df.set_index('numero', inplace=True)

# Visualizar os dados históricos
plt.figure(figsize=(10, 6))
plt.plot(df)
plt.title('Histórico da Taxa SELIC')
plt.xlabel('Data')
plt.ylabel('Taxa SELIC')
plt.grid(True)
plt.show()

# Dividir os dados em treino e teste
train_data = df.iloc[:-12]  # Usar tudo exceto os últimos 12 meses como dados de treino
test_data = df.iloc[-12:]   # Usar os últimos 12 meses como dados de teste

# Ajustar o modelo ARIMA
model = statsmodels.tsa.arima.model.ARIMA(train_data, order=(5,1,0))  # Aqui você pode ajustar a ordem do modelo ARIMA
fit_model = model.fit()

# Prever as próximas taxas SELIC
forecast = fit_model.forecast(steps=12)[0]

# Plotar as previsões
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data.values, label='Dados de Treino')
plt.plot(test_data.index, test_data.values, label='Dados de Teste')
plt.plot(test_data.index, forecast, label='Previsões')
plt.title('Previsões da Taxa SELIC')
plt.xlabel('Data')
plt.ylabel('Taxa SELIC')
plt.legend()
plt.grid(True)
plt.show()

print("Previsões para as próximas 12 meses:")
print(forecast)
