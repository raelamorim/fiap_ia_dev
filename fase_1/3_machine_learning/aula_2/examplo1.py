import statsmodels.api as sm
import pandas as pd

# Criar um DataFrame com dados fictícios
data = {'Vendas_Sorvetes': [200, 300, 400, 350, 500],
        'Temperatura': [28, 30, 32, 29, 33],
        'Promocao_Marketing': [1000, 1200, 800, 900, 1100]}

df = pd.DataFrame(data)

# Adicionar uma constante para o termo de intercepto
df['Intercepto'] = 1

# Definir as variáveis independentes (X)
X = df[['Intercepto', 'Temperatura', 'Promocao_Marketing']]

# Definir a variável dependente (Y)
Y = df['Vendas_Sorvetes']

# Criar e ajustar o modelo de regressão linear múltipla
modelo = sm.OLS(Y, X).fit()

# Imprimir os resultados do modelo
print(modelo.summary())