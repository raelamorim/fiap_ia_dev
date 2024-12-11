import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Fluxo de trabalho
# Carregar os dados.
# Tratar dados faltantes, se houver.
# Fazer a codificação das variáveis categóricas (como "sex").
# Separar os dados em conjuntos de treino e teste.
# Aplicar qualquer outro pré-processamento, como escalonamento.

dados = pd.read_csv("insurance.csv")
print(dados.head())

# Contar valores nulos em cada coluna
nulos = dados.isnull().sum()
print(nulos)
# Preencher com a Média
# dados['column_name'] = dados['column_name'].fillna(dados['column_name'].mean())

# Fazer a codificação das variáveis categóricas
# Criar um codificador
label_encoder = LabelEncoder()
 
# Aplicar a codificação à coluna 'sex'
dados['sex'] = label_encoder.fit_transform(dados['sex'])

# Usar one-hot encoding para a coluna 'region'
dados = pd.get_dummies(dados, columns=['region'], drop_first=True)
# # Usando LabelEncoder para converter categorias em números
# le = LabelEncoder()
# dados['region'] = le.fit_transform(dados['region'])
print(dados.head())

# Converter a coluna booleana para inteiros (1 e 0)
dados['smoker'] = dados['smoker'].map({'yes': 1, 'no': 0})

# Adicionando constante para o intercepto
dados = sm.add_constant(dados)

# # Estratificando os dados
# separando a coluna "bmi" em faixas de 10
# Definindo as faixas de 20 em 20 anos, de 0 a 80
bins = [10, 20, 30, 40, 50, 60]  # Os limites das faixas
labels = [1, 2, 3, 4, 5]  # Rótulos para as faixas

# Criando a nova coluna 'bmi_group' com a estratificação
dados['bmi_group'] = pd.cut(dados['bmi'], bins=bins, labels=labels, right=False)
value_counts = dados['bmi_group'].value_counts()
print(value_counts)
print(dados.head())

# Criando um objeto StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
 
# Dividindo os dados
for train_index, test_index in sss.split(dados, dados['bmi_group']):
    strat_train_set = dados.iloc[train_index]
    strat_test_set = dados.iloc[test_index]

#analisando as proporções da estratificação
# print(strat_train_set['bmi_group'].value_counts() / len(train_set))
# print(strat_test_set['bmi_group'].value_counts() / len(test_set))
# print(dados['bmi_group'].value_counts() / len(dados))

# Remova a coluna 'bmi' se estiver lá
strat_train_set = strat_train_set.drop("bmi", axis=1, errors='ignore')
strat_test_set = strat_test_set.drop("bmi", axis=1, errors='ignore')

# strat_train_set['region_northwest'] = strat_train_set['region_northwest'].astype(int)

# Agora separando as variáveis X e Y
X_train = strat_train_set.drop(columns='charges')  # Remove a coluna alvo do conjunto de treino
Y_train = strat_train_set['charges']  # A coluna alvo do conjunto de treino
 
X_test = strat_test_set.drop(columns='charges')  # Remove a coluna alvo do conjunto de teste
Y_test = strat_test_set['charges']  # A coluna alvo do conjunto de teste

# Aplicando escalonamento (verificar estratégias mais adequadas à base de dados)
# Normalização (Min-Max Scaling)
# Aplicar escalonamento apenas nos dados de treino
scaler = MinMaxScaler()
X_train_scaled  = scaler.fit_transform(X_train)

# Transformar os dados de teste usando o mesmo escalonador
X_test_scaled = scaler.fit_transform(X_test)


# Padronizando o X
X_train_scaled = StandardScaler().fit_transform(X_train_scaled)
X_test_scaled = StandardScaler().fit_transform(X_test_scaled)

# Aplicando o PCA
# for n in range(2,8):
#     pca = PCA(n_components=n)
#     pca.fit_transform(X_train_scaled)
#     print(f"Variabilidade aplicada: {np.sum(pca.explained_variance_ratio_)}")

# pca = PCA()
# pca.fit_transform(X_train_scaled)
# variancia_cumulativa = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(range(1, len(variancia_cumulativa) + 1), variancia_cumulativa, marker='o')
# plt.show()

# pca = PCA(n_components=2)
# X_train_scaled = pca.fit_transform(X_train_scaled)
# print(f"Variabilidade aplicada: {pca.explained_variance_ratio_}")
# X_test_scaled = pca.fit_transform(X_test_scaled)
# print(f"Variabilidade aplicada: {pca.explained_variance_ratio_}")

df_X_train_temp = pd.DataFrame(X_train_scaled)
print(df_X_train_temp.head())


# Criando o modelo de regressão
model = sm.OLS(Y_train, X_train_scaled).fit()
# Fazendo previsões
Y_pred = model.predict(X_test_scaled)
# Avaliando o modelo
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(Y_test, Y_pred)
print("sm.OLS")
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r_squared}")

linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, Y_train)
Y_pred = linear_reg.predict(X_test_scaled)
# Avaliando o modelo
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(Y_test, Y_pred)
print("LinearRegression")
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r_squared}")


""" # Crie um conjunto de hiperparâmetros para testar
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Inicialize o modelo
dt_regressor = DecisionTreeRegressor(random_state=42)
# Configure o GridSearchCV
grid_search = GridSearchCV(estimator=dt_regressor, 
                           param_grid=param_grid, 
                           cv=5,  # utilização de 5 folds para validação cruzada
                           scoring='neg_mean_squared_error', 
                           return_train_score=True)
                           
# Ajuste o modelo aos dados de treino
grid_search.fit(X_train_scaled, Y_train)
# Mostre os melhores parâmetros encontrados
print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor score (MSE negativo):", grid_search.best_score_) """

decision_reg = DecisionTreeRegressor(max_depth=5, min_samples_leaf=4, min_samples_split=10)
decision_reg.fit(X_train_scaled, Y_train)
Y_pred = decision_reg.predict(X_test_scaled)

# Avaliando o modelo
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(Y_test, Y_pred)

# Resultados
# print("Coeficientes:")
# print(model.params)
print("DecisionTreeRegressor")
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r_squared}")

