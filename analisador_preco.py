import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Carregar os datasets de treino e teste do diretório especificado
# O conjunto de treino contém as variáveis explicativas e a variável-alvo (SalePrice)
# O conjunto de teste será usado para fazer previsões finais
train_data = pd.read_csv('C:\\Users\\Gustavo\\Documents\\Faculdade\\Fase 6\\Inteligência artificial\\N3\\home-data-for-ml-course\\train.csv')
test_data = pd.read_csv('C:\\Users\\Gustavo\\Documents\\Faculdade\\Fase 6\\Inteligência artificial\\N3\\home-data-for-ml-course\\test.csv')

# Visualizar as primeiras linhas do conjunto de treino para entender sua estrutura
print(train_data.head())

# Separar a variável-alvo (preço das casas) e as características (demais colunas)
# A coluna 'Id' é descartada, pois não é relevante para a modelagem
y = train_data['SalePrice']  # Variável-alvo
X = train_data.drop(['SalePrice', 'Id'], axis=1)  # Características do treino
X_test = test_data.drop(['Id'], axis=1)  # Características do teste (sem o preço)

# Tratar valores ausentes no conjunto de treino
# Verifica cada coluna e preenche os valores ausentes com:
# - Moda (valor mais frequente) para colunas categóricas
# - Mediana para colunas numéricas
for col in X.columns:
    if X[col].isnull().sum() > 0:  # Se houver valores ausentes
        if X[col].dtype == 'object':  # Coluna categórica
            X[col] = X[col].fillna(X[col].mode()[0])
        else:  # Coluna numérica
            X[col] = X[col].fillna(X[col].median())

# Repetir o mesmo processo para o conjunto de teste
for col in X_test.columns:
    if X_test[col].isnull().sum() > 0:  # Se houver valores ausentes
        if X_test[col].dtype == 'object':  # Coluna categórica
            X_test[col] = X_test[col].fillna(X_test[col].mode()[0])
        else:  # Coluna numérica
            X_test[col] = X_test[col].fillna(X_test[col].median())

# Codificar variáveis categóricas em valores numéricos com LabelEncoder
# Isso é necessário para que os modelos de machine learning possam processar esses dados
label_enc = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':  # Se a coluna for categórica
        X[col] = label_enc.fit_transform(X[col])  # Codifica os dados de treino
        X_test[col] = label_enc.transform(X_test[col])  # Aplica a mesma codificação no teste

# Dividir os dados de treino em conjunto de treino e validação
# O conjunto de validação será usado para avaliar os modelos
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir os modelos que serão avaliados
models = {
    "LinearRegression": LinearRegression(),  # Regressão linear básica
    "Ridge": Ridge(alpha=1.0),  # Regressão com regularização L2
    "Lasso": Lasso(alpha=0.1),  # Regressão com regularização L1
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),  # Floresta Aleatória
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)  # Regressor XGBoost
}

# Avaliar o desempenho de cada modelo
# Para cada modelo:
# 1. Ajustar o modelo com os dados de treino
# 2. Fazer previsões no conjunto de validação
# 3. Calcular RMSE e MAE para avaliar a precisão das previsões
for name, model in models.items():
    model.fit(X_train, y_train)  # Treinar o modelo
    y_pred = model.predict(X_val)  # Fazer previsões
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))  # Raiz do Erro Quadrático Médio
    mae = mean_absolute_error(y_val, y_pred)  # Erro Médio Absoluto
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")  # Exibir os resultados

# Escolher o modelo com melhor desempenho (aqui, RandomForest como exemplo)
# Treiná-lo novamente com todos os dados de treino
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X, y)  # Treinar o modelo final com todos os dados de treino

# Fazer previsões no conjunto de teste
y_test_pred = best_model.predict(X_test)

# Criar um arquivo de submissão no formato exigido pela competição
# O arquivo contém duas colunas: 'Id' e 'SalePrice'
submission = pd.DataFrame({
    "Id": test_data['Id'],  # IDs das casas no conjunto de teste
    "SalePrice": y_test_pred  # Preços previstos
})
submission.to_csv('submission.csv', index=False)  # Salvar o arquivo no disco

print("Arquivo de submissão criado: submission.csv")  # Mensagem de conclusão
