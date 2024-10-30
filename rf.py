from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.read_csv("dataset_halfSecondWindow.csv")

df.drop(['user', 'time', 'id'], axis=1, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
print(categorical_cols)
if len(categorical_cols) > 0:
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(['target'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

n_features = 60
top_features = X.columns[indices[:n_features]]

df_selected = df[top_features.to_list() + ['target']]

df_selected.to_csv("dataset_halfSecondWindow_selected_features.csv", index=False)


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import numpy as np

# # Carregar e preprocessar os dados
# df = pd.read_csv("dataset_halfSecondWindow.csv")

# # Pré-processamento: Remover colunas irrelevantes e preencher valores ausentes
# df.drop(['user', 'time'], axis=1, inplace=True)
# df.fillna(df.mean(numeric_only=True), inplace=True)

# Codificação de variáveis categóricas
# categorical_cols = df.select_dtypes(include=['object']).columns
# if len(categorical_cols) > 0:
#     le = LabelEncoder()
#     for col in categorical_cols:
#         df[col] = le.fit_transform(df[col].astype(str))

# # Separar features (X) e o alvo (y)
# X = df.drop(['target'], axis=1)
# y = df['target']

# # Dividir em conjuntos de treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Calcular importâncias das features usando RandomForest
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]

# # Definir quantidades de features para teste
# num_features_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 66]
# results = {}

# # Testar o modelo com diferentes quantidades de features
# for num_features in num_features_list:
#     selected_features = X.columns[indices[:num_features]]
#     X_train_selected = X_train[selected_features]
#     X_test_selected = X_test[selected_features]
    
#     # Treinar o modelo MLP
#     mlp = MLPClassifier(random_state=42, max_iter=500)
#     mlp.fit(X_train_selected, y_train)
    
#     # Avaliar o desempenho
#     y_pred = mlp.predict(X_test_selected)
#     accuracy = accuracy_score(y_test, y_pred)
#     results[num_features] = (accuracy, selected_features)
#     print(f"Acurácia com {num_features} features: {accuracy:.4f}")

# # Identificar o melhor resultado e exibir as features correspondentes
# best_num_features = max(results, key=lambda k: results[k][0])
# best_accuracy, best_features = results[best_num_features]

# print("\nResumo dos resultados:")
# for num_features, (accuracy, _) in results.items():
#     print(f"{num_features} features: Acurácia = {accuracy:.4f}")

# print(f"\nMelhor configuração: {best_num_features} features com acurácia de {best_accuracy:.4f}")
# print("Features selecionadas:", best_features.tolist())

