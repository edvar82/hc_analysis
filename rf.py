from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.read_csv("dataset_5secondWindow.csv")

df.drop(['user', 'time'], axis=1, inplace=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
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

for i in range(X_train.shape[1]):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

"""
android.sensor.accelerometer#std (0.0571):
Justificativa: O desvio padrão do acelerômetro foi a segunda feature mais importante na classificação, o que indica que a variação nos valores medidos pelo acelerômetro tem um grande impacto na determinação do alvo. Isso faz sentido em aplicações de sensores, onde variações nos movimentos podem ser decisivas para categorizar atividades ou comportamentos.

android.sensor.linear_acceleration#max (0.0397):
Justificativa: O valor máximo da aceleração linear também apareceu com alta importância. A aceleração linear mede mudanças bruscas na velocidade sem a influência da gravidade, o que pode ser muito relevante para detectar movimentos rápidos ou intensos em atividades físicas.

speed#mean (0.0355):
Justificativa: A média da velocidade é uma medida agregada importante, especialmente em contextos de detecção de atividades humanas ou veículos. Uma característica como a velocidade média pode diferenciar atividades (por exemplo, caminhar versus correr) ou diferentes estados de movimento (parado, acelerando, etc.).

android.sensor.gyroscope#mean (0.0300):
Justificativa: O giroscópio mede a rotação ao redor dos três eixos espaciais. A média desses valores foi a quarta feature mais importante, o que sugere que o modelo reconhece a rotação geral como um fator relevante. Isso pode ser útil para diferenciar movimentos como virar ou rotacionar o dispositivo, ou para detectar padrões em atividades que envolvem rotação.
"""