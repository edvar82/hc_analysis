import ordpy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.impute import KNNImputer

def compute_entropy_complexity(window, dx=3):
    """Computa entropia e complexidade para uma janela dada."""
    if len(window) < dx:
        return np.nan, np.nan
    try:
        entropy, complexity = ordpy.complexity_entropy(window, dx=dx)
        return entropy, complexity
    except Exception as e:
        print(f"Error computing HC: {e}")
        return np.nan, np.nan

def main():
    dx = 3  # Dimensão de embedding para permutation entropy
    window_size = 20  # Tamanho da janela móvel para capturar padrões temporais

    df = pd.read_csv("dataset_selected_features.csv")

    columns = [
        'android.sensor.linear_acceleration#mean', 'android.sensor.linear_acceleration#min',
        'android.sensor.gyroscope#mean', 'android.sensor.accelerometer#std',
        'android.sensor.gyroscope#max', 'android.sensor.gyroscope_uncalibrated#mean',
        'android.sensor.gyroscope_uncalibrated#max', 'android.sensor.gyroscope_uncalibrated#min',
        'android.sensor.gyroscope_uncalibrated#std', 'android.sensor.gyroscope#min',
        'android.sensor.gyroscope#std', 'android.sensor.linear_acceleration#max',
        'speed#max', 'speed#mean', 'speed#min', 'android.sensor.linear_acceleration#std',
        'sound#max', 'sound#mean', 'sound#min', 'android.sensor.magnetic_field#max',
        'android.sensor.magnetic_field#mean', 'android.sensor.magnetic_field#min',
        'android.sensor.game_rotation_vector#std', 'android.sensor.orientation#std',
        'android.sensor.accelerometer#max', 'android.sensor.rotation_vector#std',
        'android.sensor.step_counter#max', 'android.sensor.step_counter#mean',
        'android.sensor.step_counter#min', 'android.sensor.game_rotation_vector#max',
        'android.sensor.rotation_vector#max', 'android.sensor.game_rotation_vector#mean',
        'android.sensor.rotation_vector#mean', 'activityrecognition#1',
        'android.sensor.game_rotation_vector#min', 'android.sensor.rotation_vector#min',
        'android.sensor.gravity#std', 'android.sensor.magnetic_field_uncalibrated#min',
        'android.sensor.magnetic_field_uncalibrated#mean', 'android.sensor.magnetic_field_uncalibrated#max',
        'android.sensor.accelerometer#mean', 'android.sensor.orientation#max',
        'android.sensor.orientation#mean', 'android.sensor.orientation#min',
        'android.sensor.accelerometer#min', 'android.sensor.proximity#max',
        'android.sensor.proximity#mean', 'android.sensor.proximity#min',
        'android.sensor.gravity#min', 'android.sensor.gravity#mean',
        'android.sensor.gravity#max', 'android.sensor.magnetic_field#std',
        'android.sensor.light#min', 'android.sensor.light#mean', 'android.sensor.light#max',
        'speed#std', 'sound#std', 'android.sensor.proximity#std',
        'android.sensor.pressure#mean', 'android.sensor.pressure#min',
        'android.sensor.magnetic_field_uncalibrated#std', 'android.sensor.pressure#max',
        'android.sensor.step_counter#std', 'android.sensor.light#std',
        'android.sensor.pressure#std'
    ]

    print(f"Total de features: {len(columns)}")

    entropy_dict = {}
    complexity_dict = {}

    for col in tqdm(columns, desc="Processando Features"):
        series = df[col].values
        entropy_list = []
        complexity_list = []
        
        for i in range(len(series)):
            if i + 1 < window_size:
                entropy = np.nan
                complexity = np.nan
            else:
                window = series[i + 1 - window_size:i + 1]
                entropy, complexity = compute_entropy_complexity(window, dx=dx)
            entropy_list.append(entropy)
            complexity_list.append(complexity)
        
        entropy_dict[f"{col}_entropy"] = entropy_list
        complexity_dict[f"{col}_complexity"] = complexity_list

    entropy_df = pd.DataFrame(entropy_dict)
    complexity_df = pd.DataFrame(complexity_dict)

    df = pd.concat([df, entropy_df, complexity_df], axis=1)

    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    df_imputed['target'] = df['target']

    print(f"Total de colunas após adição das métricas: {len(df_imputed.columns)}")

    df_imputed.to_csv("dataset_with_hc_tratado.csv", index=False)

if __name__ == "__main__":
    main()
