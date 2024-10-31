import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("dataset_halfSecondWindow.csv")

df.drop(['user', 'time', 'id', 'activityrecognition#0'], axis=1, inplace=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

selected_features = [
    'speed#min', 'android.sensor.linear_acceleration#mean', 'speed#mean', 
    'android.sensor.linear_acceleration#max', 'android.sensor.linear_acceleration#min', 
    'speed#max', 'android.sensor.magnetic_field#max', 'android.sensor.gyroscope#mean', 
    'android.sensor.light#max', 'android.sensor.light#min', 'android.sensor.gyroscope#max', 
    'android.sensor.magnetic_field#min', 'android.sensor.magnetic_field#mean', 'sound#mean', 
    'android.sensor.gyroscope#min', 'android.sensor.light#mean', 'android.sensor.accelerometer#std', 
    'sound#max', 'android.sensor.magnetic_field_uncalibrated#min', 'android.sensor.magnetic_field_uncalibrated#max', 
    'sound#min', 'android.sensor.magnetic_field_uncalibrated#mean', 'android.sensor.step_counter#mean', 
    'android.sensor.step_counter#min', 'android.sensor.linear_acceleration#std', 'activityrecognition#1', 
    'android.sensor.step_counter#max', 'android.sensor.gyroscope_uncalibrated#min', 'android.sensor.gyroscope_uncalibrated#mean', 
    'android.sensor.gravity#mean', 'android.sensor.gravity#min', 'android.sensor.game_rotation_vector#mean', 
    'android.sensor.game_rotation_vector#max', 'android.sensor.gravity#max', 'android.sensor.game_rotation_vector#min', 
    'android.sensor.rotation_vector#mean', 'android.sensor.game_rotation_vector#std', 'android.sensor.rotation_vector#max', 
    'android.sensor.rotation_vector#min', 'android.sensor.gyroscope_uncalibrated#std', 'android.sensor.orientation#max', 
    'android.sensor.orientation#std', 'android.sensor.orientation#min', 'android.sensor.pressure#max', 
    'android.sensor.pressure#mean', 'android.sensor.pressure#min', 'android.sensor.accelerometer#min', 
    'android.sensor.orientation#mean', 'android.sensor.gyroscope_uncalibrated#max', 'android.sensor.accelerometer#max', 
    'android.sensor.gravity#std', 'android.sensor.accelerometer#mean', 'android.sensor.gyroscope#std', 
    'sound#std', 'android.sensor.rotation_vector#std', 'android.sensor.light#std', 
    'android.sensor.proximity#mean', 'android.sensor.magnetic_field_uncalibrated#std', 
    'android.sensor.proximity#max', 'android.sensor.proximity#min', 'android.sensor.magnetic_field#std', 
    'speed#std', 'android.sensor.pressure#std', 'android.sensor.proximity#std', 'android.sensor.step_counter#std'
]

print(len(selected_features))

df_selected = df[selected_features + ['target']]

df_selected.to_csv("dataset_selected_features.csv", index=False)

print("Novo arquivo CSV salvo com as features selecionadas.")
