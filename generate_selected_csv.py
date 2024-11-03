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

selected_features = ['android.sensor.linear_acceleration#mean', 'android.sensor.linear_acceleration#min', 'android.sensor.gyroscope#mean', 'android.sensor.accelerometer#std', 'android.sensor.gyroscope#max', 'android.sensor.gyroscope_uncalibrated#mean', 'android.sensor.gyroscope_uncalibrated#max', 'android.sensor.gyroscope_uncalibrated#min', 'android.sensor.gyroscope_uncalibrated#std', 'android.sensor.gyroscope#min', 'android.sensor.gyroscope#std', 'android.sensor.linear_acceleration#max', 'speed#max', 'speed#mean', 'speed#min', 'android.sensor.linear_acceleration#std', 'sound#max', 'sound#mean', 'sound#min', 'android.sensor.magnetic_field#max', 'android.sensor.magnetic_field#mean', 'android.sensor.magnetic_field#min', 'android.sensor.game_rotation_vector#std', 'android.sensor.orientation#std', 'android.sensor.accelerometer#max', 'android.sensor.rotation_vector#std', 'android.sensor.step_counter#max', 'android.sensor.step_counter#mean', 'android.sensor.step_counter#min', 'android.sensor.game_rotation_vector#max', 'android.sensor.rotation_vector#max', 'android.sensor.game_rotation_vector#mean', 'android.sensor.rotation_vector#mean', 'activityrecognition#1', 'android.sensor.game_rotation_vector#min', 'android.sensor.rotation_vector#min', 'android.sensor.gravity#std', 'android.sensor.magnetic_field_uncalibrated#min', 'android.sensor.magnetic_field_uncalibrated#mean', 'android.sensor.magnetic_field_uncalibrated#max', 'android.sensor.accelerometer#mean', 'android.sensor.orientation#max', 'android.sensor.orientation#mean', 'android.sensor.orientation#min', 'android.sensor.accelerometer#min', 'android.sensor.proximity#max', 'android.sensor.proximity#mean', 'android.sensor.proximity#min', 'android.sensor.gravity#min', 'android.sensor.gravity#mean', 'android.sensor.gravity#max', 'android.sensor.magnetic_field#std', 'android.sensor.light#min', 'android.sensor.light#mean', 'android.sensor.light#max', 'speed#std', 'sound#std', 'android.sensor.proximity#std', 'android.sensor.pressure#mean', 'android.sensor.pressure#min', 'android.sensor.magnetic_field_uncalibrated#std', 'android.sensor.pressure#max', 'android.sensor.step_counter#std', 'android.sensor.light#std', 'android.sensor.pressure#std']

print(len(selected_features))

df_selected = df[selected_features + ['target']]

df_selected.to_csv("dataset_selected_features.csv", index=False)

print("Novo arquivo CSV salvo com as features selecionadas.")
