import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

columns = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration", 
           "num_doors", "body_style", "drive_wheels", "engine_location", 
           "wheel_base", "length", "width", "height", "curb_weight", "engine_type", 
           "num_cylinders", "engine_size", "fuel_system", "bore", "stroke", 
           "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg", 
           "price"]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, names=columns)

df.replace('?', np.nan, inplace=True)

df = df.apply(pd.to_numeric, errors='ignore')
df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(subset=['price'], inplace=True)


door_mapping = {'two': 2, 'four': 4}
cylinders_mapping = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}

df['num_doors'] = df['num_doors'].map(door_mapping).fillna(df['num_doors'])
df['num_cylinders'] = df['num_cylinders'].map(cylinders_mapping).fillna(df['num_cylinders'])

df = pd.get_dummies(df, columns=['body_style', 'drive_wheels'], drop_first=True)

label_columns = ['make', 'aspiration', 'engine_location', 'fuel_type']
label_encoder = LabelEncoder()

for column in label_columns:
    df[column] = label_encoder.fit_transform(df[column])

df['fuel_system'] = df['fuel_system'].apply(lambda x: 1 if 'pfi' in str(x) else 0)

df['engine_type'] = df['engine_type'].apply(lambda x: 1 if 'ohc' in str(x) else 0)

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")