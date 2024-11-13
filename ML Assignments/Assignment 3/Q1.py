import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

url = "./USA_Housing.csv"
df = pd.read_csv(url)

print(df.columns)

X = df.drop(columns=['Price'])
y = df['Price']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

best_beta = None
best_r2 = -np.inf
betas = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

    beta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
    betas.append(beta)
    
    y_pred = X_test_bias @ beta
    r2 = r2_score(y_test, y_pred)
    
    if r2 > best_r2:
        best_r2 = r2
        best_beta = beta


train_size = int(0.7 * len(X_scaled))
X_train_final, X_test_final = X_scaled[:train_size], X_scaled[train_size:]
y_train_final, y_test_final = y[:train_size], y[train_size:]

X_train_final_bias = np.c_[np.ones(X_train_final.shape[0]), X_train_final]
X_test_final_bias = np.c_[np.ones(X_test_final.shape[0]), X_test_final]

y_pred_final = X_test_final_bias @ best_beta

final_r2 = r2_score(y_test_final, y_pred_final)

print(f"Best Beta Matrix: {best_beta}")
print(f"Best R2 Score (cross-validation): {best_r2}")
print(f"Final R2 Score (train on 70%, test on 30%): {final_r2}")
