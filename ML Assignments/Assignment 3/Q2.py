import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

url = "./USA_Housing.csv"
df = pd.read_csv(url)

X = df.drop(columns=['Price'])
y = df['Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_size = int(0.56 * len(X_scaled))
val_size = int(0.14 * len(X_scaled))

X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

def gradient_descent(X, y, learning_rate, iterations):
    m = len(y)
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_bias.shape[1])
    
    for i in range(iterations):
        predictions = X_bias.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X_bias.T.dot(errors)
        theta -= learning_rate * gradient
        
    return theta

learning_rates = [0.001, 0.01, 0.1, 1]
iterations = 1000
best_r2_val = -np.inf
best_r2_test = -np.inf
best_beta = None
best_learning_rate = None

for lr in learning_rates:
    beta = gradient_descent(X_train, y_train, lr, iterations)
    
    X_val_bias = np.c_[np.ones((X_val.shape[0], 1)), X_val]
    X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    y_val_pred = X_val_bias.dot(beta)
    r2_val = r2_score(y_val, y_val_pred)
    
    y_test_pred = X_test_bias.dot(beta)
    r2_test = r2_score(y_test, y_test_pred)
    
    if r2_val > best_r2_val:
        best_r2_val = r2_val
        best_r2_test = r2_test
        best_beta = beta
        best_learning_rate = lr

print(f"Learning Rate: {best_learning_rate}")
print(f"Beta Coefficients: {best_beta}")
print(f"R2 Score on Validation Set: {best_r2_val}")
print(f"R2 Score on Test Set: {best_r2_test}")