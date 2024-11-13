import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13), store_cv_values=True)
ridge_cv.fit(X_train, y_train)

lasso_cv = LassoCV(alphas=np.logspace(-6, 6, 13), max_iter=10000)
lasso_cv.fit(X_train, y_train)


y_pred_ridge = ridge_cv.predict(X_test)

y_pred_lasso = lasso_cv.predict(X_test)

r2_ridge = r2_score(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print(f"RidgeCV: Best alpha = {ridge_cv.alpha_}, R2 = {r2_ridge:.4f}, MSE = {mse_ridge:.4f}")
print(f"LassoCV: Best alpha = {lasso_cv.alpha_}, R2 = {r2_lasso:.4f}, MSE = {mse_lasso:.4f}")

if r2_ridge > r2_lasso:
    print("RidgeCV performs better.")
else:
    print("LassoCV performs better.")