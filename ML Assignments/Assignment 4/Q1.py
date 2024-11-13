import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class RidgeRegression:
    def __init__(self, learning_rate, iterations, l2_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_penalty = l2_penalty

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = (-2 * (self.X.T).dot(self.Y - Y_pred) + (2 * self.l2_penalty * self.W)) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.W) + self.b

    def cost(self):
        Y_pred = self.predict(self.X)
        cost = (np.sum((self.Y - Y_pred) ** 2) + self.l2_penalty * np.sum(self.W ** 2)) / self.m
        return cost

def generate_synthetic_data():
    np.random.seed(0)
    X = np.random.rand(100, 7)
    X[:, 1:] = X[:, :-1] + np.random.normal(0, 0.1, (100, 6))
    Y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)
    return X, Y

def evaluate_model(X_train, y_train, X_test, y_test, learning_rates, regularization_params):
    best_r2 = -float('inf')
    best_cost = float('inf')
    best_params = None

    for lr in learning_rates:
        for reg in regularization_params:
            model = RidgeRegression(learning_rate=lr, iterations=1000, l2_penalty=reg)
            model.fit(X_train, y_train)

            Y_pred = model.predict(X_test)

            if np.any(np.isnan(Y_pred)):
                print(f"NaN detected in predictions with Learning Rate: {lr}, Regularization: {reg}")
                continue

            r2 = r2_score(y_test, Y_pred)
            cost = model.cost()

            if r2 > best_r2 and cost < best_cost:
                best_r2 = r2
                best_cost = cost
                best_params = (lr, reg)

            print(f"Learning Rate: {lr}, Regularization: {reg} => R2: {r2:.4f}, Cost: {cost:.4f}")

    return best_params, best_r2, best_cost

def main():
    X, Y = generate_synthetic_data()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    regularization_params = [10**-15, 10**-10, 10**-5, 10**-3, 0, 1, 10, 20]

    best_params, best_r2, best_cost = evaluate_model(X_train, Y_train, X_test, Y_test, learning_rates, regularization_params)

    print(f"\nBest Parameters: Learning Rate: {best_params[0]}, Regularization: {best_params[1]}")
    print(f"Best R2 Score: {best_r2:.4f}, Best Cost: {best_cost:.4f}")

    best_lr, best_reg = best_params
    final_model = RidgeRegression(learning_rate=best_lr, iterations=1000, l2_penalty=best_reg)
    final_model.fit(X_train, Y_train)

    Y_pred = final_model.predict(X_test)

    plt.scatter(Y_test, Y_pred, color='blue')
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
    plt.title(f'Ridge Regression: Test vs Predicted (R2: {best_r2:.4f})')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.show()

if __name__ == "__main__":
    main()