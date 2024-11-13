import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionOVR:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.weights = {}
        self.bias = {}
        
        for c in self.classes:
            y_c = np.where(y == c, 1, 0)
            self.weights[c], self.bias[c] = self._train_binary_classifier(X, y_c)
    
    def _train_binary_classifier(self, X, y):
        m, n = X.shape
        weights = np.zeros(n)
        bias = 0
        
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, weights) + bias
            predictions = sigmoid(linear_model)
            
            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)
            
            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db
            
        return weights, bias
    
    def predict(self, X):
        probabilities = np.zeros((X.shape[0], self.num_classes))
        
        for c in self.classes:
            linear_model = np.dot(X, self.weights[c]) + self.bias[c]
            probabilities[:, c] = sigmoid(linear_model)
        
        predictions = np.argmax(probabilities, axis=1)
        return predictions

model = LogisticRegressionOVR(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")