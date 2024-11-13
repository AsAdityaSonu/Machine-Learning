import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        
        self.means = np.zeros((self.n_classes, self.n_features))
        self.variances = np.zeros((self.n_classes, self.n_features))
        self.priors = np.zeros(self.n_classes)
        
        for i in self.classes:
            X_class = X[y == i]
            self.means[i] = X_class.mean(axis=0)
            self.variances[i] = X_class.var(axis=0)
            self.priors[i] = X_class.shape[0] / float(X.shape[0])
    
    def predict(self, X):
        probs = np.zeros((X.shape[0], self.n_classes))
        
        for i in self.classes:
            mean = self.means[i]
            variance = self.variances[i]
            prior = np.log(self.priors[i])
            
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance)) - 0.5 * np.sum(((X - mean) ** 2) / variance, axis=1)
            probs[:, i] = prior + likelihood
        
        return np.argmax(probs, axis=1)

gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))