import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=5, min_samples_leaf=4, random_state=42)

dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on test set: {accuracy * 100:.2f}%")

plt.figure(figsize=(20,10))
plot_tree(dt_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True, fontsize=14)
plt.show()