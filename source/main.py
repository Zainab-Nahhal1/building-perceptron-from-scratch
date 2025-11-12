
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.weights = np.random.rand(X.shape[1])
        self.bias = np.random.rand(1)
        for _ in range(self.epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
    
    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum >= 0 else 0

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]
y = np.where(iris.target != 0, 0, 1)

# Train perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.fit(X, y)

# Predictions and plot
y_pred = np.array([perceptron.predict(x) for x in X])
df = pd.DataFrame(X, columns=iris.feature_names[:2])
df['Prediction'] = y_pred

plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['Prediction'], cmap='viridis')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Perceptron Predictions')
plt.show()
