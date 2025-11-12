
import numpy as np
from source.main import Perceptron

def test_perceptron_fit_predict():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0, 0, 0, 1])
    p = Perceptron(learning_rate=0.1, epochs=10)
    p.fit(X, y)
    preds = [p.predict(x) for x in X]
    assert all(isinstance(pred, int) for pred in preds)
