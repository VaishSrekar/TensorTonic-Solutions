import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=500):
    """
    Train binary logistic regression using gradient descent.

    Parameters:
    X : array-like of shape (N, D)
    y : array-like of shape (N,)
    lr : learning rate
    steps : number of gradient descent iterations

    Returns:
    (w, b)
    w : weight vector of shape (D,)
    b : bias (float)
    """

    # Convert to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    N, D = X.shape

    # Initialize parameters
    w = np.zeros(D)
    b = 0.0

    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Gradient descent loop
    for _ in range(steps):

        # Linear model
        z = X @ w + b

        # Predicted probabilities
        p = sigmoid(z)

        # Gradients
        dw = (1 / N) * (X.T @ (p - y))
        db = (1 / N) * np.sum(p - y)

        # Parameter update
        w -= lr * dw
        b -= lr * db

    return w, b


# Example test
X = [[0],[1],[2],[3]]
y = [0,0,1,1]

w, b = train_logistic_regression(X, y)

# Prediction check
X = np.array(X)
pred = 1/(1+np.exp(-(X@w + b)))
pred_labels = (pred >= 0.5).astype(int)

accuracy = np.mean(pred_labels == y)

print("Weights:", w)
print("Bias:", b)
print("Accuracy:", accuracy)