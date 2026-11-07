# --- cell 1 ---
import numpy as np

class NeuralNetwork:
    """
    Architecture: 784 → 128 (ReLU) → 10 (Softmax)
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr

        # He initialization for ReLU layer
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)

        # Xavier initialization for output layer
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        # Layer 1: Linear + ReLU
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)

        # Layer 2: Linear + Softmax
        self.z2 = self.a1 @ self.W2 + self.b2

        # Numerically stable softmax
        exp_z = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return probs

    def backward(self, X, y, probs):
        N = X.shape[0]

        # 1. Cross-entropy loss
        loss = -np.mean(np.log(np.clip(probs[np.arange(N), y], 1e-15, 1.0)))

        # 2. Softmax + cross-entropy gradient
        dz2 = probs.copy()
        dz2[np.arange(N), y] -= 1
        dz2 /= N

        # 3. Gradients for layer 2
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # 4. Backpropagate through ReLU
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)

        # 5. Gradients for layer 1
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # 6. Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return float(loss)

    def train_step(self, X, y):
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# --- cell 2 ---
import numpy as np

class NeuralNetwork:
    """
    Architecture: 784 → 128 (ReLU) → 10 (Softmax)
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr

        # He initialization for ReLU layer
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)

        # Xavier initialization for output layer
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        # Layer 1: Linear + ReLU
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)

        # Layer 2: Linear + Softmax
        self.z2 = self.a1 @ self.W2 + self.b2

        # Numerically stable softmax
        exp_z = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return probs

    def backward(self, X, y, probs):
        N = X.shape[0]

        # 1. Cross-entropy loss
        loss = -np.mean(np.log(np.clip(probs[np.arange(N), y], 1e-15, 1.0)))

        # 2. Softmax + cross-entropy gradient
        dz2 = probs.copy()
        dz2[np.arange(N), y] -= 1
        dz2 /= N

        # 3. Gradients for layer 2
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # 4. Backpropagate through ReLU
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)

        # 5. Gradients for layer 1
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # 6. Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return float(loss)

    def train_step(self, X, y):
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)