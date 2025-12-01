import numpy as np
from typing import Sequence, Tuple, List, Union, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from .base import ClassifierStrategy


def _to_tuple(hidden_layer_sizes) -> Tuple[int, ...]:
    if hidden_layer_sizes is None:
        return (32,)
    if isinstance(hidden_layer_sizes, int):
        return (hidden_layer_sizes,)
    return tuple(int(h) for h in hidden_layer_sizes)


class BackpropagationClassifier(BaseEstimator, ClassifierMixin):
    """Simple fully-connected neural network trained with backpropagation."""

    def __init__(
        self,
    hidden_layer_sizes: Union[Sequence[int], int] = (32, 16),
        activation: str = "relu",
        learning_rate: float = 0.01,
        max_iter: int = 200,
        batch_size: int = 32,
        l2: float = 0.0,
        early_stopping: bool = True,
        patience: int = 20,
        tol: float = 1e-4,
    random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    # ---------------------- Fit helpers ----------------------
    def _check_activation(self) -> None:
        if self.activation not in {"relu", "tanh", "sigmoid"}:
            raise ValueError("activation must be 'relu', 'tanh', or 'sigmoid'")

    def _activation(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "tanh":
            return np.tanh(z)
        # sigmoid
        return 1.0 / (1.0 + np.exp(-z))

    def _activation_grad(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(z.dtype)
        if self.activation == "tanh":
            t = np.tanh(z)
            return 1 - t * t
        sig = 1.0 / (1.0 + np.exp(-z))
        return sig * (1 - sig)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z_shifted = z - z.max(axis=1, keepdims=True)
        exp_scores = np.exp(z_shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        zs = []
        for idx in range(len(self.weights_) - 1):
            z = activations[-1] @ self.weights_[idx] + self.biases_[idx]
            zs.append(z)
            activations.append(self._activation(z))
        z_out = activations[-1] @ self.weights_[-1] + self.biases_[-1]
        zs.append(z_out)
        activations.append(self._softmax(z_out))
        return activations, zs

    def _backward(self, activations: List[np.ndarray], zs: List[np.ndarray], y_true: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        grads_w = [np.zeros_like(w) for w in self.weights_]
        grads_b = [np.zeros_like(b) for b in self.biases_]
        batch_size = y_true.shape[0]

        delta = (activations[-1] - y_true) / batch_size
        grads_w[-1] = activations[-2].T @ delta + self.l2 * self.weights_[-1]
        grads_b[-1] = delta.sum(axis=0)

        for layer in range(len(self.weights_) - 2, -1, -1):
            delta = (delta @ self.weights_[layer + 1].T) * self._activation_grad(zs[layer])
            grads_w[layer] = activations[layer].T @ delta + self.l2 * self.weights_[layer]
            grads_b[layer] = delta.sum(axis=0)
        return grads_w, grads_b

    def _cross_entropy(self, y_true: np.ndarray, probs: np.ndarray) -> float:
        eps = 1e-12
        return float(-np.mean(np.sum(y_true * np.log(probs + eps), axis=1)))

    def _init_params(self, n_features: int, n_classes: int, rng: np.random.Generator) -> None:
        layer_sizes = [n_features, *_to_tuple(self.hidden_layer_sizes), n_classes]
        self.weights_: List[np.ndarray] = []
        self.biases_: List[np.ndarray] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.weights_.append(rng.uniform(-limit, limit, size=(in_dim, out_dim)))
            self.biases_.append(np.zeros(out_dim))

    def fit(self, X, y):
        X_np = np.asarray(X, dtype=float)
        y_np = np.asarray(y)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X_np) == 0:
            raise ValueError("X cannot be empty")

        self._check_activation()
        self.classes_, y_idx = np.unique(y_np, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("Need at least two classes for classification")

        y_onehot = np.eye(n_classes)[y_idx]
        n_samples, n_features = X_np.shape
        self._rng = np.random.default_rng(self.random_state)
        self._init_params(n_features, n_classes, self._rng)
        batch_size = max(1, min(self.batch_size, n_samples))

        self.loss_curve_: List[float] = []
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(self.max_iter):
            perm = self._rng.permutation(n_samples)
            X_shuffled = X_np[perm]
            y_shuffled = y_onehot[perm]
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]
                activations, zs = self._forward(xb)
                probs = activations[-1]
                loss = self._cross_entropy(yb, probs)
                epoch_loss += loss * len(xb)
                grads_w, grads_b = self._backward(activations, zs, yb)
                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.learning_rate * grads_w[i]
                    self.biases_[i] -= self.learning_rate * grads_b[i]

            epoch_loss /= n_samples
            self.loss_curve_.append(epoch_loss)

            if self.verbose:
                print(f"[Backprop] epoch={epoch+1} loss={epoch_loss:.5f}")

            if self.early_stopping:
                if best_loss - epoch_loss > self.tol:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        self.n_iter_ = len(self.loss_curve_)
        self.n_features_in_ = n_features
        self._fitted = True
        return self

    def predict_proba(self, X):
        if not hasattr(self, "weights_"):
            raise RuntimeError("Model has not been fitted yet.")
        X_np = np.asarray(X, dtype=float)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        if X_np.ndim != 2:
            raise ValueError("X must be 2D")
        if hasattr(self, "n_features_in_") and X_np.shape[1] != self.n_features_in_:
            raise ValueError("Number of features does not match fitted data")
        activations, _ = self._forward(X_np)
        return activations[-1]

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = probs.argmax(axis=1)
        return self.classes_[indices]

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class BackpropStrategy(ClassifierStrategy):
    def build(self):
        return BackpropagationClassifier(**self.params)
