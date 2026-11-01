"""Create a small synthetic dataset shared by all three examples."""
import numpy as np

def make_dataset(n_samples=4000, seed=42):
    rng = np.random.default_rng(seed)
    # The model sees the distance between the guess and secret number.
    distance = rng.integers(0, 20, size=n_samples)
    attempts_used = rng.integers(1, 7, size=n_samples)
    X = np.column_stack([distance, attempts_used]).astype(np.float32)
    # Close means within 3 numbers of the secret.
    y = (distance <= 3).astype(np.int64)
    return X, y
