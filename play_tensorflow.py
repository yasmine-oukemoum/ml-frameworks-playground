"""Version 3: TensorFlow/Keras — concise high-level neural-network training."""
import numpy as np
import tensorflow as tf
from data_utils import make_dataset
from game_utils import play_game

tf.keras.utils.set_random_seed(42)
X, y = make_dataset()
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(X, y, epochs=80, batch_size=64, verbose=0)
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"TensorFlow/Keras model trained. Training accuracy: {accuracy:.3f}")

def predict_close(features):
    probability = model.predict(np.asarray(features, dtype=np.float32), verbose=0)[0, 0]
    return float(probability) >= 0.5

if __name__ == "__main__":
    play_game(predict_close)
