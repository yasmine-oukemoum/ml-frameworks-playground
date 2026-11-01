"""Version 1: Scikit-learn — simple, high-level ML API."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data_utils import make_dataset
from game_utils import play_game

X, y = make_dataset()
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X, y)

accuracy = model.score(X, y)
print(f"Scikit-learn model trained. Training accuracy: {accuracy:.3f}")

def predict_close(features):
    return int(model.predict(features)[0]) == 1

if __name__ == "__main__":
    play_game(predict_close)
