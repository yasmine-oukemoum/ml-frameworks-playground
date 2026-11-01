"""Version 2: PyTorch — define the network and training loop explicitly."""
import numpy as np
import torch
from torch import nn
from data_utils import make_dataset
from game_utils import play_game

torch.manual_seed(42)
X, y = make_dataset()
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

for epoch in range(300):
    probabilities = model(X_tensor)
    loss = loss_fn(probabilities, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = (model(X_tensor) >= 0.5).float()
    accuracy = (predictions == y_tensor).float().mean().item()
print(f"PyTorch model trained. Training accuracy: {accuracy:.3f}")

def predict_close(features):
    with torch.no_grad():
        x = torch.tensor(np.asarray(features), dtype=torch.float32)
        return float(model(x)[0, 0]) >= 0.5

if __name__ == "__main__":
    play_game(predict_close)
