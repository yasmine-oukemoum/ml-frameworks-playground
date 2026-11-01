# ML Libraries Guessing Game

A small terminal game to compare **Scikit-learn**, **PyTorch**, and **TensorFlow/Keras**.

The computer chooses a secret number from 1 to 20. After each guess, a model classifies how close the guess is:
- `CLOSE`: the guess is within 3 of the secret number.
- `FAR`: the guess is more than 3 away.

This is an educational demo. The game itself knows the secret number, so the model is only used to demonstrate the different library APIs—not to make the game smarter.

## 1. Create an environment

Python 3.11 is recommended.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install the library you want to try

Install one at a time to keep setup simple.

### Scikit-learn
```bash
pip install numpy scikit-learn
python play_sklearn.py
```

### PyTorch
Install PyTorch using the official selector if you need a specific CPU/GPU build: https://pytorch.org/get-started/locally/
Then run:
```bash
pip install numpy
python play_pytorch.py
```

### TensorFlow / Keras
```bash
pip install numpy tensorflow
python play_tensorflow.py
```

TensorFlow installation support depends on your operating system and Python version. If installation fails, check the official install guide: https://www.tensorflow.org/install

## 3. What to compare

| Library | Main idea | In this project |
|---|---|---|
| Scikit-learn | Ready-to-use ML algorithms | `model.fit()` and `model.predict()` |
| PyTorch | Build a neural network and write the training loop | `loss.backward()` and `optimizer.step()` |
| TensorFlow/Keras | Build and train a neural network with a high-level API | `model.compile()` and `model.fit()` |

All three scripts use the same kind of synthetic training data and the same `CLOSE`/`FAR` task. The model architectures and training APIs are not identical, so this is an API-learning comparison, not a scientific benchmark.

## 4. Push to GitHub

Create an empty repository on GitHub, then from this project folder run:
```bash
git init
git add .
git commit -m "Add ML libraries guessing game"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ml-libraries-guessing-game.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## Project structure
```text
ml-libraries-guessing-game/
├── README.md
├── requirements-common.txt
├── .gitignore
├── data_utils.py
├── game_utils.py
├── play_sklearn.py
├── play_pytorch.py
└── play_tensorflow.py
```
