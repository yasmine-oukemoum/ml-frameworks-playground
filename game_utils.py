"""Shared terminal game logic."""
import numpy as np

def play_game(predict_close):
    secret = int(np.random.randint(1, 21))
    attempts = 0
    max_attempts = 6

    print("\n🎮 Guess the secret number (1–20)!")
    print("You have 6 attempts. The model will classify each guess as CLOSE or FAR.")

    while attempts < max_attempts:
        raw = input(f"\nAttempt {attempts + 1}/{max_attempts} — your guess: ").strip()
        try:
            guess = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if not 1 <= guess <= 20:
            print("Choose a number from 1 to 20.")
            continue

        attempts += 1
        distance = abs(guess - secret)
        features = np.array([[distance, attempts]], dtype=np.float32)
        close = bool(predict_close(features))

        print("🤖 Model says:", "CLOSE" if close else "FAR")
        # The game's actual hint makes it possible to learn the rules of the game.
        if guess == secret:
            print(f"🎉 Correct! The secret number was {secret}.")
            return
        print("Game hint:", "too low" if guess < secret else "too high")

    print(f"Game over! The secret number was {secret}.")
