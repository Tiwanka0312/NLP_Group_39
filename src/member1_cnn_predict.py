import os
from pathlib import Path

# Reduce unnecessary TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from preprocessing import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_best_model.keras"

UNCERTAIN_THRESHOLD = 0.60


def load_cnn_model():
    """Load the trained CNN model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CNN model file was not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


def predict_news(news_text, model):
    """
    Predict whether a new article is fake or real.
    """

    if not isinstance(news_text, str):
        raise TypeError("News text must be a string.")

    if not news_text.strip():
        raise ValueError("News text cannot be empty.")

    cleaned_text = clean_text(news_text)

    if not cleaned_text:
        raise ValueError(
            "No valid text remained after preprocessing."
        )

    # CNN output represents the probability of Real News
    real_probability = float(
        model.predict(
            tf.constant([cleaned_text]),
            verbose=0
        )[0][0]
    )

    fake_probability = 1.0 - real_probability

    confidence = max(
        fake_probability,
        real_probability
    )

    if confidence < UNCERTAIN_THRESHOLD:
        prediction = "UNCERTAIN"
    elif real_probability >= 0.50:
        prediction = "REAL NEWS"
    else:
        prediction = "FAKE NEWS"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
    }


def read_news_article():
    """
    Read one news article from the terminal.
    """

    news_text = input(
        "\nEnter news text:\n> "
    ).strip()

    if news_text.lower() == "exit":
        return None

    return news_text


def main():
    try:
        model = load_cnn_model()
    except (FileNotFoundError, OSError) as error:
        print("Model loading error:", error)
        return

    print("=" * 55)
    print("Fake News Detection - CNN Model")
    print("=" * 55)
    print("Model loaded successfully.")

    while True:
        news_text = read_news_article()

        if news_text is None:
            print("Application closed.")
            break

        try:
            result = predict_news(
                news_text,
                model
            )

            print("\nPrediction Result")
            print("-" * 40)

            print(
                "Prediction       :",
                result["prediction"]
            )

            print(
                "Confidence       :",
                f'{result["confidence"] * 100:.2f}%'
            )

            print(
                "Fake Probability :",
                f'{result["fake_probability"] * 100:.2f}%'
            )

            print(
                "Real Probability :",
                f'{result["real_probability"] * 100:.2f}%'
            )

        except (TypeError, ValueError) as error:
            print("Prediction error:", error)


if __name__ == "__main__":
    main()