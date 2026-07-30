from pathlib import Path

import joblib

# Import this because the saved TF-IDF vectorizer uses clean_text
from preprocessing import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "logistic_regression_model.joblib"
VECTORIZER_PATH = (
    MODELS_DIR
    / "logistic_regression_tfidf_vectorizer.joblib"
)


def load_saved_files():
    """Load the trained model and TF-IDF vectorizer."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found: {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_news(news_text, model, vectorizer):
    """Predict whether an unseen news article is fake or real."""

    if not news_text.strip():
        raise ValueError("News text cannot be empty.")

    news_features = vectorizer.transform([news_text])

    predicted_label = int(
        model.predict(news_features)[0]
    )

    probabilities = model.predict_proba(
        news_features
    )[0]

    class_positions = {
        int(label): position
        for position, label in enumerate(model.classes_)
    }

    fake_probability = probabilities[
        class_positions[0]
    ]

    real_probability = probabilities[
        class_positions[1]
    ]

    confidence = max(
        fake_probability,
        real_probability
    )

    if confidence < 0.60:
        prediction = "UNCERTAIN"
    elif predicted_label == 1:
        prediction = "REAL NEWS"
    else:
        prediction = "FAKE NEWS"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
    }


def main():
    model, vectorizer = load_saved_files()

    print("=" * 55)
    print("Fake News Detection - Logistic Regression")
    print("=" * 55)
    print("Paste a complete news article.")
    print("Type 'exit' to close the application.")

    while True:
        print("\nEnter news text:")
        news_text = input("> ").strip()

        if news_text.lower() == "exit":
            print("Application closed.")
            break

        try:
            result = predict_news(
                news_text,
                model,
                vectorizer
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

        except ValueError as error:
            print("Error:", error)


if __name__ == "__main__":
    main()