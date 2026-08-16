from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "svm_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "svm_tfidf_vectorizer.joblib"

UNCERTAIN_THRESHOLD = 0.60


def load_saved_files():
    """Load the saved SVM model and TF-IDF vectorizer."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SVM model file not found: {MODEL_PATH}"
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer file not found: {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_news(news_text, model, vectorizer):
    """Predict whether an unseen article is fake or real."""

    if not isinstance(news_text, str):
        raise TypeError("News text must be a string.")

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

    fake_probability = float(
        probabilities[class_positions[0]]
    )

    real_probability = float(
        probabilities[class_positions[1]]
    )

    confidence = max(
        fake_probability,
        real_probability
    )

    if confidence < UNCERTAIN_THRESHOLD:
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
    try:
        model, vectorizer = load_saved_files()
    except (FileNotFoundError, OSError) as error:
        print("Model loading error:", error)
        return

    print("=" * 55)
    print("Fake News Detection - Support Vector Machine")
    print("=" * 55)
    print("Model loaded successfully.")
    print("Type 'exit' to close the application.")

    while True:
        news_text = input("\nEnter news text:\n> ").strip()

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

        except (TypeError, ValueError) as error:
            print("Prediction error:", error)


if __name__ == "__main__":
    main()