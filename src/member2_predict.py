from pathlib import Path
import re
import sys

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Find the project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "naive_bayes_bundle.joblib"
)


def prepare_nltk_resources() -> None:
    """Make sure required NLTK resources are available."""

    required_resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, download_name in required_resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


prepare_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Apply the same preprocessing used during model training."""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove numbers, punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenization, stop-word removal and lemmatization
    tokens = text.split()

    cleaned_tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOP_WORDS and len(word) > 1
    ]

    cleaned_text = " ".join(cleaned_tokens)

    return re.sub(r"\s+", " ", cleaned_text).strip()


def load_model_bundle():
    """Load the trained model and TF-IDF vectorizer."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "vectorizer",
        "label_mapping",
    }

    missing_keys = required_keys.difference(bundle.keys())

    if missing_keys:
        raise KeyError(
            f"Model bundle is missing: {missing_keys}"
        )

    return bundle


def predict_news(
    title: str,
    article: str,
    bundle
) -> None:
    """Predict whether the supplied news is fake or real."""

    full_text = f"{title.strip()} {article.strip()}".strip()
    cleaned_news = clean_text(full_text)

    if not cleaned_news:
        print(
            "\nThe entered news does not contain enough "
            "valid English text."
        )
        return

    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    label_mapping = bundle["label_mapping"]

    news_features = vectorizer.transform([cleaned_news])

    prediction = int(model.predict(news_features)[0])
    probabilities = model.predict_proba(news_features)[0]

    predicted_class = label_mapping[prediction]

    fake_probability = float(probabilities[0]) * 100
    real_probability = float(probabilities[1]) * 100
    confidence = float(probabilities[prediction]) * 100

    print("\n" + "=" * 45)
    print("FAKE NEWS DETECTION RESULT")
    print("=" * 45)
    print(f"Prediction       : {predicted_class}")
    print(f"Confidence       : {confidence:.2f}%")
    print(f"Fake Probability : {fake_probability:.2f}%")
    print(f"Real Probability : {real_probability:.2f}%")
    print("=" * 45)


def main() -> None:
    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, KeyError, OSError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    print("=" * 45)
    print("FAKE NEWS DETECTION — NAIVE BAYES")
    print("=" * 45)
    print("Enter an English news title and article.")
    print("Type 'exit' as the title to close the program.")

    while True:
        title = input("\nEnter news title: ").strip()

        if title.lower() == "exit":
            print("Program closed.")
            break

        article = input("Enter news article: ").strip()

        if not title and not article:
            print("Please enter a news title or article.")
            continue

        predict_news(title, article, bundle)

if __name__ == "__main__":
    main()