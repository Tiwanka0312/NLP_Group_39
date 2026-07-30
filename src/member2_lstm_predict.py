from pathlib import Path
import json
import pickle
import re
import sys

import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Run this program from the NLP_Group_39 project root folder
PROJECT_ROOT = Path.cwd()

MODEL_PATH = PROJECT_ROOT / "models" / "best_lstm_model.keras"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "lstm_tokenizer.pkl"
CONFIG_PATH = PROJECT_ROOT / "models" / "lstm_config.json"


def prepare_nltk_resources() -> None:
    """Download required NLTK resources only when unavailable."""

    required_resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, resource_name in required_resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


prepare_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Apply the same preprocessing used during LSTM training."""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation, numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = text.split()

    cleaned_tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOP_WORDS and len(word) > 1
    ]

    cleaned_text = " ".join(cleaned_tokens)

    return re.sub(r"\s+", " ", cleaned_text).strip()


def load_lstm_components():
    """Load the trained LSTM model, tokenizer and configuration."""

    required_files = [
        MODEL_PATH,
        TOKENIZER_PATH,
        CONFIG_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}\n"
                "Run this command from the NLP_Group_39 project root folder."
            )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    return model, tokenizer, config


def predict_news(
    title: str,
    article: str,
    model,
    tokenizer,
    config
) -> None:
    """Predict whether entered news is Fake or Real."""

    full_text = f"{title.strip()} {article.strip()}".strip()
    cleaned_news = clean_text(full_text)

    if not cleaned_news:
        print(
            "\nThe entered news does not contain enough "
            "valid English text."
        )
        return

    sequence = tokenizer.texts_to_sequences(
        [cleaned_news]
    )

    max_length = int(config["max_length"])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    real_probability = float(
        model.predict(
            padded_sequence,
            verbose=0
        )[0][0]
    )

    fake_probability = 1.0 - real_probability

    threshold = float(
        config.get("prediction_threshold", 0.5)
    )

    if real_probability >= threshold:
        prediction = "Real News"
        confidence = real_probability
    else:
        prediction = "Fake News"
        confidence = fake_probability

    print("\n" + "=" * 45)
    print("FAKE NEWS DETECTION RESULT — LSTM")
    print("=" * 45)
    print(f"Prediction       : {prediction}")
    print(f"Confidence       : {confidence * 100:.2f}%")
    print(f"Fake Probability : {fake_probability * 100:.2f}%")
    print(f"Real Probability : {real_probability * 100:.2f}%")
    print("=" * 45)


def main() -> None:
    try:
        model, tokenizer, config = load_lstm_components()
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"\nError: {error}")
        sys.exit(1)

    print("=" * 45)
    print("FAKE NEWS DETECTION — LSTM")
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

        predict_news(
            title,
            article,
            model,
            tokenizer,
            config
        )


if __name__ == "__main__":
    main()