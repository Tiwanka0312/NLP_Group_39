import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean a single news article.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove HTML tags
    5. Remove numbers
    6. Remove punctuation and special characters
    7. Remove extra spaces
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Keep only English letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the complete dataset.
    """

    required_columns = {"content", "label"}

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    cleaned_data = dataframe.copy()

    print("Starting text preprocessing...")

    cleaned_data["clean_content"] = cleaned_data["content"].apply(clean_text)

    # Remove records with empty text after cleaning
    cleaned_data = cleaned_data[
        cleaned_data["clean_content"].str.strip() != ""
    ].copy()

    cleaned_data = cleaned_data.reset_index(drop=True)

    print("Text preprocessing completed.")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")

    return cleaned_data


if __name__ == "__main__":
    sample_news = """
    BREAKING NEWS!!! Visit https://example.com for more information.
    The Government announced 500 new projects on 25/07/2026.
    """

    cleaned_news = clean_text(sample_news)

    print("Original text:")
    print(sample_news)

    print("\nCleaned text:")
    print(cleaned_news)