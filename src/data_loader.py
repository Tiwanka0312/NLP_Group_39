from pathlib import Path

import pandas as pd


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data"

FAKE_DATASET = DATA_FOLDER / "Fake.csv"
TRUE_DATASET = DATA_FOLDER / "True.csv"


def load_news_data() -> pd.DataFrame:
    """Load, label, combine, and shuffle the fake and real news datasets."""

    if not FAKE_DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {FAKE_DATASET}")

    if not TRUE_DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {TRUE_DATASET}")

    fake_data = pd.read_csv(FAKE_DATASET)
    true_data = pd.read_csv(TRUE_DATASET)

    required_columns = {"title", "text"}

    if not required_columns.issubset(fake_data.columns):
        raise ValueError(
            f"Fake.csv must contain these columns: {required_columns}"
        )

    if not required_columns.issubset(true_data.columns):
        raise ValueError(
            f"True.csv must contain these columns: {required_columns}"
        )

    # 0 = Fake News, 1 = Real News
    fake_data["label"] = 0
    true_data["label"] = 1

    # Handle missing title and text values
    for dataframe in (fake_data, true_data):
        dataframe["title"] = dataframe["title"].fillna("").astype(str)
        dataframe["text"] = dataframe["text"].fillna("").astype(str)

        # Combine headline and article text
        dataframe["content"] = (
            dataframe["title"].str.strip()
            + " "
            + dataframe["text"].str.strip()
        ).str.strip()

    combined_data = pd.concat(
        [fake_data, true_data],
        ignore_index=True,
    )

    # Remove empty and duplicate articles
    combined_data = combined_data[
        combined_data["content"].str.len() > 0
    ]

    combined_data = combined_data.drop_duplicates(
        subset=["content"]
    )

    # Shuffle the dataset
    combined_data = combined_data.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    return combined_data


if __name__ == "__main__":
    dataset = load_news_data()

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {dataset.shape}")

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nClass distribution:")
    print(dataset["label"].value_counts())

    print("\nMissing values:")
    print(dataset[["title", "text", "content", "label"]].isnull().sum())

    print("\nFirst five records:")
    print(dataset[["title", "label"]].head())