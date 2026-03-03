# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


def load_avocado_data(file_path: str = "") -> pd.DataFrame:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "vakhariapujan/avocado-prices-and-sales-volume-2015-2023",
        file_path,
    )
    return df


def clean_avocado_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.drop_duplicates()

    numeric_cols = [
        col
        for col in df.columns
        if col not in {"date", "type", "region", "year"}
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

    if "average_price" in df.columns:
        df = df[df["average_price"].notna()]

    df = df.reset_index(drop=True)
    return df


def main() -> None:
    file_path = ""
    df = load_avocado_data(file_path)
    df_clean = clean_avocado_data(df)

    print("First 5 records (raw):")
    print(df.head())
    print("\nFirst 5 records (cleaned):")
    print(df_clean.head())


if __name__ == "__main__":
    main()
