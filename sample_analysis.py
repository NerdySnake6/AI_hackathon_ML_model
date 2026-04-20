"""Run a small local demo of the query classification pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from solution import PredictionModel

ROOT = Path(__file__).resolve().parent
LOCAL_DATA_CANDIDATES = (
    ROOT / "train.csv",
    ROOT / "data" / "train.csv",
)

SAMPLE_QUERIES = [
    "купить iphone 15 pro цена",
    "рецепт борща с фото",
    "рик и морти 7 сезон",
    "смотреть чебурашка онлайн",
    "новинки сериалов 2024",
    "гарри потер и философский камень",
]


def load_demo_frame() -> pd.DataFrame:
    """Load a small demo frame from local data or a synthetic fallback."""
    for path in LOCAL_DATA_CANDIDATES:
        if path.exists():
            frame = pd.read_csv(path)
            if "QueryText" not in frame.columns:
                raise ValueError(f"Expected 'QueryText' column in {path}")
            return frame[["QueryText"]].head(10).copy()

    return pd.DataFrame({"QueryText": SAMPLE_QUERIES})


def main() -> None:
    """Run the local prediction demo and print the resulting rows."""
    frame = load_demo_frame()
    model = PredictionModel()
    predictions = model.predict(frame)

    print("Demo predictions:")
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
