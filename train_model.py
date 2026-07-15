"""
Trains the LinearRegression model used by main.py and saves it to model.pkl.

Run this again any time you have new/updated data:
    python train_model.py path/to/your_dataset.csv

Expects a CSV with these exact columns:
    Study_Hours, Previous_Marks, Sleep_Hours, Attendance, Final_Marks
"""

import sys
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DEFAULT_CSV = "student_marks_prediction_dataset_5000.csv"
FEATURES = ["Study_Hours", "Previous_Marks", "Sleep_Hours", "Attendance"]
TARGET = "Final_Marks"


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_CSV)
    if not csv_path.exists():
        print(f"Could not find {csv_path}. Pass a path: python train_model.py your_data.csv")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
    print(f"R^2: {r2_score(y_test, pred):.3f}")

    # Refit on the full dataset before saving, so the deployed model uses all available data.
    model.fit(X, y)
    out_path = Path(__file__).parent / "model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
