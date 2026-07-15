"""
FastAPI backend for the Student Score Card Prediction frontend.

Trained on student_marks_prediction_dataset_5000.csv:
  features -> Study_Hours, Previous_Marks, Sleep_Hours, Attendance
  target   -> Final_Marks
Model: scikit-learn LinearRegression (MAE ~4.1, R^2 ~0.89 on a held-out split).
model.pkl in this folder is already trained and ready to use.

To retrain on new/updated data, see train_model.py in this folder.

Run it:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000
Then open http://localhost:8000 — this serves the frontend (static/) and
the /predict API from the same origin, so no CORS setup is needed.
"""

import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.pkl"

# Matches the columns in student_marks_prediction_dataset_5000.csv, in
# training order: Study_Hours, Previous_Marks, Sleep_Hours, Attendance.
FEATURE_ORDER = [
    "study_hours",
    "previous_marks",
    "sleep_hours",
    "attendance",
]


class StudentInput(BaseModel):
    study_hours: float = Field(..., ge=0, le=12)
    previous_marks: float = Field(..., ge=0, le=100)
    sleep_hours: float = Field(..., ge=0, le=10)
    attendance: float = Field(..., ge=0, le=100)


class PredictionResponse(BaseModel):
    predicted_score: float
    factors: dict


app = FastAPI(title="Student Score Card Prediction API")

# Allow the frontend (possibly on a different port during dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Place your trained model there as 'model.pkl'."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def compute_factor_breakdown(payload: StudentInput) -> dict:
    """
    Rough contribution breakdown for the UI's factor bars, scaled against the
    dataset's observed ranges (Study_Hours 0.5-12, Sleep_Hours 4-10).
    This is a display heuristic, not the model's real feature importance.
    For a real LinearRegression model, you can compute true per-prediction
    contributions as `coef_[i] * value_i` instead.
    """
    return {
        "study": round(min(payload.study_hours, 12) / 12 * 100),
        "attendance": round(payload.attendance),
        "previous": round(payload.previous_marks),
        "wellbeing": round(min(payload.sleep_hours, 10) / 10 * 100),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StudentInput):
    model = load_model()

    row = [[getattr(payload, name) for name in FEATURE_ORDER]]

    try:
        prediction = model.predict(row)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}")

    predicted_score = max(0, min(100, round(float(prediction), 1)))

    return PredictionResponse(
        predicted_score=predicted_score,
        factors=compute_factor_breakdown(payload),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# Serves index.html directly from this same server at http://localhost:8000/
# Put index.html in a folder called "static" next to this file to use this.
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
