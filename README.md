# Student Score Card Prediction — Backend

`model.pkl` is already trained on `student_marks_prediction_dataset_5000.csv`
(scikit-learn LinearRegression, MAE ~4.1, R² ~0.89) — you can run this
immediately, no extra setup needed.

## Run it
```
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Open http://localhost:8000 — this serves the frontend (from `static/`) and
the `/predict` API together, so no CORS setup is needed.

## Retraining
If you get more/updated data, run:
```
python train_model.py path/to/your_dataset.csv
```
It expects columns `Study_Hours, Previous_Marks, Sleep_Hours, Attendance, Final_Marks`
and overwrites `model.pkl`.

## Notes
- `/predict` expects JSON matching `StudentInput` in `main.py`:
  `study_hours, previous_marks, sleep_hours, attendance`.
- `/health` is a simple check endpoint to confirm the server is running.
- The factor breakdown shown on the result screen is a display heuristic
  based on value ranges, not the model's real per-feature contribution —
  see the comment in `compute_factor_breakdown()` in `main.py` for how to
  compute genuine coefficients-based contributions instead.
