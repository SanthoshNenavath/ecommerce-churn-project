from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = FastAPI(title="E-Commerce Churn Prediction")

templates = Jinja2Templates(directory="app/templates")

# -------------------------------------------------
# Train model at startup
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "rfm_with_churn.csv"

df = pd.read_csv(DATA_PATH)

X = df[["Frequency", "Monetary"]]
y = df["Churn"]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

THRESHOLD = 0.5

# -------------------------------------------------
# Home page
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None
        }
    )

# -------------------------------------------------
# Prediction from form
# -------------------------------------------------
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    frequency: int = Form(...),
    monetary: float = Form(...)
):
    X = np.array([[frequency, monetary]])
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]
    decision = "YES" if prob >= THRESHOLD else "NO"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": {
                "frequency": frequency,
                "monetary": monetary,
                "probability": round(float(prob), 4),
                "decision": decision
            }
        }
    )
