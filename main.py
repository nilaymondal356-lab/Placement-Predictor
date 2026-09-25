from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
import joblib
import numpy as np
from typing import Literal
from pydantic import BaseModel
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load('placement_model.pkl')

class StudentData(BaseModel):
    CGPA : float
    Internships : int
    Projects : int
    WorkshopsCertifications: int
    AptitudeTestScore : int
    SoftSkillsRating: float
    ExtracurricularActivities : Literal[ 'Yes','No']
    PlacementTraining : Literal['Yes' , 'No']
    SSC_Marks : int
    HSC_Marks : int
# ── Static file routes ──────────────────────────────────────────────
@app.get('/')
def serve_index():
    return FileResponse(BASE_DIR / 'index.html')

@app.get('/style.css')
def serve_css():
    return FileResponse(BASE_DIR / 'style.css', media_type='text/css')

@app.get('/script.js')
def serve_js():
    return FileResponse(BASE_DIR / 'script.js', media_type='application/javascript')

@app.get('/hero-background.mp4')
def serve_video():
    return FileResponse(BASE_DIR / 'hero-background.mp4', media_type='video/mp4')

# ── Prediction endpoint ─────────────────────────────────────────────
@app.post('/predict')
def predict(student : StudentData):
    ExtracurricularActivities = 1 if student.ExtracurricularActivities == 'Yes' else 0
    PlacementTraining = 1 if student.PlacementTraining == 'Yes' else 0
    input_data = np.array([[
        student.CGPA,
        student.Internships,
        student.Projects,
        student.WorkshopsCertifications,
        student.AptitudeTestScore,
        student.SoftSkillsRating,
        ExtracurricularActivities,
        PlacementTraining,
        student.SSC_Marks,
        student.HSC_Marks
    ]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    return {"placement_status": "Placed" if prediction[0] == 1 else "Not Placed",
            "confidence": round(float(probability)*100,2)
            }