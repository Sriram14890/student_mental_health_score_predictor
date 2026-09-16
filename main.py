import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware


# Load trained model
model = joblib.load("mental_health_model.pkl")

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_headers = ['*'],
    allow_origins = ['*'],
    allow_methods = ['*'],
)


# =========================
# Input Data Model
# =========================

class StudentData(BaseModel):

    age: int = Field(..., ge=10, le=100)

    gender: Literal["Male", "Female"]

    country: str

    academic_level: Literal[
        "Undergraduate",
        "Graduate",
        "High School"
    ]

    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]

    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]

    avg_daily_usage_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    daily_unlocks: int = Field(
        ...,
        ge=0
    )

    study_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    physical_activity_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    sleep_hours_per_night: float = Field(
        ...,
        ge=0,
        le=24
    )

    stress_level: Literal[
        "Medium",
        "Low",
        "Very High",
        "High"
    ]


# =========================
# Output Data Model
# =========================

class PredictionResponse(BaseModel):

    predicted_mental_health_score: float


# =========================
# Home Route
# =========================

@app.get("/")
def greet():

    return "Welcome to Student Performance Tracker"


# =========================
# Prediction Route
# =========================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: StudentData):

    try:

        # Countries used during model training
        top_countries = [
            "Other",
            "India",
            "USA",
            "Canada",
            "Australia",
            "UK",
            "Germany",
            "Mexico",
            "Turkey",
            "France"
        ]

        # Group country
        country_group = (
            data.country
            if data.country in top_countries
            else "Other"
        )

        # Create DataFrame
        input_row = pd.DataFrame([{

            "age": data.age,

            "gender": data.gender,

            "country": data.country,

            "academic_level": data.academic_level,

            "most_used_platform": data.most_used_platform,

            "purpose_of_use": data.purpose_of_use,

            "avg_daily_usage_hours": data.avg_daily_usage_hours,

            "daily_unlocks": data.daily_unlocks,

            "study_hours": data.study_hours,

            "physical_activity_hours":
                data.physical_activity_hours,

            "sleep_hours_per_night":
                data.sleep_hours_per_night,

            "stress_level": data.stress_level,

            "grouped_country": country_group

        }])


        # Print input for debugging
        print("\n==============================")
        print("INPUT DATA")
        print("==============================")
        print(input_row)

        print("\nCOLUMNS:")
        print(input_row.columns.tolist())


        # Make prediction
        prediction = model.predict(input_row)[0]


        # Print prediction
        print("\n==============================")
        print("PREDICTION")
        print("==============================")
        print(prediction)


        # Return prediction
        return PredictionResponse(
            predicted_mental_health_score=
                round(float(prediction), 2)
        )


    except Exception as e:

        # Print complete error in terminal
        import traceback

        print("\n==============================")
        print("ERROR")
        print("==============================")

        traceback.print_exc()

        # Send error to Swagger
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )