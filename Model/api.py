# We create a API by packing our trained model using FastAPI

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Loading the pickle file that contains our Trained ML Model
with open("ml_model.pkl","rb") as f:
    model = pickle.load(f)

# Creating the main app for our API
app = FastAPI()

# We define the data model using Pydantic
# We have refered the dataset to create the base model
class DiabetesInput(BaseModel):
    Age : int
    Gender : int
    BMI : float
    SBP : int
    DBP : int
    FPG : float
    Chol : float
    Tri : float
    HDL : float
    LDL : float
    ALT : float
    BUN : float
    CCR : float
    FFPG : float
    smoking : int
    drinking : int
    family_histroy : int

# Now we define the route handler for our API
@app.post("/predict")
# Argument handler when our API is called fron Node
def predict_diabetes(data : DiabetesInput):
    # We will be passing this array to the model as input for prediction
    # This will be passed to the API as JSON from Electron
    features = np.array([[
        data.Age,
        data.Gender,
        data.BMI,
        data.SBP,
        data.DBP,
        data.FPG,
        data.Chol,
        data.Tri,
        data.HDL,
        data.LDL,
        data.ALT,
        data.BUN,
        data.CCR,
        data.FFPG,
        data.smoking,
        data.drinking,
        data.family_histroy
    ]])
    print(features.shape)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    diabetes_probabilty = probability[1]*100
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"

    return{
        "Prediction":int(prediction),
        "Result": result,
        "Probability": int(diabetes_probabilty)
}



