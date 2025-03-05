from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from io import StringIO
from preprocessing import clean_data, feature_engineering  # Import preprocessing functions

app = FastAPI()

# Load the trained model
model = joblib.load("model.pkl")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the CSV file content
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Apply preprocessing steps
        df = clean_data(df)  # Clean the data (drop missing values)
        df = feature_engineering(df)  # Standardize the features

        # Ensure column names match model features
        # Define expected feature columns based on your dataset
        expected_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        df = df[expected_features]

        # Make predictions
        predictions = model.predict(df)

        return {"predicted_price": predictions.tolist()}

    except Exception as e:
        return {"error": str(e)}
