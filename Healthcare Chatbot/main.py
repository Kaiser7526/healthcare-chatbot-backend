import json
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import Chat
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from fastapi.middleware.cors import CORSMiddleware

# ✅ Import dependencies
from database import SessionLocal

# ✅ Create a database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()

# ✅ Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (can be restricted)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# --- Load the Trained ML Model ---
model = joblib.load("symptom_model.pkl")

# --- Load the Full List of Symptoms (Features) ---
with open("features.json", "r") as f:
    full_feature_list = json.load(f)
print(f"✅ Loaded feature list (Symptom count: {len(full_feature_list)})")

# ✅ Load Disease-Symptom Dataset
df = pd.read_csv("symptom_disease_expanded.csv")  # This file should match the new dataset format

# ✅ Load SpaCy's NLP Model
nlp = spacy.load("en_core_web_sm")

# ✅ Setup NLP Matcher for Symptom Extraction
matcher = PhraseMatcher(nlp.vocab)

# Function to preprocess text (replace spaces with underscores)
def preprocess_text(text: str):
    """
    Preprocess the input text by replacing spaces with underscores for multi-word symptoms.
    """
    return text.replace(" ", "_").lower()  # Replace spaces with underscores and lowercase the text

# Function to extract symptoms from input text
def extract_symptoms(text: str):
    """Extract symptoms from free-text input using NLP."""
    # Preprocess the text (for better matching with the symptom list)
    # text = preprocess_text(text)
    text = text.lower()
    # Process the text using SpaCy
    doc = nlp(text)
    print(f"Processed Text: {doc}")

    # Initialize an empty set for extracted symptoms
    extracted_symptoms = set()

    # Convert symptoms in full_feature_list to lowercase for case-insensitive matching
    lower_feature_list = [symptom.lower() for symptom in full_feature_list]

    # Create patterns only for symptoms that exist in the input text
    patterns = [nlp.make_doc(symptom) for symptom in lower_feature_list if symptom in text]

    # Add patterns for symptoms that match the input text
    # patterns = [nlp.make_doc(symptom) for symptom in full_feature_list if symptom.lower() in text]

    # Debugging: Print what patterns are being added
    print("Matching Patterns:", [pattern.text for pattern in patterns])

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    matcher.add("SYMPTOM", patterns)  # Add the patterns for the matching symptoms

    # Use the matcher to find symptom matches
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        extracted_symptoms.add(span.text.lower())  # Add to the set of extracted symptoms
    
    print(f"Extracted Symptoms: {extracted_symptoms}")
    return list(extracted_symptoms)

# ✅ API Model for Free-Text Input
class FreeTextInput(BaseModel):
    text: str

# ✅ API Model for Structured Symptom Input
class SymptomInput(BaseModel):
    symptoms: list  # User provides a list of symptoms

@app.post("/predict_from_text")
def predict_from_text(input_data: FreeTextInput):
    """Process free-text input, extract symptoms, and predict disease."""
    extracted_symptoms = extract_symptoms(input_data.text)

    # ✅ If no symptoms are found, return a chatbot response instead of using the model
    if not extracted_symptoms:
        print('extracted symptoms not available')
        return {"response": "Hello! I am a healthcare chatbot. How can I assist you today?"}

    # ✅ Create the input vector for the model
    input_vector = np.zeros(len(full_feature_list))
    full_feature_list_lower = [symptom.lower() for symptom in full_feature_list]

    for symptom in extracted_symptoms:
        symptom = symptom.lower()
        if symptom in full_feature_list_lower:
            index = full_feature_list_lower.index(symptom)
            input_vector[index] = 1

    # ✅ Convert to DataFrame to match training format
    input_df = pd.DataFrame([input_vector], columns=full_feature_list)

    # ✅ Predict disease
    predicted_disease = model.predict(input_df)[0]

    return {"extracted_symptoms": extracted_symptoms, "predicted_disease": predicted_disease}

# --- 🔹 PREDICTION ENDPOINT FOR SYMPTOM LIST INPUT ---
@app.post("/predict_symptoms")
def predict_symptoms(user_input: SymptomInput):
    """Process structured symptom list input and predict disease."""
    input_vector = [1 if symptom in user_input.symptoms else 0 for symptom in full_feature_list]
    
    # Ensure vector has correct length
    if len(input_vector) != len(full_feature_list):
        return {"error": "Mismatch in input vector length."}

    input_array = np.array(input_vector).reshape(1, -1)

    # Predict disease
    prediction = model.predict(input_array)

    return {"predicted_disease": prediction[0]}

# ✅ Store Chat Messages in Database
@app.post("/chatbot")
def chatbot(user_input: FreeTextInput, db: Session = Depends(get_db)):
    # ✅ Extract symptoms only from the current message (ignore past messages)
    extracted_symptoms = extract_symptoms(user_input.text)

    print("🔍 Extracted Symptoms:", extracted_symptoms)

    # ✅ If no symptoms found, return a default response
    if not extracted_symptoms:
        return {"response": "Hello! I am a healthcare chatbot. Please describe your symptoms."}

    # ✅ Convert extracted symptoms into input vector for ML model
    input_vector = [1 if symptom in extracted_symptoms else 0 for symptom in full_feature_list]
    input_df = pd.DataFrame([input_vector], columns=full_feature_list)

    # ✅ Predict disease based on symptoms
    predicted_disease = model.predict(input_df)[0]

    # ✅ Save chat history to database
    chat_entry = Chat(user_message=user_input.text, bot_response=predicted_disease)
    db.add(chat_entry)
    db.commit()

    return {"extracted_symptoms": extracted_symptoms, "predicted_disease": predicted_disease}

@app.post("/debug_symptoms")
def debug_symptoms(user_input: FreeTextInput):
    extracted_symptoms = extract_symptoms(user_input.text)
    return {"Extracted Symptoms": extracted_symptoms}

# ✅ Retrieve Chat History from Database
@app.get("/history")
def get_chat_history(db: Session = Depends(SessionLocal)):
    """Fetch chat history from the database."""
    chats = db.query(Chat).all()
    return [{"id": chat.id, "user_message": chat.user_message, "bot_response": chat.bot_response} for chat in chats]

# ✅ Create Database Tables
Base.metadata.create_all(bind=engine)
