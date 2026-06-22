from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Intent Classifier API", version="1.0")

# Load artifacts on startup
model     = load_model("intent_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
le        = pickle.load(open("label_encoder.pkl", "rb"))
MAX_LEN   = 30

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Intent Classifier API is running!"}

@app.post("/predict")
def predict(data: TextInput):
    seq   = pad_sequences(tokenizer.texts_to_sequences([data.text]), maxlen=MAX_LEN, padding='post')
    probs = model.predict(seq, verbose=0)[0]
    top3  = probs.argsort()[-3:][::-1]
    return {
        "input": data.text,
        "top1": {"intent": le.classes_[top3[0]], "confidence": round(float(probs[top3[0]])*100, 1)},
        "top2": {"intent": le.classes_[top3[1]], "confidence": round(float(probs[top3[1]])*100, 1)},
        "top3": {"intent": le.classes_[top3[2]], "confidence": round(float(probs[top3[2]])*100, 1)},
    }
