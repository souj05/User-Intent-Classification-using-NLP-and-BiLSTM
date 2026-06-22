import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Intent Classifier",
    page_icon="🧠",
    layout="centered"
)

# ─── Load Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = load_model("intent_model.keras")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    le        = pickle.load(open("label_encoder.pkl", "rb"))
    return model, tokenizer, le

model, tokenizer, le = load_artifacts()
MAX_LEN = 30

# ─── UI ────────────────────────────────────────────────────────
st.title("🧠 User Intent Classifier")
st.markdown("**NLP + BiLSTM | CLINC150 | 151 Intents**")
st.markdown("---")

text = st.text_input("💬 Enter your message:", placeholder="e.g. What is the weather today?")

if st.button("🔍 Predict Intent") and text:
    seq    = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN, padding='post')
    probs  = model.predict(seq, verbose=0)[0]
    top3   = probs.argsort()[-3:][::-1]

    st.markdown("### 🎯 Prediction Results")

    st.success(f"**Top Intent:** `{le.classes_[top3[0]]}` — {round(probs[top3[0]]*100, 1)}%")
    st.info   (f"**2nd Intent:** `{le.classes_[top3[1]]}` — {round(probs[top3[1]]*100, 1)}%")
    st.warning(f"**3rd Intent:** `{le.classes_[top3[2]]}` — {round(probs[top3[2]]*100, 1)}%")

    st.markdown("---")
    st.markdown("#### 📊 Confidence Bar")
    st.progress(float(probs[top3[0]]))

st.markdown("---")
st.markdown("### 🧪 Try these examples:")
examples = [
    "What is the weather like today?",
    "Book me a flight to New York",
    "Transfer $500 to my savings account",
    "Play some jazz music",
    "Set a timer for 10 minutes"
]
for ex in examples:
    st.code(ex)

st.markdown("---")
st.markdown("👩‍💻 Developed by **Sowjanya Tadimarri** | Made with ❤️ using Streamlit")
