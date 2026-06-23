import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob

st.set_page_config(page_title="Intent + Emotion Classifier", page_icon="🧠", layout="centered")

@st.cache_resource
def load_artifacts():
    model     = load_model("intent_model.keras")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    le        = pickle.load(open("label_encoder.pkl", "rb"))
    return model, tokenizer, le

model, tokenizer, le = load_artifacts()
MAX_LEN = 30

st.title("🧠 Intent + Emotion Classifier")
st.markdown("**NLP + BiLSTM + Sentiment Analysis | CLINC150 | 151 Intents**")
st.markdown("---")

text = st.text_input("💬 Enter your message:", placeholder="e.g. What is the weather today?")

if st.button("🔍 Analyze") and text:

    seq    = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_LEN, padding='post')
    probs  = model.predict(seq, verbose=0)[0]
    top3   = probs.argsort()[-3:][::-1]

    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        sentiment = "😊 Positive"
        sentiment_msg = "User seems happy and positive!"
    elif polarity < -0.1:
        sentiment = "😔 Negative"
        sentiment_msg = "User seems frustrated or sad!"
    else:
        sentiment = "😐 Neutral"
        sentiment_msg = "User seems calm and neutral!"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Intent Prediction")
        st.success("Top: " + le.classes_[top3[0]] + " — " + str(round(probs[top3[0]]*100, 1)) + "%")
        st.info("2nd: " + le.classes_[top3[1]] + " — " + str(round(probs[top3[1]]*100, 1)) + "%")
        st.warning("3rd: " + le.classes_[top3[2]] + " — " + str(round(probs[top3[2]]*100, 1)) + "%")
        st.markdown("#### 📊 Confidence")
        st.progress(float(probs[top3[0]]))

    with col2:
        st.markdown("### ❤️ Emotion Analysis")
        st.metric("Sentiment", sentiment)
        st.metric("Polarity Score", round(polarity, 2))
        st.info(sentiment_msg)

st.markdown("---")
st.markdown("### 🧪 Try these examples:")
st.code("What is the weather like today?")
st.code("I am so happy to book a flight to New York!")
st.code("I hate waiting, transfer my money now!")
st.code("Play some relaxing jazz music please")
st.code("Set a timer for 10 minutes")

st.markdown("---")
st.markdown("👩‍💻 Developed by **Sowjanya Tadimarri** | Made with ❤️ using Streamlit")
