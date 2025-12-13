import streamlit as st
import joblib
import re

# Load model
nb_model = joblib.load("naive_bayes.pkl")
tfidf = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

st.set_page_config(page_title="Analisis Sentimen Roblox", layout="centered")

st.title("🎮 Analisis Sentimen Review Game Roblox")
st.write("Metode: Naive Bayes + SMOTE")

user_input = st.text_area(
    "Masukkan review game Roblox:",
    height=150
)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        clean_text = preprocess(user_input)
        vector = tfidf.transform([clean_text])
        prediction = nb_model.predict(vector)
        label = label_encoder.inverse_transform(prediction)

        st.success(f"Hasil Sentimen: **{label[0]}**")
