import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess  # atau copy langsung fungsi di atas

# Load model & vectorizer
model = joblib.load("mnb_sm_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

# Load dataset
df = pd.read_csv("dataset.csv")

st.set_page_config(page_title="Analisis Sentimen Roblox", layout="wide")
st.title("🎮 Analisis Sentimen Review Game Roblox")
st.caption("Metode: Multinomial Naive Bayes + SMOTE")

menu = st.sidebar.selectbox(
    "Menu",
    ["Prediksi Sentimen", "Dataset", "Visualisasi", "Metodologi"]
)

# =====================
# PREDIKSI
# =====================
if menu == "Prediksi Sentimen":
    st.subheader("Prediksi Sentimen Review")
    text = st.text_area("Masukkan review:", height=150)

    if st.button("Prediksi"):
        if text.strip() == "":
            st.warning("Teks tidak boleh kosong.")
        else:
            clean = preprocess(text)
            vec = tfidf.transform([clean])
            pred = model.predict(vec)[0]

            if pred == "Positif":
                st.success("Sentimen: POSITIF")
            else:
                st.error("Sentimen: NEGATIF")

# =====================
# DATASET
# =====================
elif menu == "Dataset":
    st.subheader("Preview Dataset")
    st.dataframe(df.head(50))
    st.markdown(f"**Total data:** {len(df)}")

# =====================
# VISUALISASI
# =====================
elif menu == "Visualisasi":
    st.subheader("Distribusi Sentimen")
    st.bar_chart(df['Sentiment'].value_counts())

# =====================
# METODOLOGI
# =====================
else:
    st.markdown("""
    **Tahapan Penelitian:**
    1. Scraping review game Roblox dari Google Play
    2. Preprocessing teks (cleaning, normalisasi, stopword, stemming)
    3. Pembobotan kata menggunakan TF-IDF
    4. Penyeimbangan data menggunakan SMOTE
    5. Klasifikasi sentimen menggunakan Multinomial Naive Bayes
    """)
