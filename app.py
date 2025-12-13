import streamlit as st
import pandas as pd
import joblib
import re
import requests
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================
# SETUP
# =========================
nltk.download('stopwords')

st.set_page_config(
    page_title="Analisis Sentimen Roblox",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* ================= FORCE GLOBAL ================= */
    html, body, [class*="css"] {
        background-color: #0f1115 !important;
        color: #e6e6e6 !important;
    }

    /* ================= TEXT AREA (INI SUMBER MERAH) ================= */
    div[data-baseweb="textarea"] textarea {
        background-color: #1b1e24 !important;
        color: #e6e6e6 !important;
        border: 1px solid #555 !important;
    }

    div[data-baseweb="textarea"] textarea:focus {
        border-color: #777 !important;
        box-shadow: 0 0 0 2px rgba(120,120,120,0.6) !important;
        outline: none !important;
    }

    /* ================= BUTTON ================= */
    div.stButton > button {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #555 !important;
    }

    div.stButton > button:hover {
        background-color: #2a2a2a !important;
        border-color: #777 !important;
    }

    div.stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(120,120,120,0.6) !important;
        outline: none !important;
    }

    /* ================= SIDEBAR ================= */
    section[data-testid="stSidebar"] {
        background-color: #14161a !important;
        border-right: 1px solid #2a2a2a !important;
    }

    /* ================= CARD ================= */
    .card {
        background-color: #1b1e24 !important;
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 0 15px rgba(0,0,0,0.4);
    }

    .card-positive {
        border-left: 6px solid #4caf50 !important;
    }

    .card-negative {
        border-left: 6px solid #b23b3b !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# LOAD MODEL & DATA
# =========================
model = joblib.load("mnb_sm_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")
df = pd.read_csv("dataset.csv")

# =========================
# PREPROCESSING (INPUT USER)
# =========================
stop_words = stopwords.words('indonesian')

url_kamus = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
kamus_data = pd.read_excel(BytesIO(requests.get(url_kamus).content))
kamus = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))

def preprocess(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    tokens = text.split()
    tokens = [kamus.get(word, word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

# =========================
# HEADER
# =========================
st.title("🎮 Analisis Sentimen Review Game Roblox")
st.caption("Metode: Multinomial Naive Bayes + SMOTE")

menu = st.sidebar.selectbox(
    "Menu",
    ["Prediksi Sentimen", "Dataset", "Visualisasi", "Metodologi"]
)

# =========================
# PREDIKSI
# =========================
if menu == "Prediksi Sentimen":
    st.subheader("Prediksi Sentimen Review")

    text = st.text_area(
        "Masukkan review game Roblox:",
        height=150,
        placeholder="Contoh: gamenya bagus"
    )

    if st.button("Prediksi"):
        if text.strip() == "":
            st.warning("Teks tidak boleh kosong.")
        else:
            clean_text = preprocess(text)

            if len(clean_text.split()) < 2:
                st.warning("Masukkan minimal 2 kata agar prediksi lebih akurat.")
            else:
                vector = tfidf.transform([clean_text])
                prediction = model.predict(vector)[0]

                if prediction == "Positif":
                    st.markdown(
                        """
                        <div class="card card-positive">
                            <h2>✅ Sentimen POSITIF</h2>
                            <p>Review menunjukkan respon positif terhadap game Roblox.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div class="card card-negative">
                            <h2>❌ Sentimen NEGATIF</h2>
                            <p>Review menunjukkan respon negatif terhadap game Roblox.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# =========================
# DATASET (PAGINATION)
# =========================
elif menu == "Dataset":
    st.subheader("Preview Dataset")

    preview_df = df[['steming_data', 'Sentiment']].copy()
    preview_df.columns = ['Review', 'Sentiment']

    page_size = 100
    total_pages = (len(preview_df) // page_size) + 1

    if "page" not in st.session_state:
        st.session_state.page = 1

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Prev") and st.session_state.page > 1:
            st.session_state.page -= 1
    with col3:
        if st.button("Next ➡️") and st.session_state.page < total_pages:
            st.session_state.page += 1

    start = (st.session_state.page - 1) * page_size
    end = start + page_size

    st.dataframe(preview_df.iloc[start:end], use_container_width=True)
    st.caption(f"Halaman {st.session_state.page} dari {total_pages}")

# =========================
# VISUALISASI
# =========================
elif menu == "Visualisasi":
    st.subheader("Visualisasi Distribusi Sentimen")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", len(df))
    col2.metric("Positif", len(df[df['Sentiment'] == 'Positif']))
    col3.metric("Negatif", len(df[df['Sentiment'] == 'Negatif']))

    st.bar_chart(df['Sentiment'].value_counts())

# =========================
# METODOLOGI
# =========================
else:
    st.subheader("Metodologi Penelitian")
    st.markdown("""
    1. Scraping review game Roblox dari Google Play Store  
    2. Preprocessing teks (cleaning, normalisasi, stopword, stemming)  
    3. Pembobotan kata menggunakan TF-IDF  
    4. Penyeimbangan data menggunakan SMOTE  
    5. Klasifikasi menggunakan Multinomial Naive Bayes  
    """)

    st.info(
        "Aplikasi menggunakan tema gelap akademik untuk meningkatkan kenyamanan visual "
        "dan hasil prediksi ditampilkan dalam bentuk card agar mudah dipahami."
    )
