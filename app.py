import streamlit as st
import pandas as pd
import joblib
import re
import requests
from io import BytesIO
import nltk
from nltk.corpus import stopwords

# =========================
# SETUP
# =========================
nltk.download('stopwords')

st.set_page_config(
    page_title="Analisis Sentimen Roblox",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        background-color: #0f1115 !important;
        color: #e6e6e6 !important;
    }

    div[data-baseweb="textarea"] textarea {
        background-color: #1b1e24 !important;
        color: #e6e6e6 !important;
        border: 1px solid #555 !important;
    }

    div.stButton > button {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #555 !important;
    }

    .card {
        background-color: #1b1e24;
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 0 15px rgba(0,0,0,0.4);
    }

    .card-positive { border-left: 6px solid #4caf50; }
    .card-negative { border-left: 6px solid #b23b3b; }

    .conf-card {
        background: linear-gradient(180deg, #1b1e24, #16191f);
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }

    .conf-title {
        font-size: 14px;
        color: #9aa0a6;
        margin-bottom: 0.3rem;
    }

    .conf-value {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .conf-desc {
        font-size: 13px;
        color: #b0b0b0;
    }
        /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14161a, #0f1115) !important;
        border-right: 1px solid #2a2a2a !important;
        padding-top: 1.5rem;
    }

    /* Judul sidebar */
    .sidebar-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    /* Deskripsi sidebar */
    .sidebar-desc {
        font-size: 13px;
        color: #9aa0a6;
        margin-bottom: 1rem;
    }

    /* Radio menu */
    div[role="radiogroup"] > label {
        background-color: #1b1e24;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        border: 1px solid #2a2a2a;
        transition: all 0.2s ease-in-out;
    }

    /* Hover menu */
    div[role="radiogroup"] > label:hover {
        background-color: #22262d;
        border-color: #3a3f47;
    }

    /* Selected menu */
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #262b33;
        border-left: 4px solid #4caf50;
        font-weight: 600;
    }
    .method-card {
        background-color: #16191f;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }

    .method-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .method-text {
        font-size: 15px;
        line-height: 1.8;
        color: #d1d1d1;
    }

    .method-step {
        background-color: #1b1e24;
        border-left: 4px solid #4caf50;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.7rem;
    }

    .method-note {
        font-size: 14px;
        color: #b0b0b0;
    }
        /* ===== LOADING ANALYSIS ===== */
    .loading-box {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 1rem 1.2rem;
        background: #16191f;
        border-radius: 12px;
        border: 1px solid #2a2a2a;
        box-shadow: 0 0 20px rgba(0,0,0,0.35);
        margin-top: 1rem;
    }

    .loader {
        width: 26px;
        height: 26px;
        border: 3px solid #2a2a2a;
        border-top: 3px solid #4caf50;
        border-radius: 50%;
        animation: spin 0.9s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        font-size: 14px;
        color: #c7c7c7;
    }

    .highlight-box {
    background-color: #16191f;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}

    .highlight-title {
        font-size: 14px;
        color: #9aa0a6;
        margin-bottom: 0.6rem;
    }

    .word-pos {
        color: #4caf50;
        font-weight: 600;
    }

    .word-neg {
        color: #e57373;
        font-weight: 600;
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
# PREPROCESSING
# =========================
stop_words = set(stopwords.words('indonesian'))

# PENTING: JANGAN HAPUS KATA NEGASI
negation_words = {"tidak", "nggak", "ga", "gak", "bukan"}
stop_words = stop_words - negation_words


url_kamus = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
kamus_data = pd.read_excel(BytesIO(requests.get(url_kamus).content))
kamus = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))

negation_words = {"tidak", "nggak", "ga", "gak", "bukan"}

def preprocess(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    tokens = text.split()
    tokens = [kamus.get(w, w) for w in tokens]

    result_tokens = []
    skip_next = False

    for i in range(len(tokens)):
        if skip_next:
            skip_next = False
            continue

        # HANDLE NEGASI
        if tokens[i] in negation_words and i + 1 < len(tokens):
            negated_word = f"neg_{tokens[i+1]}"
            result_tokens.append(negated_word)
            skip_next = True
        else:
            if tokens[i] not in stop_words:
                result_tokens.append(tokens[i])

    return " ".join(result_tokens)


# =========================
# LEXICON (UNTUK HIGHLIGHT)
# =========================
custom_positive_game = {
    "bagus","keren","mantap","seru","asyik","lancar","smooth",
    "worthit","recommended","puas","stabil", "bagus", "keren", "mantap", "seru", "asyik",
    "enak", "lancar", "smooth", "ringan",
    "oke", "ok", "recommended", "rekomen",
    "worthit", "worth", "top", "best",
    "suka", "senang", "puas",
    "stabil", "halus",
    "grafisbagus", "grafiskeren",
    "rame", "ramai",
    "updatebagus", "makinbagus", "baik", "mudah"
}

custom_negative_game = {
    "burik","bug","lag","lemot","error","crash","ngehang",
    "parah","rusak","ampas","payah","delay","kecewa","neg_suka", "neg_bagus", "neg_ok", "neg_guna", "neg_keren", "neg_mantap", "neg_seru", "neg_asyik",
    "neg_enak", "neg_lancar", "neg_smooth", "neg_ringan",    "burik", "jelek", "jele", "jlek", "parah", "rusak", "patah",
    "bug", "buggy", "lag", "laggy", "lemot", "lelet", "ngelag"
    "error", "eror", "crash", "freeze", "ngehang",
    "ngelag", "delay", "forceclose",
    "ampas", "payah", "ngaco", "kacau", "acakadut",
    "bosen", "bosan", "ngebosenin", "kecewa",
    "nyebelin", "ngeselin", "ribet", "rempong",
    "gabisa", "gajelas", "ga_jelas",
    "dc", "disconnect", "serverdown", "maintenance",
    "mahal", "paytowin", "p2w", "iklan", "iklanbanyak",
    "hack", "cheat", "curang", "akunilang", "kehack", "haram", "bau", "tai", "berat", "lama",
    "muak", "gls", "uninstall", "cape", "capek"
}

def highlight_words(text):
    result = []
    for w in text.split():
        if w in custom_positive_game:
            result.append(f"<span style='color:#4caf50;font-weight:600'>{w}</span>")
        elif w in custom_negative_game:
            result.append(f"<span style='color:#e57373;font-weight:600'>{w}</span>")
        else:
            result.append(w)
    return " ".join(result)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #1f2933, #111827);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 0 25px rgba(0,0,0,0.4);
        margin-bottom: 2rem;
    ">
        <h1 style="margin:0; font-weight:800;">
            Analisis Sentimen Ulasan Game Roblox
        </h1>
        <p style="margin-top:6px; color:#c7c7c7;">
            Menggunakan Multinomial Naive Bayes dan SMOTE
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.markdown(
    """
    <div class="sidebar-title">📊 Dashboard</div>
    <div class="sidebar-desc">
        Analisis sentimen ulasan game Roblox
    </div>
    """,
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "",
    [
        "🎯 Prediksi Sentimen",
        "📂 Dataset",
        "📈 Visualisasi",
        "📘 Metodologi"
    ]
)

# =========================
# PREDIKSI
# =========================
if menu == "🎯 Prediksi Sentimen":
    st.subheader("Prediksi Sentimen Review")

    text = st.text_area(
        "Masukkan Review",
        height=170,
        placeholder="Contoh: game nya seru dan bagus banget!"
    )

    if st.button("🔍 Analisis Sentimen"):
        if text.strip() == "":
            st.warning("Review tidak boleh kosong.")
        else:
            loading_placeholder = st.empty()

            loading_placeholder.markdown(
                """
                <div class="loading-box">
                    <div class="loader"></div>
                    <div class="loading-text">Menganalisis sentimen, mohon tunggu...</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            import time
            time.sleep(0.6)  # ⬅️ INI KUNCI SUPAYA KELIATAN

            clean_text = preprocess(text)
            vector = tfidf.transform([clean_text])
            prediction = model.predict(vector)[0]
            proba = model.predict_proba(vector)[0]
            confidence = max(proba) * 100

            loading_placeholder.empty()
            card_class = "card-positive" if prediction == "Positif" else "card-negative"
            emoji = "😊" if prediction == "Positif" else "😞"

            st.markdown(
                f"""
                <div class="card {card_class}">
                    <h2>{emoji} {prediction} ({confidence:.2f}%)</h2>
                    <p>Hasil analisis sentimen berdasarkan model.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ===== HIGHLIGHT KATA =====
            st.markdown(
                f"""
                <div class="highlight-box">
                    <div class="highlight-title">🔎 Highlight Kata (Hasil Preprocessing)</div>
                    <div>{highlight_words(clean_text)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ===== CONFIDENCE INTERPRETATION =====
            if confidence >= 75:
                conf_text = "Model sangat yakin terhadap hasil prediksi."
            elif confidence >= 60:
                conf_text = "Model cukup yakin terhadap hasil prediksi."
            else:
                conf_text = "Model kurang yakin, hasil perlu ditinjau."

            st.markdown(
                f"""
                <div class="conf-card">
                    <div class="conf-title">Confidence Score</div>
                    <div class="conf-value">{confidence:.2f}%</div>
                    <div class="conf-desc">{conf_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(int(confidence))

# =========================
# DATASET
# =========================
elif menu == "📂 Dataset":
    st.subheader("Preview Dataset")
    preview_df = df[['steming_data', 'Sentiment']].copy()
    preview_df.columns = ['Review', 'Sentiment']
    st.dataframe(preview_df, use_container_width=True)

# =========================
# VISUALISASI
# =========================
elif menu == "📈 Visualisasi":
    st.subheader("Visualisasi Distribusi Sentimen")

    # =========================
    # METRIC SUMMARY
    # =========================
    total_data = len(df)
    total_pos = len(df[df["Sentiment"] == "Positif"])
    total_neg = len(df[df["Sentiment"] == "Negatif"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-size:14px;color:#9aa0a6;">Total Data</div>
                <div style="font-size:28px;font-weight:700;">{total_data}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="card card-positive">
                <div style="font-size:14px;color:#9aa0a6;">Sentimen Positif</div>
                <div style="font-size:28px;font-weight:700;">{total_pos}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="card card-negative">
                <div style="font-size:14px;color:#9aa0a6;">Sentimen Negatif</div>
                <div style="font-size:28px;font-weight:700;">{total_neg}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # DISTRIBUSI SENTIMEN
    # =========================
    st.markdown("### Distribusi Sentimen")

    sentiment_count = df["Sentiment"].value_counts().reset_index()
    sentiment_count.columns = ["Sentiment", "Jumlah"]
    sentiment_count["Persentase"] = (
        sentiment_count["Jumlah"] / sentiment_count["Jumlah"].sum() * 100
    ).round(2)

    st.dataframe(
        sentiment_count,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # BAR CHART (MINIMALIS)
    # =========================
    st.markdown("### Grafik Jumlah Sentimen")

    chart_df = sentiment_count.set_index("Sentiment")[["Jumlah"]]
    st.bar_chart(chart_df)

    # =========================
    # INTERPRETASI
    # =========================
    dominan = sentiment_count.iloc[0]["Sentiment"]

    st.info(
        f"""
        **Interpretasi:**
        Mayoritas ulasan pengguna cenderung bersentimen **{dominan}**.
        Hal ini menunjukkan persepsi pengguna terhadap game Roblox
        didominasi oleh sentimen tersebut berdasarkan dataset yang dianalisis.
        """
    )
    
# =========================
# METODOLOGI
# =========================
else:
    st.subheader("Metodologi Penelitian")

    # =========================
    # PENDAHULUAN
    # =========================
    st.markdown(
        """
        <div class="method-card">
            <div class="method-title">🎯 Tujuan Penelitian</div>
            <div class="method-text">
                Penelitian ini bertujuan untuk menganalisis sentimen ulasan pengguna
                terhadap game <strong>Roblox</strong> dengan mengklasifikasikan
                ulasan ke dalam dua kategori sentimen, yaitu <strong>positif</strong>
                dan <strong>negatif</strong>, menggunakan pendekatan
                <strong>machine learning</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="method-card">
            <div class="method-title">🔄 Alur Penelitian</div>
            <div class="method-text">
                Alur penelitian ini terdiri dari tahap-tahap sebagai berikut:
                <ul>
                    <li>Pengumpulan Data</li>
                    <li>Processing Data</li>
                    <li>Representasi Teks</li>
                    <li>Penyeimbangan Data</li>
                    <li>Klasifikasi Sentimen</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # MODEL
    # =========================
    st.markdown(
        """
        <div class="method-card">
            <div class="method-title">🤖 Model Klasifikasi</div>
            <div class="method-text">
                Algoritma <strong>Multinomial Naive Bayes</strong> dipilih karena
                memiliki performa yang baik dalam klasifikasi teks dan efisien
                dalam pengolahan data berbasis frekuensi kata.
                Model dilatih menggunakan data yang telah diseimbangkan
                dengan SMOTE untuk meningkatkan akurasi dan generalisasi.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # CONFIDENCE SCORE
    # =========================
    st.markdown(
        """
        <div class="method-card">
            <div class="method-title">📊 Confidence Score</div>
            <div class="method-text">
                Confidence score diperoleh dari probabilitas prediksi yang dihasilkan
                oleh model Multinomial Naive Bayes. Nilai ini menunjukkan tingkat
                keyakinan model terhadap hasil klasifikasi sentimen yang diberikan.
            </div>
            <div class="method-note">
                Semakin tinggi nilai confidence, semakin besar keyakinan model terhadap
                hasil prediksi yang dihasilkan.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # KETERBATASAN
    # =========================
    st.markdown(
        """
        <div class="method-card">
            <div class="method-title">⚠️ Keterbatasan Penelitian</div>
            <ul class="method-text">
                <li>Model belum mampu menangkap konteks kalimat kompleks atau sarkasme.</li>
                <li>Hasil analisis dipengaruhi oleh kualitas data dan proses pelabelan.</li>
                <li>Confidence score bersifat probabilistik, bukan kebenaran absolut.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="text-align:center; color:#6b7280; margin-top:2rem;">
            © 2025 Analisis Sentimen Roblox
        </div>
        """,
        unsafe_allow_html=True
    )