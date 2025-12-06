# import streamlit as st
# import pickle
# import re
#
# # ---------------- PAGE SETTINGS ---------------- #
# st.set_page_config(
#     page_title="Fake News Detector",
#     layout="centered"
# )
#
# st.title("📰 Fake News Detection")
# st.write("Paste a news article below and check whether it is real or fake.")
#
# # ---------------- LOAD MODEL & VECTORIZER ---------------- #
# with open("best_model.pkl", "rb") as f:
#     model = pickle.load(f)
#
# with open("vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)
#
# # ---------------- TEXT PREPROCESSING ---------------- #
# def wordopt(text):
#     text = text.lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'<.*?>+', '', text)
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()
#
# # ---------------- INPUT ---------------- #
# news = st.text_area(
#     "Paste the news article here:",
#     height=250
# )
#
# # ---------------- PREDICTION ---------------- #
# if st.button("Check News"):
#     if news.strip() == "":
#         st.warning("Please enter some text")
#     else:
#         clean_text = wordopt(news)
#
#         # Vectorize
#         X = vectorizer.transform([clean_text])
#
#         # Predict
#         prediction = model.predict(X)[0]
#
#         # Output
#         if prediction == 0:
#             st.error("🛑 Fake News")
#         else:
#             st.success("✅ Real News")

import streamlit as st
import pickle
import re
import string

# load model & vectorizer
model = pickle.load(open("best_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def wordopt(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("Fake News Detection")

news = st.text_area("Paste a news article here:")

if st.button("Check News"):
    clean_text = wordopt(news)

    if len(clean_text.split()) < 80:
        st.warning("Please enter a longer article (minimum 80 words).")
    else:
        X = vectorizer.transform([clean_text])
        proba = model.predict_proba(X)[0]
        confidence = max(proba)

        if confidence < 0.65:
            st.info("⚠️ Uncertain prediction (low confidence).")
        else:
            st.success("Real News ✅" if proba[1] > proba[0] else "Fake News ❌")
            st.caption(f"Confidence: {confidence:.2f}")
