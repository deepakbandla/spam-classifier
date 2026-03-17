import os
import sys

import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide"
)

model = pickle.load(open("models/model.pkl", "rb"))
pipeline = pickle.load(open("models/pipeline.pkl", "rb"))

st.title("📧 Spam Email Detection System")

st.markdown("""
This is a **Machine Learning based spam classifier** trained on the  
**Apache SpamAssassin dataset**.

It uses:
- Custom NLP preprocessing
- Scikit-learn pipeline
- Logistic Regression model
""")

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📨 Enter Email Content")
    email_text = st.text_area(
        "Paste your email here:",
        value=st.session_state.email_text,
        height=220
    )

    if st.button("🔍 Predict Spam"):

        if email_text.strip() == "":
            st.warning("Please enter some email text.")
        else:
            X = np.array([email_text], dtype=object)

            X_transformed = pipeline.transform(X)

            prediction = model.predict(X_transformed)
            probabilities = model.predict_proba(X_transformed)

            spam_prob = probabilities[0][1]

            st.markdown("### 🧠 Prediction Result")

            if prediction[0] == 1:
                st.error(f"🚨 SPAM detected\n\nConfidence: **{spam_prob:.2f}**")
            else:
                st.success(f"✅ NOT spam\n\nConfidence: **{1 - spam_prob:.2f}**")
with col2:
    st.subheader("⚙️ Model Info")

    st.info("Algorithm: Logistic Regression")
    st.info("Dataset: Apache SpamAssassin")
    st.info("Features: Custom NLP + Word Count Vectorization")


st.markdown("---")
st.header("🧪 Try Example Emails")

example_spam = """
Congratulations! You have been selected as a winner.
You have won $1000. Click this link to claim your prize now!!!
"""
example_ham = """
Hi team,
Please find attached the report for today's meeting.
Let me know your feedback.
Thanks.
"""

col1, col2 = st.columns(2)

if col1.button("Load Spam Example"):
    st.session_state.email_text = example_spam
if col2.button("Load Ham Example"):
    st.session_state.email_text = example_ham


st.markdown("---")
st.header("📊 Dataset Statistics")

c1, c2, c3 = st.columns(3)

c1.metric("Total Emails", "3000+")
c2.metric("Spam Emails", "≈ 1800")
c3.metric("Ham Emails", "≈ 1200")
st.markdown("---")
