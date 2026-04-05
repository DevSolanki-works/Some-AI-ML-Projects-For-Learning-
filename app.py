import streamlit as st
import pandas as pd
import joblib
import os
import requests # To talk to your local Llama 3
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="AI Intelligence Hub", layout="wide")

# --- PRO FEATURE: LOCAL LLM FALLBACK ---
def ask_llama(quote):
    """Talks to your local Ollama server for a second opinion"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": f"""
Analyze this quote: '{quote}'
Classify it into ONE of these categories: [Humor, Life, Inspirational, Love].
If it doesn't fit any, output 'Unknown'. 
Provide a 1-sentence reason why.
""",
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()['response'].strip()
    except:
        return "Llama 3 Offline"

# --- UI HEADER ---
st.title("🛡️ Pro-Grade Sentiment & Intent Engine")
st.info("System Architecture: Random Forest Classifier + Llama 3 Hybrid Fallback")

# --- SIDEBAR: DATA CONTROLS ---
with st.sidebar:
    st.header("Control Panel")
    confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 50)
    if st.button("🔄 Clear Brain & Retrain"):
        # (Insert your training logic from previous step here)
        st.success("Brain Refreshed!")

# --- MAIN INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    user_quote = st.text_area("Input Stream:", placeholder="Type a quote here...", height=150)
    predict_btn = st.button("🔍 Analyze Intent")

if predict_btn and user_quote:
    if os.path.exists('quote_model.pkl'):
        model = joblib.load('quote_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        # 1. Get Prediction AND Probabilities
        input_vector = vectorizer.transform([user_quote])
        prediction = model.predict(input_vector)[0]
        probs = model.predict_proba(input_vector).max() * 100 # Highest confidence %
        
        with col2:
            st.metric("Model Confidence", f"{probs:.1f}%")
            
            # --- THE HYBRID LOGIC ---
            if probs < confidence_threshold:
                st.warning(f"Low Confidence ({probs:.1f}%). Summoning Llama 3...")
                with st.spinner("Llama 3 is thinking..."):
                    llm_opinion = ask_llama(user_quote)
                st.subheader(f"Final Decision: :orange[{llm_opinion}] (LLM)")
                st.caption("The local model was unsure, so the Gen AI took over.")
            else:
                st.subheader(f"Final Decision: :green[{prediction.upper()}] (Local RF)")
                st.progress(probs / 100)
    else:
        st.error("No model found. Train it first!")

