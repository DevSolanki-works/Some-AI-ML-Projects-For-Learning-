import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import requests
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


# --- DATA ENGINE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "scraped_data.csv")

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=['Text', 'Author', 'Tag'])

def train_engine(df):
    if len(df) < 5: return None, None
    # Filter out categories with only 1 entry to avoid split errors
    df = df.groupby('Tag').filter(lambda x: len(x) > 1)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(df['Text'])
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, df['Tag'])
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return model, vectorizer

# --- UI LAYOUT ---
st.title("🛡️ AI Product Suite: Active Learning Edition")
df = load_data()

# --- SIDEBAR: ANALYTICS ---
with st.sidebar:
    st.header("📊 Database Stats")
    st.write(f"Total Samples: {len(df)}")
    st.write(f"Unique Categories: {df['Tag'].nunique()}")
    
    # Visualizing Category Distribution
    if not df.empty:
        fig = px.pie(df, names='Tag', title="Data Balance", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# --- MAIN: PREDICTION & FEEDBACK ---
user_input = st.text_input("Enter Quote:", placeholder="Analyze something new...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "quote_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl")

if user_input:
    model = joblib.load(MODEL_PATH)
    vec = joblib.load(VECTORIZER_PATH)

    # Prediction logic
    vec_input = vec.transform([user_input])
    prediction = model.predict(vec_input)[0]
    probs = model.predict_proba(vec_input).max() * 100

    st.subheader(f"Result: :green[{prediction.upper()}] ({probs:.1f}% confidence)")

    # --- THE ACTIVE LEARNING FEATURE ---
    st.write("---")
    st.write("🎯 **Is the AI wrong? Correct it to make it smarter:**")
    correct_tag = st.selectbox("What should this be?", options=df['Tag'].unique())
    
    if st.button("✅ Submit Correction & Retrain"):
        new_data = pd.DataFrame([{'Text': user_input, 'Author': 'UserContribution', 'Tag': correct_tag}])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        
        with st.spinner("Learning from you..."):
            train_engine(df)
        st.success(f"Added to database! AI now knows '{user_input}' is {correct_tag}.")
        st.balloons()

