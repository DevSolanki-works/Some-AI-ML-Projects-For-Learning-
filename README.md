# 🚀 The AI/ML Learning Lab: 
**Developer: Dev Solanki** | *Built during a 10-hour Sunday sprint* ☕🔥

This repository is a modular suite of Machine Learning and Generative AI projects, moving from basic data scraping to full-scale automated pipelines.

## 🏗️ Project Architecture
I transitioned this repo from a flat-file structure to a **Professional Modular Architecture** to support scalability and clean deployment.

* **01_Quote_Classifier:** A Streamlit web app that classifies quotes in real-time. Includes an "Active Learning" mode to retrain models based on user input.
* **02_Book_Price_Predictor:** A web-scraping project that taught me the "Garbage In, Garbage Out" rule of ML (R²: 0.0011).
* **03_Housing_Model:** A regression model achieving **80.51% accuracy** using the California Housing dataset.
* **04_Gym_Bro_GenAI:** A Generative AI pipeline that uses **Llama 3** (via Ollama) to transform boring quotes into high-octane gym motivation.

## 🛠️ Technical Highlights
- **Path Management:** Implemented `os.path` relative routing for cross-platform compatibility.
- **Local LLM Inference:** Integrated Llama 3 for zero-shot text transformation.
- **Data Engineering:** Built custom scrapers using `BeautifulSoup4` with rotating headers for stealth.
- **Model Persistence:** Managed serialized `.pkl` models for fast inference.

## 🚦 How to Run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. To run the UI: `cd 01_Quote_Classifier && streamlit run app.py`