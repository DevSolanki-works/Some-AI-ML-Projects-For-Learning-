import requests
from bs4 import BeautifulSoup

def get_boring_quotes(limit=3):
    """Step 1: Data Ingestion (The Scraper)"""
    url = "http://quotes.toscrape.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Grab the text of the first few quotes
    quotes = [q.find('span', class_='text').text for q in soup.find_all('div', class_='quote')[:limit]]
    return quotes

def gym_bro_transformer(quote):
    """Step 2: The Generative AI Engine"""
    # This is Prompt Engineering. We lock the AI into a specific persona.
    prompt = f"""
    Take this boring philosophical quote: {quote}
    Rewrite it to sound like an aggressive, high-energy gym motivation speech.
    Keep it under 2 sentences. No emojis. Output ONLY the new quote.
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3", 
        "prompt": prompt, 
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        return response.json()['response'].strip()
    except Exception as e:
        return "Llama 3 is asleep. Wake it up!"

# --- THE EXECUTION PIPELINE ---
print("🚀 Initiating Gen AI Data Pipeline...\n")

# 1. Scrape the raw data
raw_data = get_boring_quotes(3)

# 2. Transform the data using the LLM
for i, original in enumerate(raw_data, 1):
    print(f"--- Sample {i} ---")
    print(f"📖 Original: {original}")
    
    # Pass the scraped text directly into the AI
    rewritten = gym_bro_transformer(original)
    
    print(f"🔥 Gen AI: {rewritten}\n")