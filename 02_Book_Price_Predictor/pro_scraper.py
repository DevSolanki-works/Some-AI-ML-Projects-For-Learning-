import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# Disguise our script as a real browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_books():
    books_data = []
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"

    print("🚀 Initializing Stealth Scraper...")

    for page in range(1, 4): # Let's do 3 pages
        print(f"Reading Page {page}...")
        response = requests.get(base_url.format(page), headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        books = soup.find_all('article', class_='product_pod')

        for book in books:
            title = book.h3.a['title']
            raw_price = book.find('p', class_='price_color').text
            # Regex: find only digits and dots
            price = re.sub(r'[^\d.]', '', raw_price)
            
            # The "Hard" Part: Extracting the Star Rating from a class name
            # Format: <p class="star-rating Three">
            rating_classes = book.find('p', class_='star-rating')['class']
            rating = rating_classes[1] # Gets the "Three" or "Four" part
            
            books_data.append({
                'Title': title, 
                'Price': float(price), 
                'Rating': rating
            })
        
        time.sleep(1) # Be "polite" so the server doesn't block us

    response = requests.get(base_url.format(page), headers=HEADERS)
    response.encoding = response.apparent_encoding # Force Python to use the site's detected encoding
    soup = BeautifulSoup(response.text, 'html.parser')

    df = pd.DataFrame(books_data)
    df.to_csv('book_data.csv', index=False)
    print(f"Done! Captured {len(df)} books.")


scrape_books()

