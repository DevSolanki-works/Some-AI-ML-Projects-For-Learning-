import requests
from bs4 import BeautifulSoup
import pandas as pd

print("Initiating Web Scraper...")

quotes_data = []

# Scrape the first 5 pages of the site
for page in range(1, 25):
    url = f"http://quotes.toscrape.com/page/{page}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all quote blocks
    quotes = soup.find_all('div', class_='quote')
    
    for quote in quotes:
        text = quote.find('span', class_='text').text.strip('“”')
        author = quote.find('small', class_='author').text
        # Grab the first tag (e.g., 'inspirational', 'humor')
        tags = quote.find('div', class_='tags').find_all('a', class_='tag')
        main_tag = tags[0].text if tags else "general"
        
        quotes_data.append({'Text': text, 'Author': author, 'Tag': main_tag})

# Save our raw data
df = pd.DataFrame(quotes_data)
df.to_csv('scraped_data.csv', index=False)
print(f"Successfully scraped {len(df)} records and saved to CSV!")
