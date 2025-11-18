import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/Glossary_of_video_game_terms"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

resp = requests.get(URL, headers=headers)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

terms = []

for dt in soup.find_all("dt"):
    term = dt.get_text(" ", strip=True)

    # Filter: geen lege dingen, geen footnotes, geen hele zinnen
    if term and not term.startswith("[") and len(term) < 80:
        terms.append(f"<s>{term}</s>")

for t in terms:
    print(t)
