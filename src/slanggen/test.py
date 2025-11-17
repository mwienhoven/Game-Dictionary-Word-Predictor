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

# Glossary is een <dl> met <dt> (term) en <dd> (definitie)
entries = []

for dl in soup.find_all("dl"):
    dts = dl.find_all("dt")
    dds = dl.find_all("dd")

    # skip als de aantallen niet matchen
    if len(dts) != len(dds) or len(dts) == 0:
        continue
    
    for dt, dd in zip(dts, dds):
        term = dt.get_text(" ", strip=True)
        definition = dd.get_text(" ", strip=True)

        # Bouw de <s>...</s> regel
        sentence = f"{term}: {definition}"
        entries.append(f"<s>{sentence}</s>")

# Print de eerste 20 regels als voorbeeld
for line in entries[:20]:
    print(line)
