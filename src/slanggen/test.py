import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/List_of_video_game_franchises"
headers = {"User-Agent": "Mozilla/5.0"}

resp = requests.get(URL, headers=headers)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")

# Vind de main content div
content_div = soup.find("div", {"id": "mw-content-text"})

terms = []

# Loop over alle <ul> in content_div
for ul in content_div.find_all("ul"):
    for li in ul.find_all("li"):
        text = li.get_text(strip=True)
        terms.append(text)

clean_terms = []

# Opschonen: footnotes verwijderen en headers filteren
ignore = ["Top", "0–9"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

for t in terms:
    clean = t.split("[")[0].strip()  # verwijder footnotes
    if clean and clean not in ignore and len(clean) > 1:
        clean_terms.append(clean)  # <s> pas toevoegen later

# Bepaal begin- en eindindex
start_idx = next((i for i, t in enumerate(clean_terms) if t == "1080° Snowboarding"), 0)
end_idx = next((i for i, t in reversed(list(enumerate(clean_terms))) if t == "Zumba Fitness"), len(clean_terms)-1)

# Bewaar alleen de games tussen start en eind
game_terms = clean_terms[start_idx:end_idx + 1]

# Print met enkel <s> tags
for t in game_terms:
    print(f"<s>{t}</s>")