import re
from pathlib import Path

import requests
import torch
from bs4 import BeautifulSoup
from slanggen.custom_logger import logger
from torch.nn.utils.rnn import pad_sequence


def get_data(filename: Path, url: str) -> list[str]:
    logger.info(f"Getting data from {url}")
    # Send a GET request to the website
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find main content div
    content_div = soup.find("div", {"id": "mw-content-text"})
    terms = []

    # Loop over all <ul> in the content div
    for ul in content_div.find_all("ul"):
        for li in ul.find_all("li"):
            text = li.get_text(strip=True)
            terms.append(text)

    # Clean terms: remove footnotes and headers
    ignore = ["Top", "0–9"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    clean_terms = []
    for t in terms:
        clean = t.split("[")[0].strip()  # remove footnotes
        if clean and clean not in ignore and len(clean) > 1:
            clean_terms.append(clean)

    # Determine start and end index (specific games)
    start_idx = next((i for i, t in enumerate(clean_terms) if t == "1080° Snowboarding"), 0)
    end_idx = next((i for i, t in reversed(list(enumerate(clean_terms))) if t == "Zumba Fitness"), len(clean_terms) - 1)

    game_terms = clean_terms[start_idx:end_idx + 1]

    # Process each word: lowercase and wrap with <s> tags
    processed_words = [f"<s>{t.lower()}</s>" for t in game_terms]

    # Save the processed words to a file
    logger.info(f"Saving {len(processed_words)} processed words to {filename}")
    with open(filename, "w", encoding="utf-8") as file:
        for word in processed_words:
            file.write(word + "\n")

    return processed_words


def load_data(filename: Path, url: str) -> list[str]:
    if not filename.exists():
        logger.info(f"File {filename} not found. donwloading from {url}")
        processed_words = get_data(filename, url)
    else:
        logger.info(f"Loading processed words from {filename}")
        with open(filename, "r", encoding="utf-8") as file:
            processed_words = [line.strip() for line in file]
    logger.info(f"Loaded {len(processed_words)} words")
    return processed_words


def preprocess(corpus: list[str], tokenizer) -> torch.Tensor:
    encoded_sequences = [tokenizer.encode(word).ids for word in corpus]
    padded_sequences = pad_sequence(
        [torch.tensor(seq) for seq in encoded_sequences], batch_first=True
    )
    return padded_sequences


class ShiftedDataset:
    def __init__(self, sequences: torch.Tensor):
        self.X = sequences[:, :-1]
        self.y = sequences[:, 1:]

    def to(self, device):
        self.X = self.X.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __repr__(self):
        return f"ShiftedDataset {self.X.shape}"
