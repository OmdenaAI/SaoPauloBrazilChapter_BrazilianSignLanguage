import asyncio
import os

import aiohttp
from bs4 import BeautifulSoup

BASE_URL = "https://libras.cin.ufpe.br/"
WORDS_FILE = "v_librasil_words.txt"
WORDS_N_LINKS_FILE = "v_librasil_words_n_links.txt"
PROGRESS_FILE = "progress.txt"


async def fetch_words(session, page):
    """Fetch words from a given page, returning an empty list if no words found."""
    url = f"{BASE_URL}?page={page}"

    try:
        async with session.get(url, timeout=30) as response:
            if response.status != 200:
                print(f"Skipping page {page}: HTTP {response.status}")
                return []

            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            words = [a.text.strip() for a in soup.select("table.table tbody tr td a")]
            return words

    except aiohttp.ClientError as e:
        print(f"Request failed for page {page}: {e}")
        return []


async def fetch_words_and_links(session, page):
    """Fetch words and their links from a given page."""
    url = f"{BASE_URL}?page={page}"

    try:
        async with session.get(url, timeout=60) as response:
            if response.status != 200:
                print(f"Skipping page {page}: HTTP {response.status}")
                return []

            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")

            # Extract words and links together
            words_links = [
                (a.text.strip(), a["href"])
                for a in soup.select("table.table tbody tr td a")
            ]

            return words_links

    except aiohttp.ClientError as e:
        print(f"Request failed for page {page}: {e}")
        return []


def load_last_page():
    """Read last scraped page from progress file or start from page 1."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 1  # Default to first page if no progress file exists


def save_last_page(page):
    """Save last successfully scraped page to progress file."""
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(page))


def save_words(words):
    """Append scraped words to words file."""
    with open(WORDS_FILE, "a") as f:
        for word in words:
            f.write(word + "\n")


def save_words_and_links(words_n_links):
    """Append scraped words and links to words and links file."""
    with open(WORDS_N_LINKS_FILE, "a") as f:
        for word, link in words_n_links:
            f.write(word + f" {link}" + "\n")


async def scrape_all_pages():
    """Scrape all pages dynamically until an empty page is found."""
    all_words = []
    page = load_last_page()  # Start from last saved page
    session = None

    try:
        session = aiohttp.ClientSession()
        while True:
            words_and_links = await fetch_words_and_links(session, page)
            if not words_and_links:
                print(f"No more words found. Stopping at page {page}.")
                break

            save_words_and_links(words_and_links)  # Save words immediately
            save_last_page(page)  # Save progress
            print(f"Scraped {len(words_and_links)} words from page {page}.")
            page += 1

    except asyncio.CancelledError:
        print("\nScraping process was cancelled.")

    finally:
        if session:
            await session.close()
            print("Session closed.")

    print(f"\nTotal words scraped: {len(all_words)}")
    return all_words


# Run the scraper
try:
    asyncio.run(scrape_all_pages())

except KeyboardInterrupt:
    print("\nScraping interrupted by user.")
