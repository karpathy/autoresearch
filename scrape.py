import sys
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from markdownify import markdownify as md
import urllib3

# Suppress insecure request warnings if we have to use verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def search(query, max_results=5):
    """Search DuckDuckGo and return a list of results."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
        return results
    except Exception as e:
        return [{"title": "Error", "href": "", "body": str(e)}]

def get_content(url):
    """Fetch content from a URL and convert it to markdown."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Try with verification first, fallback to no verification if it fails due to SSL
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.SSLError:
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, 'lxml')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        markdown = md(str(soup), heading_style="ATX")
        return markdown.strip()
    except Exception as e:
        return f"Error fetching {url}: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scrape.py search <query> [max_results]")
        print("       python scrape.py get <url>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "search":
        if len(sys.argv) < 3:
            print("Error: Missing query for search")
            sys.exit(1)
        query = sys.argv[2]
        max_results = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        results = search(query, max_results)
        for i, r in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Title: {r.get('title', 'N/A')}")
            print(f"Link: {r.get('href', 'N/A')}")
            print(f"Snippet: {r.get('body', 'N/A')}")
            print("-" * 20)
    elif cmd == "get":
        if len(sys.argv) < 3:
            print("Error: Missing URL for get")
            sys.exit(1)
        url = sys.argv[2]
        print(get_content(url))
    else:
        print(f"Unknown command: {cmd}")
