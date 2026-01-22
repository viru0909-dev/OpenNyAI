"""
Indian Legal Document Scraper
==============================
Scraping utilities for collecting legal documents from Indian Kanoon and e-Courts.

WARNING: Always respect robots.txt and rate limits when scraping.
"""

import time
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Generator
from urllib.parse import urljoin, urlparse

from loguru import logger


class IndianKanoonScraper:
    """
    Scraper for Indian Kanoon (https://indiankanoon.org/).
    
    Features:
    - Court-wise and year-wise crawling
    - Rate limiting to respect server resources
    - HTML structure preservation for semantic signals
    - Document deduplication via hashing
    """
    
    BASE_URL = "https://indiankanoon.org"
    
    # Court identifiers on Indian Kanoon
    COURTS = {
        "supremecourt": "Supreme Court of India",
        "allahabad": "Allahabad High Court",
        "bombay": "Bombay High Court",
        "calcutta": "Calcutta High Court",
        "delhi": "Delhi High Court",
        "madras": "Madras High Court",
        "karnataka": "Karnataka High Court",
        "kerala": "Kerala High Court",
        "punjab": "Punjab and Haryana High Court",
        "rajasthan": "Rajasthan High Court",
        "gujarat": "Gujarat High Court",
        "hyderabad": "Telangana High Court",
    }
    
    def __init__(
        self,
        output_dir: str = "./data/raw/indian_kanoon",
        rate_limit: float = 2.0,
        max_retries: int = 3
    ):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Directory to save scraped documents.
            rate_limit: Minimum seconds between requests.
            max_retries: Maximum retry attempts for failed requests.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self._last_request_time = 0
        self._seen_hashes = set()
        
        logger.info(f"Initialized IndianKanoonScraper with output: {output_dir}")
    
    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            wait_time = self.rate_limit - elapsed + random.uniform(0.5, 1.5)
            time.sleep(wait_time)
        self._last_request_time = time.time()
    
    def _get_page(self, url: str) -> Optional[str]:
        """
        Fetch a page with retry logic.
        
        Args:
            url: URL to fetch.
            
        Returns:
            HTML content or None if failed.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "requests and beautifulsoup4 are required. "
                "Install with: pip install requests beautifulsoup4"
            )
        
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; OpenNyAI Research Bot; +https://opennyai.org)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None
    
    def _extract_document_links(self, html: str, court: str) -> List[str]:
        """Extract document links from a search results page."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        # Indian Kanoon uses specific class for result links
        for result in soup.find_all('div', class_='result'):
            title_link = result.find('a', class_='result_title')
            if title_link and title_link.get('href'):
                full_url = urljoin(self.BASE_URL, title_link['href'])
                links.append(full_url)
        
        return links
    
    def _extract_judgment_content(self, html: str) -> Dict[str, str]:
        """Extract structured content from a judgment page."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        content = {
            "title": "",
            "court": "",
            "date": "",
            "judges": "",
            "citations": [],
            "full_text": "",
            "html": html,  # Preserve HTML for semantic signals
        }
        
        # Extract title
        title_elem = soup.find('h2', class_='doc_title')
        if title_elem:
            content["title"] = title_elem.get_text(strip=True)
        
        # Extract judgment text
        judgment_elem = soup.find('div', {'id': 'judgment'}) or soup.find('pre')
        if judgment_elem:
            content["full_text"] = judgment_elem.get_text(separator='\n')
        
        # Extract metadata
        for meta in soup.find_all('p', class_='docsource_main'):
            text = meta.get_text(strip=True)
            if 'Court' in text:
                content["court"] = text
            elif 'Bench' in text or 'Judge' in text:
                content["judges"] = text
        
        # Extract citations
        for cite in soup.find_all('a', class_='cite_link'):
            content["citations"].append(cite.get_text(strip=True))
        
        return content
    
    def _compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash for deduplication."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def scrape_court(
        self,
        court: str,
        year: Optional[int] = None,
        max_documents: int = 100
    ) -> Generator[Dict, None, None]:
        """
        Scrape documents from a specific court.
        
        Args:
            court: Court identifier (e.g., 'supremecourt').
            year: Optional year filter.
            max_documents: Maximum documents to scrape.
            
        Yields:
            Document dictionaries.
        """
        if court not in self.COURTS:
            raise ValueError(f"Unknown court: {court}. Available: {list(self.COURTS.keys())}")
        
        logger.info(f"Starting scrape for {self.COURTS[court]}, year={year}")
        
        # Build search URL
        search_url = f"{self.BASE_URL}/search/?formInput=doctypes:{court}"
        if year:
            search_url += f" fromdate:{year}-01-01 todate:{year}-12-31"
        
        documents_scraped = 0
        page = 0
        
        while documents_scraped < max_documents:
            # Fetch search results page
            page_url = f"{search_url}&pagenum={page}"
            html = self._get_page(page_url)
            
            if not html:
                break
            
            # Extract document links
            doc_links = self._extract_document_links(html, court)
            
            if not doc_links:
                logger.info(f"No more documents found on page {page}")
                break
            
            for doc_url in doc_links:
                if documents_scraped >= max_documents:
                    break
                
                # Fetch document
                doc_html = self._get_page(doc_url)
                if not doc_html:
                    continue
                
                # Extract content
                content = self._extract_judgment_content(doc_html)
                
                # Check for duplicates
                doc_hash = self._compute_hash(content["full_text"])
                if doc_hash in self._seen_hashes:
                    logger.debug(f"Duplicate document skipped: {doc_url}")
                    continue
                
                self._seen_hashes.add(doc_hash)
                
                content["url"] = doc_url
                content["hash"] = doc_hash
                content["court_id"] = court
                content["year"] = year
                
                documents_scraped += 1
                logger.info(f"Scraped {documents_scraped}/{max_documents}: {content['title'][:50]}...")
                
                yield content
            
            page += 1
    
    def save_document(self, doc: Dict, format: str = "json"):
        """
        Save a document to disk.
        
        Args:
            doc: Document dictionary.
            format: Output format ('json' or 'txt').
        """
        import json
        
        # Create filename from hash
        filename = f"{doc['court_id']}_{doc['hash'][:16]}"
        
        if format == "json":
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        else:
            filepath = self.output_dir / f"{filename}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(doc["full_text"])
        
        return filepath


class ECourtsAPI:
    """
    Interface for e-Courts API (when available).
    
    Note: This is a placeholder for when official API access is available.
    Currently, e-Courts requires CAPTCHA handling for programmatic access.
    """
    
    BASE_URL = "https://services.ecourts.gov.in"
    
    def __init__(self):
        logger.warning(
            "ECourtsAPI: Official API access is limited. "
            "Consider using pre-compiled datasets like ILDC instead."
        )
    
    def search_cases(self, **kwargs):
        """Placeholder for case search functionality."""
        raise NotImplementedError(
            "e-Courts API requires authentication and CAPTCHA handling. "
            "Use pre-compiled datasets or contact e-Courts for API access."
        )


if __name__ == "__main__":
    # Example usage (dry run)
    print("=" * 60)
    print("Indian Legal Document Scraper")
    print("=" * 60)
    print("\nAvailable Courts:")
    for court_id, court_name in IndianKanoonScraper.COURTS.items():
        print(f"  - {court_id}: {court_name}")
    
    print("\nUsage Example:")
    print("""
    from src.data.scraper import IndianKanoonScraper
    
    scraper = IndianKanoonScraper(output_dir="./data/raw/judgments")
    
    # Scrape Supreme Court judgments from 2023
    for doc in scraper.scrape_court("supremecourt", year=2023, max_documents=50):
        scraper.save_document(doc)
        print(f"Saved: {doc['title'][:50]}...")
    """)
