#!/usr/bin/env python3
"""
Indian Kanoon Scraping Script
==============================
Script to collect legal documents from Indian Kanoon for training.

Usage:
    python scripts/scrape_indian_kanoon.py --court supremecourt --year 2023 --max-docs 100

WARNING: Always respect robots.txt and rate limits when scraping.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import IndianKanoonScraper
from src.utils import setup_logging

from loguru import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape legal documents from Indian Kanoon"
    )
    
    parser.add_argument(
        "--court",
        type=str,
        default="supremecourt",
        choices=list(IndianKanoonScraper.COURTS.keys()),
        help="Court to scrape from"
    )
    
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to filter (optional)"
    )
    
    parser.add_argument(
        "--max-docs",
        type=int,
        default=100,
        help="Maximum documents to scrape"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw/indian_kanoon",
        help="Output directory for scraped documents"
    )
    
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Seconds between requests"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "txt"],
        default="json",
        help="Output format"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually scrape, just show what would be done"
    )
    
    return parser.parse_args()


def main():
    """Main scraping function."""
    args = parse_args()
    
    # Setup logging
    log_file = f"logs/scraping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs("logs", exist_ok=True)
    setup_logging(log_level="INFO", log_file=log_file)
    
    print("=" * 60)
    print("Indian Kanoon Document Scraper")
    print("=" * 60)
    print(f"Court: {IndianKanoonScraper.COURTS.get(args.court, args.court)}")
    print(f"Year filter: {args.year or 'All years'}")
    print(f"Max documents: {args.max_docs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Rate limit: {args.rate_limit}s between requests")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No actual scraping will be performed]")
        print("\nTo run actual scraping, remove the --dry-run flag:")
        print(f"  python scripts/scrape_indian_kanoon.py --court {args.court} --max-docs {args.max_docs}")
        return
    
    # Confirm before proceeding
    print("\n⚠️  WARNING: Web scraping should be done responsibly.")
    print("   - Respect rate limits")
    print("   - Check robots.txt")
    print("   - Don't overload servers")
    
    confirm = input("\nProceed with scraping? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Scraping cancelled.")
        return
    
    # Initialize scraper
    scraper = IndianKanoonScraper(
        output_dir=args.output_dir,
        rate_limit=args.rate_limit
    )
    
    # Start scraping
    print(f"\nStarting scrape...")
    
    documents_saved = 0
    stats = {
        "total_attempted": 0,
        "successful": 0,
        "failed": 0,
        "duplicates": 0
    }
    
    try:
        for doc in scraper.scrape_court(
            court=args.court,
            year=args.year,
            max_documents=args.max_docs
        ):
            stats["total_attempted"] += 1
            
            try:
                filepath = scraper.save_document(doc, format=args.format)
                documents_saved += 1
                stats["successful"] += 1
                
                logger.info(f"Saved: {filepath.name}")
                
            except Exception as e:
                stats["failed"] += 1
                logger.error(f"Failed to save document: {e}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
    
    except Exception as e:
        logger.error(f"Scraping error: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Documents saved: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Output directory: {args.output_dir}")
    print(f"Log file: {log_file}")
    
    # Save metadata
    metadata = {
        "court": args.court,
        "year": args.year,
        "documents_saved": documents_saved,
        "scrape_date": datetime.now().isoformat(),
        "stats": stats
    }
    
    metadata_path = Path(args.output_dir) / "scrape_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
