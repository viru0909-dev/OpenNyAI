"""
Legal Text Preprocessor
========================
Text preprocessing utilities specifically designed for legal documents.
"""

import re
import unicodedata
from typing import List, Optional

from loguru import logger


class LegalTextPreprocessor:
    """
    Preprocessor for legal text documents.
    
    Handles:
    - Text cleaning and normalization
    - Legal-specific preprocessing
    - Sentence segmentation for legal texts
    - Citation extraction
    """
    
    # Common legal abbreviations to preserve
    LEGAL_ABBREVIATIONS = [
        "vs.", "v.", "no.", "nos.", "sec.", "s.", "art.", "arts.",
        "cl.", "para.", "paras.", "sch.", "pt.", "pts.", "r.", "rr.",
        "hon.", "hon'ble", "shri", "smt.", "dr.", "mr.", "mrs.", "ms.",
        "ltd.", "pvt.", "co.", "corp.", "inc.", "llp.", "llc.",
    ]
    
    # Patterns for legal citations
    CITATION_PATTERNS = [
        r'\d{4}\s*\(\d+\)\s*SCC\s*\d+',  # Supreme Court Citation
        r'AIR\s*\d{4}\s*SC\s*\d+',  # AIR Citation
        r'\d{4}\s*Cri\.?\s*L\.?J\.?\s*\d+',  # Criminal Law Journal
        r'ILR\s*\d{4}\s*\w+\s*\d+',  # Indian Law Reports
    ]
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_stopwords: bool = False,
        preserve_legal_terms: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase.
            remove_stopwords: Whether to remove stopwords.
            preserve_legal_terms: Whether to preserve legal terminology.
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.preserve_legal_terms = preserve_legal_terms
        
        # Compile regex patterns
        self.citation_regex = re.compile(
            '|'.join(self.CITATION_PATTERNS),
            re.IGNORECASE
        )
        
        logger.info("Initialized LegalTextPreprocessor")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize legal text.
        
        Args:
            text: Input text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve legal symbols
        text = re.sub(r'[^\w\s\-.,;:()\'\"§¶/\\]', '', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        
        # Handle Hon'ble, etc.
        text = re.sub(r"(\w)'(\w)", r"\1'\2", text)
        
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def normalize_legal_text(self, text: str) -> str:
        """
        Normalize legal-specific text patterns.
        
        Args:
            text: Input legal text.
            
        Returns:
            Normalized text.
        """
        # Standardize section references
        text = re.sub(r'[Ss]ection\s+(\d+)', r'Section \1', text)
        text = re.sub(r'[Ss]ec\.?\s*(\d+)', r'Section \1', text)
        
        # Standardize article references
        text = re.sub(r'[Aa]rticle\s+(\d+)', r'Article \1', text)
        text = re.sub(r'[Aa]rt\.?\s*(\d+)', r'Article \1', text)
        
        # Standardize versus
        text = re.sub(r'\s+vs?\.?\s+', ' v. ', text, flags=re.IGNORECASE)
        
        # Standardize "the petitioner" / "the respondent"
        text = re.sub(r'petitioner[\s\-]?(\d+)', r'Petitioner-\1', text, flags=re.IGNORECASE)
        text = re.sub(r'respondent[\s\-]?(\d+)', r'Respondent-\1', text, flags=re.IGNORECASE)
        
        return text
    
    def extract_citations(self, text: str) -> List[str]:
        """
        Extract legal citations from text.
        
        Args:
            text: Input text containing citations.
            
        Returns:
            List of extracted citations.
        """
        citations = self.citation_regex.findall(text)
        return list(set(citations))
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment legal text into sentences.
        
        Handles legal-specific sentence boundaries correctly.
        
        Args:
            text: Input text to segment.
            
        Returns:
            List of sentences.
        """
        # Protect abbreviations
        protected = text
        for abbrev in self.LEGAL_ABBREVIATIONS:
            pattern = re.escape(abbrev)
            protected = re.sub(
                pattern,
                abbrev.replace('.', '<DOT>'),
                protected,
                flags=re.IGNORECASE
            )
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', protected)
        
        # Restore dots in abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline for legal text.
        
        Args:
            text: Input text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        # Step 1: Basic cleaning
        text = self.clean_text(text)
        
        # Step 2: Legal normalization
        if self.preserve_legal_terms:
            text = self.normalize_legal_text(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess.
            
        Returns:
            List of preprocessed texts.
        """
        return [self.preprocess(text) for text in texts]


if __name__ == "__main__":
    # Example usage
    preprocessor = LegalTextPreprocessor()
    
    sample_text = """
    In the case of State of Maharashtra vs. John Doe, 
    as per Section 302 of IPC and sec. 34 of IPC,
    the Hon'ble Supreme Court in 2019 (5) SCC 123 held that...
    """
    
    cleaned = preprocessor.preprocess(sample_text)
    print(f"Cleaned text:\n{cleaned}")
    
    citations = preprocessor.extract_citations(sample_text)
    print(f"\nExtracted citations: {citations}")
