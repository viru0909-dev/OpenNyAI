"""
Legal Document Summarizer
==========================
Model for generating summaries of legal documents.
"""

from typing import Dict, List, Optional

import torch
from loguru import logger


class LegalSummarizer:
    """
    Summarization model for legal documents.
    
    Features:
    - Abstractive summarization using transformer models
    - Extractive summarization for key points
    - Configurable summary length
    - Support for various legal document types
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        max_input_length: int = 1024,
        max_output_length: int = 256,
        min_output_length: int = 50,
        device: Optional[str] = None
    ):
        """
        Initialize the summarizer.
        
        Args:
            model_name: Pretrained model name from Hugging Face.
            max_input_length: Maximum input sequence length.
            max_output_length: Maximum summary length.
            min_output_length: Minimum summary length.
            device: Device to run the model on.
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        logger.info(f"Initialized LegalSummarizer with {model_name}")
    
    def load_model(self):
        """Load the pretrained summarization model."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )
        
        logger.info(f"Loading summarization model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self._is_loaded = True
        
        logger.info("Summarization model loaded successfully")
    
    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """
        Generate a summary of the input text.
        
        Args:
            text: Input text to summarize.
            max_length: Maximum summary length (uses default if None).
            min_length: Minimum summary length (uses default if None).
            num_beams: Number of beams for beam search.
            no_repeat_ngram_size: Size of n-grams to avoid repeating.
            
        Returns:
            Generated summary.
        """
        if not self._is_loaded:
            self.load_model()
        
        max_length = max_length or self.max_output_length
        min_length = min_length or self.min_output_length
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        ).to(self.device)
        
        # Generate summary
        self.model.eval()
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def summarize_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> List[str]:
        """
        Generate summaries for multiple texts.
        
        Args:
            texts: List of input texts.
            max_length: Maximum summary length.
            min_length: Minimum summary length.
            
        Returns:
            List of generated summaries.
        """
        return [
            self.summarize(text, max_length, min_length)
            for text in texts
        ]
    
    def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from a legal document.
        
        Uses extractive summarization to identify important sentences.
        
        Args:
            text: Input legal text.
            num_points: Number of key points to extract.
            
        Returns:
            List of key sentences.
        """
        import re
        
        # Simple sentence-based extraction
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= num_points:
            return sentences
        
        # Score sentences by length and position (simple heuristic)
        scored = []
        for i, sent in enumerate(sentences):
            # Position score (earlier = more important)
            pos_score = 1 - (i / len(sentences))
            # Length score (moderate length = better)
            len_score = min(len(sent) / 200, 1)
            # Combined score
            score = 0.5 * pos_score + 0.5 * len_score
            scored.append((sent, score))
        
        # Sort by score and return top points
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:num_points]]
    
    def save_model(self, output_dir: str):
        """Save the model to a directory."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Nothing to save.")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Summarization model saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    summarizer = LegalSummarizer()
    print(f"Summarizer initialized: {summarizer.model_name}")
