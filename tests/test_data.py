"""
Tests for data loading and preprocessing modules.
"""

import pytest
from pathlib import Path
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import LegalDataLoader, LegalTextPreprocessor


class TestLegalDataLoader:
    """Tests for LegalDataLoader class."""
    
    def test_initialization(self):
        """Test loader initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = LegalDataLoader(data_dir=tmpdir)
            assert loader.raw_dir.exists()
            assert loader.processed_dir.exists()
    
    def test_load_json(self):
        """Test JSON file loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON file
            test_data = [{"text": "Test document 1"}, {"text": "Test document 2"}]
            json_path = Path(tmpdir) / "test.json"
            with open(json_path, 'w') as f:
                json.dump(test_data, f)
            
            loader = LegalDataLoader(data_dir=tmpdir)
            loaded_data = loader.load_json(json_path)
            
            assert len(loaded_data) == 2
            assert loaded_data[0]["text"] == "Test document 1"
    
    def test_save_processed_data(self):
        """Test saving processed data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = LegalDataLoader(data_dir=tmpdir)
            
            test_data = [{"text": "Processed document"}]
            output_path = loader.save_processed_data(test_data, "output", format="json")
            
            assert output_path.exists()
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data == test_data


class TestLegalTextPreprocessor:
    """Tests for LegalTextPreprocessor class."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = LegalTextPreprocessor()
        
        dirty_text = "  This   has   extra   spaces.  "
        cleaned = preprocessor.clean_text(dirty_text)
        
        assert "   " not in cleaned
        assert cleaned.startswith("This")
    
    def test_normalize_legal_text(self):
        """Test legal text normalization."""
        preprocessor = LegalTextPreprocessor()
        
        text = "As per sec. 302 and section 34"
        normalized = preprocessor.normalize_legal_text(text)
        
        assert "Section 302" in normalized
        assert "Section 34" in normalized
    
    def test_extract_citations(self):
        """Test citation extraction."""
        preprocessor = LegalTextPreprocessor()
        
        text = "The court in 2019 (5) SCC 123 and AIR 2020 SC 456 held..."
        citations = preprocessor.extract_citations(text)
        
        assert len(citations) >= 1
    
    def test_segment_sentences(self):
        """Test sentence segmentation."""
        preprocessor = LegalTextPreprocessor()
        
        text = "First sentence. Second sentence. Third sentence."
        sentences = preprocessor.segment_sentences(text)
        
        assert len(sentences) == 3
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        preprocessor = LegalTextPreprocessor()
        
        texts = ["  Text one  ", "  Text two  "]
        processed = preprocessor.preprocess_batch(texts)
        
        assert len(processed) == 2
        assert all(not t.startswith(" ") for t in processed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
