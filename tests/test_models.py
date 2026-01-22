"""
Tests for model modules.
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LegalNERModel, LegalDocumentClassifier, LegalSummarizer


class TestLegalNERModel:
    """Tests for LegalNERModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = LegalNERModel()
        
        assert model.model_name == "ai4bharat/indic-bert"
        assert len(model.LEGAL_LABELS) == 21
        assert model.label2id["O"] == 0
    
    def test_label_mappings(self):
        """Test label-to-id and id-to-label mappings."""
        model = LegalNERModel()
        
        # Check bidirectional mapping
        for label in model.LEGAL_LABELS:
            label_id = model.label2id[label]
            assert model.id2label[label_id] == label


class TestLegalDocumentClassifier:
    """Tests for LegalDocumentClassifier class."""
    
    def test_initialization_case_type(self):
        """Test classifier initialization for case types."""
        classifier = LegalDocumentClassifier(classification_type="case_type")
        
        assert classifier.classification_type == "case_type"
        assert "Civil" in classifier.labels
        assert "Criminal" in classifier.labels
    
    def test_initialization_court_level(self):
        """Test classifier initialization for court levels."""
        classifier = LegalDocumentClassifier(classification_type="court_level")
        
        assert classifier.classification_type == "court_level"
        assert "Supreme Court" in classifier.labels
        assert "High Court" in classifier.labels
    
    def test_invalid_classification_type(self):
        """Test error handling for invalid classification type."""
        with pytest.raises(ValueError):
            LegalDocumentClassifier(classification_type="invalid")


class TestLegalSummarizer:
    """Tests for LegalSummarizer class."""
    
    def test_initialization(self):
        """Test summarizer initialization."""
        summarizer = LegalSummarizer()
        
        assert summarizer.model_name == "facebook/bart-large-cnn"
        assert summarizer.max_input_length == 1024
        assert summarizer.max_output_length == 256
    
    def test_extract_key_points(self):
        """Test key point extraction."""
        summarizer = LegalSummarizer()
        
        text = (
            "This is the first important point about the case. "
            "The second point discusses the legal provisions. "
            "Third, we consider the evidence presented. "
            "Fourth, the court examines witness testimony. "
            "Finally, the judgment is pronounced."
        )
        
        key_points = summarizer.extract_key_points(text, num_points=3)
        
        assert len(key_points) == 3
        assert all(isinstance(p, str) for p in key_points)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
