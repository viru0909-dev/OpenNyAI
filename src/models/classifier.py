"""
Legal Document Classifier
==========================
Classification model for legal documents.
"""

from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger


class LegalDocumentClassifier:
    """
    Classification model for legal documents.
    
    Classifies documents by:
    - Case type (Civil, Criminal, Constitutional, etc.)
    - Court level (Supreme, High, District, etc.)
    - Subject matter (Property, Family, Criminal, Labor, etc.)
    """
    
    # Case type labels
    CASE_TYPES = [
        "Civil",
        "Criminal",
        "Constitutional",
        "Administrative",
        "Family",
        "Labor",
        "Tax",
        "Intellectual Property",
        "Environmental",
        "Consumer",
        "Other"
    ]
    
    # Court levels
    COURT_LEVELS = [
        "Supreme Court",
        "High Court",
        "District Court",
        "Sessions Court",
        "Tribunal",
        "Magistrate Court",
        "Special Court",
        "Other"
    ]
    
    def __init__(
        self,
        model_name: str = "ai4bharat/indic-bert",
        classification_type: str = "case_type",
        device: Optional[str] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Pretrained model name from Hugging Face.
            classification_type: Type of classification ('case_type' or 'court_level').
            device: Device to run the model on.
        """
        self.model_name = model_name
        self.classification_type = classification_type
        
        # Set labels based on classification type
        if classification_type == "case_type":
            self.labels = self.CASE_TYPES
        elif classification_type == "court_level":
            self.labels = self.COURT_LEVELS
        else:
            raise ValueError(f"Unknown classification type: {classification_type}")
        
        self.num_labels = len(self.labels)
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        logger.info(
            f"Initialized LegalDocumentClassifier for {classification_type} "
            f"with {self.num_labels} labels"
        )
    
    def load_model(self):
        """Load the pretrained classification model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )
        
        logger.info(f"Loading classification model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)
        self._is_loaded = True
        
        logger.info("Classification model loaded successfully")
    
    def predict(
        self,
        text: str,
        return_probabilities: bool = False
    ) -> Dict:
        """
        Classify a legal document.
        
        Args:
            text: Input document text.
            return_probabilities: Whether to return all class probabilities.
            
        Returns:
            Dictionary with 'label', 'confidence', and optionally 'probabilities'.
        """
        if not self._is_loaded:
            self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            predicted_id = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_id].item()
        
        result = {
            "label": self.id2label[predicted_id],
            "confidence": confidence
        }
        
        if return_probabilities:
            result["probabilities"] = {
                self.id2label[i]: prob.item()
                for i, prob in enumerate(probabilities)
            }
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        return_probabilities: bool = False
    ) -> List[Dict]:
        """
        Classify multiple documents.
        
        Args:
            texts: List of document texts.
            return_probabilities: Whether to return all class probabilities.
            
        Returns:
            List of classification results.
        """
        return [
            self.predict(text, return_probabilities)
            for text in texts
        ]
    
    def get_top_k_predictions(
        self,
        text: str,
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top-k predictions for a document.
        
        Args:
            text: Input document text.
            k: Number of top predictions to return.
            
        Returns:
            List of (label, probability) tuples.
        """
        result = self.predict(text, return_probabilities=True)
        probs = result["probabilities"]
        
        # Sort by probability
        sorted_probs = sorted(
            probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_probs[:k]
    
    def save_model(self, output_dir: str):
        """Save the model to a directory."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Nothing to save.")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Classification model saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    classifier = LegalDocumentClassifier(classification_type="case_type")
    print(f"Classifier initialized for: {classifier.classification_type}")
    print(f"Labels: {classifier.labels}")
