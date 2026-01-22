"""
Legal Named Entity Recognition Model
=====================================
NER model for identifying legal entities in judicial documents.
"""

from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger


class LegalNERModel:
    """
    Named Entity Recognition model for legal documents.
    
    Identifies entities such as:
    - PETITIONER: Person/Organization filing the case
    - RESPONDENT: Person/Organization defending
    - JUDGE: Presiding judge
    - ADVOCATE: Legal representatives
    - COURT: Court name
    - DATE: Important dates
    - STATUTE: Legal provisions and acts
    - CASE_NUMBER: Case reference numbers
    - ORGANIZATION: Companies, institutions
    - LOCATION: Places mentioned
    """
    
    # Legal entity labels
    LEGAL_LABELS = [
        "O",  # Outside any entity
        "B-PETITIONER", "I-PETITIONER",
        "B-RESPONDENT", "I-RESPONDENT",
        "B-JUDGE", "I-JUDGE",
        "B-ADVOCATE", "I-ADVOCATE",
        "B-COURT", "I-COURT",
        "B-DATE", "I-DATE",
        "B-STATUTE", "I-STATUTE",
        "B-CASE_NUMBER", "I-CASE_NUMBER",
        "B-ORGANIZATION", "I-ORGANIZATION",
        "B-LOCATION", "I-LOCATION",
    ]
    
    def __init__(
        self,
        model_name: str = "ai4bharat/indic-bert",
        num_labels: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Legal NER model.
        
        Args:
            model_name: Pretrained model name from Hugging Face.
            num_labels: Number of entity labels (auto-detected if None).
            device: Device to run the model on ('cuda', 'cpu', or None for auto).
        """
        self.model_name = model_name
        self.num_labels = num_labels or len(self.LEGAL_LABELS)
        self.label2id = {label: i for i, label in enumerate(self.LEGAL_LABELS)}
        self.id2label = {i: label for i, label in enumerate(self.LEGAL_LABELS)}
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        logger.info(f"Initialized LegalNERModel with {model_name} on {self.device}")
    
    def load_model(self):
        """Load the pretrained model and tokenizer."""
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )
        
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)
        self._is_loaded = True
        
        logger.info("Model loaded successfully")
    
    def predict(self, text: str) -> List[Dict[str, str]]:
        """
        Predict named entities in the given text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            List of entities with 'text', 'label', 'start', 'end' keys.
        """
        if not self._is_loaded:
            self.load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
        
        # Extract entities
        entities = []
        current_entity = None
        
        for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:  # Special token
                continue
            
            label = self.id2label[pred]
            
            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    "text": text[start:end],
                    "label": entity_type,
                    "start": start,
                    "end": end
                }
            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    # Continue current entity
                    current_entity["text"] = text[current_entity["start"]:end]
                    current_entity["end"] = end
            else:
                # Outside entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Don't forget last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict[str, str]]]:
        """
        Predict entities for multiple texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            List of entity lists for each text.
        """
        return [self.predict(text) for text in texts]
    
    def save_model(self, output_dir: str):
        """
        Save the model to a directory.
        
        Args:
            output_dir: Directory to save the model.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Nothing to save.")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_dir: str, device: Optional[str] = None):
        """
        Load a fine-tuned model from a directory.
        
        Args:
            model_dir: Directory containing the saved model.
            device: Device to load the model on.
            
        Returns:
            LegalNERModel instance.
        """
        instance = cls(model_name=model_dir, device=device)
        instance.load_model()
        return instance


if __name__ == "__main__":
    # Example usage
    model = LegalNERModel()
    print(f"Model initialized with {len(model.LEGAL_LABELS)} labels")
    print(f"Labels: {model.LEGAL_LABELS}")
