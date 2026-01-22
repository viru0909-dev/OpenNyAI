"""
Rhetorical Role Labeling Model
===============================
Model for segmenting legal judgments into functional parts.

Based on OpenNyAI's 13 rhetorical role schema for Indian court judgments.
"""

from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger


class RhetoricalRoleLabeler:
    """
    Rhetorical Role Labeling (RRL) for legal documents.
    
    Segments judgments into 13 functional parts:
    - PREAMBLE: Case header, parties, court info
    - FACTS: Factual background
    - ISSUE: Legal questions to be decided
    - ARGUMENT_PETITIONER: Arguments by petitioner
    - ARGUMENT_RESPONDENT: Arguments by respondent
    - ANALYSIS: Court's examination
    - STATUTE: Statutory provisions discussed
    - PRECEDENT_RELIED: Cases cited and followed
    - PRECEDENT_NOT_RELIED: Cases distinguished
    - RATIO: Legal principle established
    - RULING_LOWER_COURT: Lower court's decision
    - RULING_PRESENT_COURT: Current court's decision
    - NONE: Non-classifiable content
    """
    
    # Rhetorical Role Labels
    RRL_LABELS = [
        "PREAMBLE",
        "FACTS",
        "ISSUE",
        "ARGUMENT_PETITIONER",
        "ARGUMENT_RESPONDENT",
        "ANALYSIS",
        "STATUTE",
        "PRECEDENT_RELIED",
        "PRECEDENT_NOT_RELIED",
        "RATIO",
        "RULING_LOWER_COURT",
        "RULING_PRESENT_COURT",
        "NONE"
    ]
    
    # Valid label transitions (CRF constraint)
    VALID_TRANSITIONS = {
        "PREAMBLE": ["PREAMBLE", "FACTS", "ISSUE"],
        "FACTS": ["FACTS", "ISSUE", "ARGUMENT_PETITIONER", "ANALYSIS"],
        "ISSUE": ["ISSUE", "ARGUMENT_PETITIONER", "ARGUMENT_RESPONDENT", "ANALYSIS"],
        "ARGUMENT_PETITIONER": ["ARGUMENT_PETITIONER", "ARGUMENT_RESPONDENT", "ANALYSIS"],
        "ARGUMENT_RESPONDENT": ["ARGUMENT_RESPONDENT", "ARGUMENT_PETITIONER", "ANALYSIS"],
        "ANALYSIS": ["ANALYSIS", "STATUTE", "PRECEDENT_RELIED", "PRECEDENT_NOT_RELIED", "RATIO"],
        "STATUTE": ["STATUTE", "ANALYSIS", "PRECEDENT_RELIED"],
        "PRECEDENT_RELIED": ["PRECEDENT_RELIED", "PRECEDENT_NOT_RELIED", "ANALYSIS", "RATIO"],
        "PRECEDENT_NOT_RELIED": ["PRECEDENT_NOT_RELIED", "PRECEDENT_RELIED", "ANALYSIS"],
        "RATIO": ["RATIO", "RULING_LOWER_COURT", "RULING_PRESENT_COURT"],
        "RULING_LOWER_COURT": ["RULING_LOWER_COURT", "ANALYSIS", "RATIO", "RULING_PRESENT_COURT"],
        "RULING_PRESENT_COURT": ["RULING_PRESENT_COURT"],
        "NONE": RRL_LABELS,  # NONE can transition to any
    }
    
    def __init__(
        self,
        model_name: str = "law-ai/InLegalBERT",
        use_crf: bool = True,
        use_bilstm: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the RRL model.
        
        Args:
            model_name: Base transformer model (InLegalBERT recommended).
            use_crf: Whether to use CRF layer for label constraints.
            use_bilstm: Whether to use BiLSTM for sequence modeling.
            device: Device to run the model on.
        """
        self.model_name = model_name
        self.use_crf = use_crf
        self.use_bilstm = use_bilstm
        self.num_labels = len(self.RRL_LABELS)
        
        self.label2id = {label: i for i, label in enumerate(self.RRL_LABELS)}
        self.id2label = {i: label for i, label in enumerate(self.RRL_LABELS)}
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        logger.info(f"Initialized RRL model with {model_name}, CRF={use_crf}, BiLSTM={use_bilstm}")
    
    def load_model(self):
        """Load the pretrained model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )
        
        logger.info(f"Loading RRL model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        
        # Build classification architecture
        hidden_size = self.encoder.config.hidden_size
        
        if self.use_bilstm:
            self.lstm = torch.nn.LSTM(
                hidden_size,
                hidden_size // 2,
                bidirectional=True,
                batch_first=True,
                num_layers=2,
                dropout=0.1
            )
        
        self.classifier = torch.nn.Linear(hidden_size, self.num_labels)
        
        if self.use_crf:
            try:
                from torchcrf import CRF
                self.crf = CRF(self.num_labels, batch_first=True)
            except ImportError:
                logger.warning("torchcrf not available, disabling CRF")
                self.use_crf = False
        
        # Move to device
        self.encoder.to(self.device)
        if self.use_bilstm:
            self.lstm.to(self.device)
        self.classifier.to(self.device)
        if self.use_crf:
            self.crf.to(self.device)
        
        self._is_loaded = True
        logger.info("RRL model loaded successfully")
    
    def _encode_sentences(self, sentences: List[str]) -> torch.Tensor:
        """Encode sentences using the transformer."""
        # Get [CLS] embedding for each sentence
        embeddings = []
        
        for sent in sentences:
            inputs = self.tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding="max_length"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embedding)
        
        return torch.cat(embeddings, dim=0).unsqueeze(0)  # [1, num_sentences, hidden_size]
    
    def predict(self, sentences: List[str]) -> List[Dict[str, str]]:
        """
        Predict rhetorical roles for sentences in a judgment.
        
        Args:
            sentences: List of sentences from a judgment.
            
        Returns:
            List of dicts with 'sentence', 'role', 'confidence' keys.
        """
        if not self._is_loaded:
            self.load_model()
        
        self.encoder.eval()
        if self.use_bilstm:
            self.lstm.eval()
        self.classifier.eval()
        
        # Encode sentences
        sentence_embeddings = self._encode_sentences(sentences)
        
        # Apply BiLSTM if enabled
        if self.use_bilstm:
            lstm_out, _ = self.lstm(sentence_embeddings)
            features = lstm_out
        else:
            features = sentence_embeddings
        
        # Get logits
        logits = self.classifier(features)  # [1, num_sentences, num_labels]
        
        # Decode predictions
        if self.use_crf:
            predictions = self.crf.decode(logits)[0]
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        else:
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            predictions = torch.argmax(probs, dim=-1).tolist()
        
        # Build results
        results = []
        for i, (sent, pred_id) in enumerate(zip(sentences, predictions)):
            confidence = probs[i, pred_id].item() if not self.use_crf else 1.0
            results.append({
                "sentence": sent,
                "role": self.id2label[pred_id],
                "confidence": confidence
            })
        
        return results
    
    def predict_document(self, text: str) -> List[Dict[str, str]]:
        """
        Predict rhetorical roles for an entire document.
        
        Args:
            text: Full judgment text.
            
        Returns:
            List of sentence-role predictions.
        """
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return []
        
        return self.predict(sentences)
    
    def get_summary_by_roles(
        self,
        predictions: List[Dict[str, str]],
        roles_to_include: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Group sentences by their rhetorical roles.
        
        Args:
            predictions: Output from predict().
            roles_to_include: Specific roles to include (default: all).
            
        Returns:
            Dictionary mapping roles to lists of sentences.
        """
        if roles_to_include is None:
            roles_to_include = self.RRL_LABELS
        
        grouped = {role: [] for role in roles_to_include}
        
        for pred in predictions:
            if pred["role"] in roles_to_include:
                grouped[pred["role"]].append(pred["sentence"])
        
        # Remove empty roles
        return {k: v for k, v in grouped.items() if v}
    
    def generate_structured_summary(
        self,
        predictions: List[Dict[str, str]]
    ) -> str:
        """
        Generate a structured summary based on rhetorical roles.
        
        Args:
            predictions: Output from predict().
            
        Returns:
            Formatted summary string.
        """
        grouped = self.get_summary_by_roles(predictions)
        
        summary_parts = []
        
        # Issue
        if "ISSUE" in grouped:
            issues = " ".join(grouped["ISSUE"][:2])
            summary_parts.append(f"**ISSUE:** {issues}")
        
        # Key Facts
        if "FACTS" in grouped:
            facts = " ".join(grouped["FACTS"][:3])
            summary_parts.append(f"**FACTS:** {facts}")
        
        # Precedents Relied
        if "PRECEDENT_RELIED" in grouped:
            precedents = " ".join(grouped["PRECEDENT_RELIED"][:2])
            summary_parts.append(f"**PRECEDENTS RELIED:** {precedents}")
        
        # Ratio
        if "RATIO" in grouped:
            ratio = " ".join(grouped["RATIO"][:2])
            summary_parts.append(f"**RATIO:** {ratio}")
        
        # Final Ruling
        if "RULING_PRESENT_COURT" in grouped:
            ruling = " ".join(grouped["RULING_PRESENT_COURT"][:2])
            summary_parts.append(f"**RULING:** {ruling}")
        
        return "\n\n".join(summary_parts)
    
    def save_model(self, output_dir: str):
        """Save the model to a directory."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.encoder.save_pretrained(f"{output_dir}/encoder")
        self.tokenizer.save_pretrained(f"{output_dir}/encoder")
        
        torch.save({
            'classifier': self.classifier.state_dict(),
            'lstm': self.lstm.state_dict() if self.use_bilstm else None,
            'crf': self.crf.state_dict() if self.use_crf else None,
        }, f"{output_dir}/layers.pt")
        
        logger.info(f"RRL model saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    model = RhetoricalRoleLabeler()
    print(f"RRL Model initialized with {len(model.RRL_LABELS)} labels:")
    for label in model.RRL_LABELS:
        print(f"  - {label}")
