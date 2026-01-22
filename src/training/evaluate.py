"""
Model Evaluation
================
Evaluation utilities for legal NLP models.
"""

from typing import Any, Dict, List, Optional

from loguru import logger


class Evaluator:
    """
    Evaluation utilities for legal NLP models.
    
    Supports:
    - NER evaluation (precision, recall, F1)
    - Classification evaluation (accuracy, macro/micro F1)
    - Summarization evaluation (ROUGE scores)
    """
    
    def __init__(self, model: Any, test_data: str):
        """
        Initialize the evaluator.
        
        Args:
            model: Model instance to evaluate.
            test_data: Path to test data.
        """
        self.model = model
        self.test_data = test_data
        
        logger.info("Initialized Evaluator")
    
    def evaluate_ner(self) -> Dict[str, float]:
        """
        Evaluate NER model performance.
        
        Returns:
            Dictionary with precision, recall, F1 per entity type.
        """
        try:
            from seqeval.metrics import (
                classification_report,
                f1_score,
                precision_score,
                recall_score,
            )
        except ImportError:
            raise ImportError(
                "seqeval is required. Install with: pip install seqeval"
            )
        
        import json
        
        # Load test data
        with open(self.test_data, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
        
        all_true_labels = []
        all_pred_labels = []
        
        for sample in test_samples:
            text = sample["text"]
            true_labels = sample["labels"]
            
            # Get predictions
            predictions = self.model.predict(text)
            
            # Convert to label sequences (simplified)
            pred_labels = ["O"] * len(true_labels)  # Placeholder
            
            all_true_labels.append(true_labels)
            all_pred_labels.append(pred_labels)
        
        # Calculate metrics
        results = {
            "precision": precision_score(all_true_labels, all_pred_labels),
            "recall": recall_score(all_true_labels, all_pred_labels),
            "f1": f1_score(all_true_labels, all_pred_labels),
            "report": classification_report(all_true_labels, all_pred_labels)
        }
        
        logger.info(f"NER Evaluation - F1: {results['f1']:.4f}")
        return results
    
    def evaluate_classification(self) -> Dict[str, float]:
        """
        Evaluate classification model performance.
        
        Returns:
            Dictionary with accuracy and F1 scores.
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score
        )
        import json
        
        # Load test data
        with open(self.test_data, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
        
        true_labels = []
        pred_labels = []
        
        for sample in test_samples:
            text = sample["text"]
            true_label = sample["label"]
            
            # Get prediction
            prediction = self.model.predict(text)
            pred_labels.append(prediction["label"])
            true_labels.append(true_label)
        
        # Calculate metrics
        results = {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "f1_macro": f1_score(true_labels, pred_labels, average="macro"),
            "f1_micro": f1_score(true_labels, pred_labels, average="micro"),
            "report": classification_report(true_labels, pred_labels)
        }
        
        logger.info(f"Classification Evaluation - Accuracy: {results['accuracy']:.4f}")
        return results
    
    def evaluate_summarization(self) -> Dict[str, float]:
        """
        Evaluate summarization model using ROUGE scores.
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores.
        """
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError(
                "rouge-score is required. Install with: pip install rouge-score"
            )
        
        import json
        
        # Load test data
        with open(self.test_data, 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
        
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for sample in test_samples:
            text = sample["text"]
            reference = sample["summary"]
            
            # Generate summary
            prediction = self.model.summarize(text)
            
            # Calculate ROUGE scores
            scores = scorer.score(reference, prediction)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        results = {
            "rouge1": sum(rouge1_scores) / len(rouge1_scores),
            "rouge2": sum(rouge2_scores) / len(rouge2_scores),
            "rougeL": sum(rougeL_scores) / len(rougeL_scores),
        }
        
        logger.info(f"Summarization Evaluation - ROUGE-L: {results['rougeL']:.4f}")
        return results


if __name__ == "__main__":
    print("Evaluator module ready for use")
