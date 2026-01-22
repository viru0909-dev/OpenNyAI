#!/usr/bin/env python3
"""
Evaluation Script for OpenNyAI Models
======================================
Command-line script for evaluating trained models.

Usage:
    python scripts/evaluate.py --model-type ner --model-path models/trained/ner --test-data data/processed/test.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LegalNERModel, LegalDocumentClassifier, LegalSummarizer
from src.training import Evaluator
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate OpenNyAI models"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ner", "classifier", "summarizer"],
        required=True,
        help="Type of model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save evaluation results"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level="INFO")
    
    print(f"\n{'='*60}")
    print("OpenNyAI Model Evaluation")
    print(f"{'='*60}")
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"{'='*60}\n")
    
    # Load model
    if args.model_type == "ner":
        model = LegalNERModel.from_pretrained(args.model_path)
    elif args.model_type == "classifier":
        model = LegalDocumentClassifier(model_name=args.model_path)
        model.load_model()
    elif args.model_type == "summarizer":
        model = LegalSummarizer(model_name=args.model_path)
        model.load_model()
    
    # Create evaluator
    evaluator = Evaluator(model=model, test_data=args.test_data)
    
    # Run evaluation
    if args.model_type == "ner":
        results = evaluator.evaluate_ner()
    elif args.model_type == "classifier":
        results = evaluator.evaluate_classification()
    elif args.model_type == "summarizer":
        results = evaluator.evaluate_summarization()
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    for key, value in results.items():
        if key != "report":
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    if "report" in results:
        print(f"\nDetailed Report:\n{results['report']}")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
