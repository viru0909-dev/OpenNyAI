#!/usr/bin/env python3
"""
Inference Script for OpenNyAI Models
=====================================
Command-line script for running predictions with trained models.

Usage:
    python scripts/predict.py --model models/trained/ner --input "Your legal text here"
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LegalNERModel, LegalDocumentClassifier, LegalSummarizer
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run predictions with OpenNyAI models"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ner", "classifier", "summarizer"],
        default="ner",
        help="Type of model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input text for prediction"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input file (one text per line or JSON)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save predictions"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "text"],
        default="json",
        help="Output format"
    )
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level="INFO")
    
    print(f"\n{'='*60}")
    print("OpenNyAI Model Inference")
    print(f"{'='*60}")
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
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
    
    # Get inputs
    texts = []
    if args.input:
        texts = [args.input]
    elif args.input_file:
        input_path = Path(args.input_file)
        if input_path.suffix == ".json":
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                texts = [item["text"] for item in data]
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Please provide --input or --input-file")
        sys.exit(1)
    
    # Run predictions
    predictions = []
    for i, text in enumerate(texts):
        print(f"Processing {i+1}/{len(texts)}...")
        
        if args.model_type == "ner":
            result = model.predict(text)
        elif args.model_type == "classifier":
            result = model.predict(text, return_probabilities=True)
        elif args.model_type == "summarizer":
            result = {"summary": model.summarize(text)}
        
        predictions.append({
            "input": text[:100] + "..." if len(text) > 100 else text,
            "prediction": result
        })
    
    # Output results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"\nPredictions saved to: {args.output_file}")
    else:
        print("\nPredictions:")
        print(json.dumps(predictions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
