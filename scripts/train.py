#!/usr/bin/env python3
"""
Training Script for OpenNyAI Models
====================================
Command-line script for training legal NLP models.

Usage:
    python scripts/train.py --model ner --config configs/training_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LegalNERModel, LegalDocumentClassifier, LegalSummarizer
from src.training import ModelTrainer
from src.utils import setup_logging, set_seed, Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train OpenNyAI legal NLP models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["ner", "classifier", "summarizer"],
        default="ner",
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.json",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/processed/val.json",
        help="Path to validation data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trained",
        help="Directory to save trained model"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="ai4bharat/indic-bert",
        help="Base pretrained model name"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level="INFO", log_file="logs/training.log")
    
    # Set random seed
    set_seed(args.seed)
    
    print(f"\n{'='*60}")
    print("OpenNyAI Model Training")
    print(f"{'='*60}")
    print(f"Model type: {args.model}")
    print(f"Base model: {args.base_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Create model based on type
    if args.model == "ner":
        model = LegalNERModel(model_name=args.base_model)
    elif args.model == "classifier":
        model = LegalDocumentClassifier(model_name=args.base_model)
    elif args.model == "summarizer":
        model = LegalSummarizer(model_name=args.base_model)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Create training configuration
    config = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_data=args.train_data,
        val_data=args.val_data if Path(args.val_data).exists() else None,
        config=config,
        output_dir=args.output_dir,
        experiment_name=f"opennyai_{args.model}"
    )
    
    # Train
    if args.early_stopping:
        metrics = trainer.train_with_early_stopping(patience=args.patience)
    else:
        metrics = trainer.train()
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Final metrics: {metrics}")
    print(f"Model saved to: {args.output_dir}")
    

if __name__ == "__main__":
    main()
