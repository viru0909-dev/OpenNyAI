"""
Model Trainer
=============
Training pipeline for legal NLP models.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger


class ModelTrainer:
    """
    Training pipeline for legal NLP models.
    
    Supports:
    - NER model training
    - Classification model training
    - Summarization model fine-tuning
    - Experiment tracking with MLflow/W&B
    """
    
    def __init__(
        self,
        model: Any,
        train_data: str,
        val_data: Optional[str] = None,
        config: Optional[Dict] = None,
        output_dir: str = "./models",
        experiment_name: str = "legal_nlp"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model instance to train.
            train_data: Path to training data.
            val_data: Path to validation data.
            config: Training configuration dictionary.
            output_dir: Directory to save trained models.
            experiment_name: Name for experiment tracking.
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        # Default configuration
        self.config = {
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 10,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "max_length": 512,
            "gradient_accumulation_steps": 1,
            "fp16": torch.cuda.is_available(),
            "logging_steps": 100,
            "eval_steps": 500,
            "save_steps": 1000,
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelTrainer for {experiment_name}")
    
    def _load_data(self):
        """Load training and validation data."""
        import json
        
        with open(self.train_data, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        val_data = None
        if self.val_data:
            with open(self.val_data, 'r', encoding='utf-8') as f:
                val_data = json.load(f)
        
        logger.info(f"Loaded {len(train_data)} training samples")
        if val_data:
            logger.info(f"Loaded {len(val_data)} validation samples")
        
        return train_data, val_data
    
    def _setup_training_args(self):
        """Set up training arguments for Hugging Face Trainer."""
        try:
            from transformers import TrainingArguments
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            warmup_steps=self.config["warmup_steps"],
            weight_decay=self.config["weight_decay"],
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=self.config["logging_steps"],
            evaluation_strategy="steps" if self.val_data else "no",
            eval_steps=self.config["eval_steps"] if self.val_data else None,
            save_steps=self.config["save_steps"],
            save_total_limit=3,
            load_best_model_at_end=bool(self.val_data),
            fp16=self.config["fp16"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            report_to=["tensorboard"],
        )
        
        return training_args
    
    def train(self):
        """
        Run the training pipeline.
        
        Returns:
            Training metrics.
        """
        try:
            from transformers import Trainer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )
        
        logger.info("Starting training...")
        
        # Load model if not already loaded
        if not self.model._is_loaded:
            self.model.load_model()
        
        # Load data
        train_data, val_data = self._load_data()
        
        # Set up training arguments
        training_args = self._setup_training_args()
        
        # Create Hugging Face Trainer
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=self.model.tokenizer,
        )
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        self.model.save_model(str(final_model_path))
        
        logger.info(f"Training completed. Model saved to {final_model_path}")
        
        return train_result.metrics
    
    def train_with_early_stopping(
        self,
        patience: int = 3,
        metric: str = "eval_loss"
    ):
        """
        Train with early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement.
            metric: Metric to monitor for early stopping.
            
        Returns:
            Training metrics.
        """
        try:
            from transformers import EarlyStoppingCallback, Trainer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )
        
        if not self.val_data:
            raise ValueError("Validation data required for early stopping")
        
        logger.info(f"Training with early stopping (patience={patience})")
        
        # Load model if not already loaded
        if not self.model._is_loaded:
            self.model.load_model()
        
        # Load data
        train_data, val_data = self._load_data()
        
        # Set up training arguments
        training_args = self._setup_training_args()
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = metric
        
        # Create Trainer with early stopping
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=self.model.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "best_model"
        self.model.save_model(str(final_model_path))
        
        logger.info(f"Training completed with early stopping. Model saved to {final_model_path}")
        
        return train_result.metrics


if __name__ == "__main__":
    print("ModelTrainer module ready for use")
