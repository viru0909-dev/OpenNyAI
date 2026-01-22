"""
Legal Data Loader
=================
Utilities for loading legal documents from various sources.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger


class LegalDataLoader:
    """
    Data loader for legal documents including PDFs, text files, and JSON.
    
    Supports:
    - Court judgments
    - FIRs (First Information Reports)
    - Legal notices
    - Case diaries
    """
    
    def __init__(self, data_dir: Union[str, Path] = "./data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Base directory containing the data files.
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized LegalDataLoader with data_dir: {self.data_dir}")
    
    def load_json(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            List of dictionaries containing the data.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return data
    
    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file.
            
        Returns:
            Pandas DataFrame containing the data.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    
    def load_text_files(self, directory: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Load all text files from a directory.
        
        Args:
            directory: Directory containing text files.
            
        Returns:
            List of dictionaries with 'filename' and 'content' keys.
        """
        directory = Path(directory)
        documents = []
        
        for file_path in directory.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append({
                'filename': file_path.name,
                'content': content
            })
        
        logger.info(f"Loaded {len(documents)} text files from {directory}")
        return documents
    
    def load_pdf(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Extracted text content.
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        logger.info(f"Extracted {len(text)} characters from {file_path}")
        return text
    
    def save_processed_data(
        self,
        data: Union[List[Dict], pd.DataFrame],
        filename: str,
        format: str = "json"
    ) -> Path:
        """
        Save processed data to the processed directory.
        
        Args:
            data: Data to save (list of dicts or DataFrame).
            filename: Output filename (without extension).
            format: Output format ('json' or 'csv').
            
        Returns:
            Path to the saved file.
        """
        if format == "json":
            output_path = self.processed_dir / f"{filename}.json"
            if isinstance(data, pd.DataFrame):
                data = data.to_dict(orient='records')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "csv":
            output_path = self.processed_dir / f"{filename}.csv"
            if isinstance(data, list):
                data = pd.DataFrame(data)
            data.to_csv(output_path, index=False, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved processed data to {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    loader = LegalDataLoader()
    print(f"Data loader initialized. Raw dir: {loader.raw_dir}")
