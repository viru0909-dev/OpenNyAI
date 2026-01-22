# 🏛️ OpenNyAI - Sovereign Legal Intelligence for Indian Judiciary

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/NLP-InLegalBERT-orange.svg" alt="NLP">
  <img src="https://img.shields.io/badge/LLM-Llama_3-purple.svg" alt="LLM">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow.svg" alt="Status">
</p>

## 📋 Overview

**OpenNyAI** is an open-source Natural Language Processing (NLP) initiative designed to build **Sovereign Legal AI** for the Indian Judicial System. With over **5.3 crore pending cases**, the Indian judiciary faces an unprecedented operational challenge. This project develops custom, domain-specific NLP models that understand the linguistic nuances of Indian Legal English, IPC, CrPC, and multilingual judgments.

> ⚠️ **Sovereign AI Philosophy**: This project prioritizes indigenous, self-hosted models over generic Western-centric LLMs to ensure data sovereignty, linguistic alignment, and reduced hallucination in high-stakes legal environments.

## 🎯 Project Objectives

| Task | Model Architecture | Purpose |
|------|-------------------|---------|
| **Legal NER** | InLegalBERT | Extract 14 entity types (PETITIONER, RESPONDENT, STATUTE, etc.) |
| **Rhetorical Role Labeling** | BiLSTM-CRF + Transformer | Segment judgments into 13 functional parts |
| **Case Summarization** | Llama 3 + RAG | Generate legally sound, structured summaries |
| **Legal Reasoning** | Instruction-Tuned Llama 3 | Draft arguments, simplify legal language |
| **Judgment Prediction** | InLegalBERT Classifier | Predict case outcomes based on ILDC corpus |

## 🏗️ Project Structure

```
OpenNyAI/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup file
├── .env.example                 # Environment variables template
│
├── data/
│   ├── raw/                     # Raw legal documents (Indian Kanoon, e-Courts)
│   ├── processed/               # Preprocessed data (CoNLL format, JSONL)
│   ├── annotations/             # Annotated datasets (NER, RRL)
│   └── corpora/                 # Standard datasets (ILDC, InJudgements, Aalap)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── scraper.py           # Indian Kanoon & e-Courts scraping
│   │   ├── loader.py            # Data loading utilities
│   │   ├── preprocessor.py      # Regex-based cleaning & normalization
│   │   └── chunker.py           # Semantic chunking for long documents
│   │
│   ├── models/
│   │   ├── inlegalbert.py       # InLegalBERT base encoder
│   │   ├── ner_model.py         # Legal Named Entity Recognition (14 classes)
│   │   ├── rrl_model.py         # Rhetorical Role Labeling (13 roles)
│   │   ├── summarizer.py        # Extractive + Abstractive Summarization
│   │   ├── classifier.py        # Document Classification
│   │   └── llama_reasoning.py   # Llama 3 instruction-tuned reasoning
│   │
│   ├── training/
│   │   ├── trainer.py           # HuggingFace Trainer wrapper
│   │   ├── lora_finetuning.py   # LoRA/QLoRA for Llama 3
│   │   └── evaluate.py          # seqeval, ROUGE, classification metrics
│   │
│   ├── rag/
│   │   ├── vectorstore.py       # Milvus/Chroma integration
│   │   ├── retriever.py         # Precedent retrieval
│   │   └── rag_pipeline.py      # RAG for evidence-based reasoning
│   │
│   └── utils/
│       ├── config.py            # Configuration management
│       ├── regex_patterns.py    # Indian legal citation patterns
│       └── helpers.py           # Utility functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_inlegalbert_ner.ipynb
│   ├── 04_rhetorical_roles.ipynb
│   └── 05_llama_finetuning.ipynb
│
├── scripts/
│   ├── scrape_indian_kanoon.py  # Data collection script
│   ├── train.py                 # Training script
│   ├── predict.py               # Inference script
│   └── evaluate.py              # Evaluation script
│
├── configs/
│   ├── model_config.yaml        # Model configurations
│   ├── training_config.yaml     # Training parameters
│   └── lora_config.yaml         # LoRA hyperparameters
│
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_regex.py
```

## 🧠 Model Architectures

### 1. Legal Named Entity Recognition (14 Entity Classes)

Based on the OpenNyAI specification:

| Entity Type | Description | Example |
|------------|-------------|---------|
| `COURT` | Court name | "Supreme Court of India" |
| `PETITIONER` | Person/Org filing case | "Kesavananda Bharati" |
| `RESPONDENT` | Person/Org defending | "State of Kerala" |
| `JUDGE` | Presiding judge | "Hon'ble Justice DY Chandrachud" |
| `LAWYER` | Legal representatives | "Adv. Fali S. Nariman" |
| `STATUTE` | Legal Act | "Indian Penal Code" |
| `PROVISION` | Section/Article | "Section 302", "Article 21" |
| `PRECEDENT` | Case citations | "AIR 1973 SC 1461" |
| `CASE_NUMBER` | Case reference | "Writ Petition (C) No. 135/2019" |
| `DATE` | Important dates | "14th February 2024" |
| `GPE` | Geopolitical entity | "Maharashtra", "New Delhi" |
| `ORG` | Organization | "CBI", "RBI" |
| `WITNESS` | Witnesses | "PW-1", "DW-3" |
| `EVIDENCE` | Evidence references | "Exhibit P-1" |

### 2. Rhetorical Role Labeling (13 Roles)

| Role | Description |
|------|-------------|
| `PREAMBLE` | Case header, parties, court info |
| `FACTS` | Factual background of the case |
| `ISSUE` | Legal questions to be decided |
| `ARGUMENT_PETITIONER` | Arguments by petitioner's counsel |
| `ARGUMENT_RESPONDENT` | Arguments by respondent's counsel |
| `ANALYSIS` | Court's examination of issues |
| `STATUTE` | Statutory provisions discussed |
| `PRECEDENT_RELIED` | Cases cited and followed |
| `PRECEDENT_NOT_RELIED` | Cases cited but distinguished |
| `RATIO` | Legal principle established |
| `RULING_LOWER_COURT` | Lower court's decision |
| `RULING_PRESENT_COURT` | Current court's decision |
| `NONE` | Non-classifiable content |

### 3. InLegalBERT - Sovereign Foundation

```python
from transformers import AutoModel, AutoTokenizer

# Load InLegalBERT (pre-trained on 5.4M Indian legal documents)
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")
```

**Performance Comparison:**

| Metric | BERT-Base | InLegalBERT | Improvement |
|--------|-----------|-------------|-------------|
| NER F1 | ~78% | ~84% | +6% |
| RRL Accuracy | ~72% | ~79% | +7% |
| Convergence | Slower | Faster | ~30% fewer epochs |

## 📊 Datasets

| Dataset | Size | Use Case | Source |
|---------|------|----------|--------|
| **ILDC** | 35K cases | Judgment prediction | [OpenDataLab](https://opendatalab.com/OpenDataLab/ILDC) |
| **InJudgements** | Balanced sample | General training | [HuggingFace](https://huggingface.co/datasets/opennyaiorg/InJudgements_dataset) |
| **Aalap Instruction** | Instruction pairs | LLM fine-tuning | [HuggingFace](https://huggingface.co/datasets/opennyaiorg/aalap_instruction_dataset) |
| **BUILD** | Annotated | Rhetorical roles | [GitHub](https://github.com/Legal-NLP-EkStep/rhetorical-role-baseline) |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM (32GB recommended for Llama 3)

### Installation

```bash
# Clone the repository
git clone https://github.com/viru0909-dev/OpenNyAI.git
cd OpenNyAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from src.models import LegalNERModel, RhetoricalRoleLabeler
from src.data import LegalTextPreprocessor

# Initialize preprocessor
preprocessor = LegalTextPreprocessor()

# Load NER model
ner_model = LegalNERModel(model_name="law-ai/InLegalBERT")
ner_model.load_model()

# Extract entities from judgment text
text = """
In the matter of Kesavananda Bharati v. State of Kerala,
the Hon'ble Supreme Court examined Article 368 of the Constitution.
The Court, comprising a 13-judge bench, delivered its verdict on 24th April 1973.
"""

entities = ner_model.predict(text)
print(entities)
# [{'text': 'Kesavananda Bharati', 'label': 'PETITIONER', ...},
#  {'text': 'State of Kerala', 'label': 'RESPONDENT', ...},
#  {'text': 'Article 368', 'label': 'PROVISION', ...}, ...]
```

## 🔧 Training Custom Models

### Fine-tune InLegalBERT for NER

```bash
python scripts/train.py \
    --model ner \
    --base-model law-ai/InLegalBERT \
    --train-data data/processed/ner_train.json \
    --val-data data/processed/ner_val.json \
    --output-dir models/legal_ner \
    --epochs 10 \
    --batch-size 16
```

### Fine-tune Llama 3 with LoRA

```bash
python scripts/train.py \
    --model llama \
    --base-model meta-llama/Meta-Llama-3-8B \
    --train-data data/corpora/aalap_instructions.jsonl \
    --output-dir models/legal_llama \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32
```

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Encoder** | InLegalBERT | Indian legal text understanding |
| **Generator** | Llama 3 (8B/70B) | Legal reasoning & drafting |
| **Fine-tuning** | LoRA/QLoRA | Parameter-efficient training |
| **Serving** | vLLM / Groq | High-throughput inference |
| **Vector DB** | Milvus / Chroma | RAG retrieval |
| **Backend** | FastAPI | Model serving |
| **Orchestration** | Spring Boot 3.2 | Enterprise integration |

## 📈 Roadmap

- [x] Project structure and base models
- [ ] Indian Kanoon scraping pipeline
- [ ] InLegalBERT NER fine-tuning
- [ ] Rhetorical Role Labeling (BiLSTM-CRF)
- [ ] Llama 3 instruction tuning with Aalap dataset
- [ ] RAG pipeline with Milvus
- [ ] Bhashini integration (multilingual)
- [ ] FastAPI inference server
- [ ] vLLM deployment configuration

## 📚 References

- [OpenNyAI Mission](https://github.com/OpenNyAI/Opennyai)
- [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)
- [Aalap Legal LLM](https://github.com/OpenNyAI/aalap_legal_llm)
- [NyaySetu Project](https://github.com/viru0909-dev/nyay-setu-working)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

<p align="center">
  <b>Building Sovereign Legal Intelligence for Accessible Justice in India</b><br>
  Made with ❤️ by the OpenNyAI Community
</p>
