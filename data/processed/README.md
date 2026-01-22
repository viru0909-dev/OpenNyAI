# OpenNyAI - Processed Data

This directory contains preprocessed data ready for model training.

## File Format

Processed data should be in JSON format with the following structures:

### For NER Training
```json
[
  {
    "text": "The petitioner John Doe filed a case...",
    "entities": [
      {"start": 14, "end": 22, "label": "PETITIONER", "text": "John Doe"}
    ]
  }
]
```

### For Classification Training
```json
[
  {
    "text": "Civil suit regarding property dispute...",
    "label": "Civil"
  }
]
```

### For Summarization Training
```json
[
  {
    "text": "Full judgment text...",
    "summary": "Brief summary of the judgment..."
  }
]
```

## File Naming Convention

- `train.json` - Training data
- `val.json` - Validation data
- `test.json` - Test data
