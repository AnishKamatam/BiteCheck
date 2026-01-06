# BiteCheck: Anomaly Detection System for Nutrition Data Quality Assurance

A machine learning framework for automated detection of data quality anomalies in nutrition product datasets using ensemble gradient boosting and semantic text embeddings.

## Abstract

This repository implements an anomaly detection system designed to identify data entry errors and inconsistencies in structured nutrition product data. The system employs an ensemble of XGBoost classifiers trained on synthetically augmented datasets using SMOTE and ADASYN oversampling techniques. Text features are encoded using SentenceTransformer embeddings, enabling the model to capture semantic relationships in product descriptions, categories, and ingredient lists. The framework demonstrates effective performance on imbalanced classification tasks with approximately 6% positive class representation.

## System Architecture

```
BiteCheck/
├── Data/
│   ├── PreOpData/          # Raw Excel files (Before/After pairs)
│   ├── PreOpDataCSV/       # Processed CSV files
│   └── PostOpData/         # Final embedded datasets
├── DataOps/
│   ├── addcaret.py         # Extract Excel data and add caret markers
│   ├── cleandata.py        # Data cleaning and preprocessing
│   ├── mergedata.py        # Merge Before/After datasets
│   └── embedtext.py        # Embed text columns using SentenceTransformer
├── ModelTraining/
│   ├── trainmodel.py       # Train ensemble models (SMOTE + ADASYN)
│   └── optimizemodel.py    # Hyperparameter optimization with Optuna
├── ModelTests/
│   └── testmodel.py       # Test ensemble model on validation datasets
└── models/
    ├── best_hyperparams.json           # Optimized hyperparameters
    ├── bitecheckSMOTE_V1.json         # SMOTE-trained model
    └── bitecheckADASYN_V1.json        # ADASYN-trained model
```

## Methodology

### Data Preprocessing Pipeline

#### 1. Data Extraction (`addcaret.py`)

The extraction module processes structured Excel files containing nutrition product data. The system identifies anomaly markers through visual indicators (yellow or cyan background colors) in the source data, converting these markers to caret (^) annotations for downstream processing.

#### 2. Data Cleaning (`cleandata.py`)

The cleaning module performs feature selection, type conversion, and data quality filtering. Anomaly labels are derived from caret markers present in the original dataset. Nutrient values are normalized to numeric format with missing values imputed using a sentinel value (-1). The pipeline removes records lacking essential identifiers (GTIN) or critical text fields (ingredients).

#### 3. Data Merging (`mergedata.py`)

The merging module combines multiple data sources by aligning temporal snapshots (Before/After pairs) using product identifiers. Features are extracted from the "Before" state while labels are derived from the "After" state, creating a supervised learning dataset that captures the correction process.

#### 4. Text Embedding (`embedtext.py`)

Textual features are transformed into dense vector representations using the SentenceTransformer framework. The `all-MiniLM-L6-v2` model generates 384-dimensional embeddings for each text field (category name, product name, ingredients), resulting in 1,152 total embedding features that capture semantic relationships in product descriptions.

### Model Training Framework

#### Hyperparameter Optimization (`optimizemodel.py`)

Hyperparameter search is conducted using Optuna's tree-structured Parzen estimator (TPE) algorithm. The optimization targets Average Precision Score (PR-AUC), which is appropriate for imbalanced classification tasks. The search space includes tree structure parameters (max_depth, min_child_weight), regularization parameters (gamma, subsample, colsample_bytree), and learning rate. Optimal hyperparameters are persisted for model training.

#### Ensemble Model Training (`trainmodel.py`)

The training framework implements an ensemble approach using two complementary oversampling strategies:

1. **SMOTE Model**: Synthetic Minority Oversampling Technique generates synthetic samples through interpolation between existing minority class instances, creating uniform coverage across the feature space.

2. **ADASYN Model**: Adaptive Synthetic Sampling generates synthetic samples with higher density near class boundaries, focusing on harder-to-classify instances.

Both models utilize identical hyperparameter configurations and XGBoost architecture. Oversampling raises the minority class representation from approximately 6% to 20% of the training set. Models are trained with `scale_pos_weight=1.0` since oversampling addresses class imbalance.

**Dataset Characteristics:**
- Feature dimensionality: 1,165 (1,152 text embeddings + 13 nutrient features)
- Training set size: 17,723 samples
- Initial class distribution: 6.09% positive class (1,079 anomalies)
- Post-SMOTE distribution: 15,978 samples, 16.67% positive class
- Post-ADASYN distribution: 15,995 samples, 16.76% positive class

#### Ensemble Strategy

The system employs a maximum probability ensemble strategy for inference. Predictions from both models are combined by taking the maximum probability score, effectively flagging instances when either model indicates high anomaly probability. This approach prioritizes recall while maintaining precision through complementary model perspectives.

### Model Evaluation

#### Test Dataset Performance

The system is evaluated on two validation datasets representing different data quality scenarios:

**Errors Dataset (Expected: 100% anomalies):**
- Detection rate (Recall): 85.49% (271 of 317 instances)
- Mean predicted probability: 0.679
- High confidence predictions (≥0.7): 197 instances
- Medium confidence predictions (0.3-0.7): 69 instances
- Low confidence predictions (<0.3): 51 instances
- Rule-based flags: 10 instances flagged due to sodium > 1500mg threshold

**Approved Dataset (Expected: 0% anomalies):**
- False positive rate: 5.68% (18 of 317 instances)
- Mean predicted probability: 0.070
- High confidence predictions (≥0.7): 1 instance
- Medium confidence predictions (0.3-0.7): 1 instance
- Low confidence predictions (<0.3): 315 instances
- Rule-based flags: 9 instances flagged due to sodium > 1500mg threshold

The system achieves a false positive rate below the 10% operational requirement while maintaining high recall on known error cases.

## Feature Engineering

### Text Embeddings (1,152 features)

Semantic representations of textual product information:
- Category name: 384-dimensional embedding
- Product name: 384-dimensional embedding
- Ingredients text: 384-dimensional embedding

### Numerical Features (13 features)

Nutritional composition metrics:
calories, total_fat, saturated_fat, trans_fat, unsaturated_fat, cholesterol, sodium, carbohydrates, dietary_fiber, total_sugars, added_sugars, protein, potassium

**Total Feature Space**: 1,165 dimensions

## Domain-Specific Rules

The system incorporates rule-based heuristics to complement machine learning predictions:

- **Sodium Threshold Rule**: Products exceeding 1500mg sodium per serving are automatically flagged as anomalies, regardless of model prediction. This deterministic rule ensures regulatory compliance and captures known high-risk patterns.

## Usage

### Pipeline Execution

1. **Data Extraction:**
   ```bash
   python DataOps/addcaret.py
   ```

2. **Data Cleaning and Merging:**
   ```bash
   python DataOps/mergedata.py
   ```

3. **Feature Embedding:**
   ```bash
   python DataOps/embedtext.py
   ```

4. **Model Training:**
   ```bash
   python ModelTraining/trainmodel.py
   ```

5. **Model Evaluation:**
   ```bash
   python ModelTests/testmodel.py
   ```

### Inference API

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# Load ensemble models
models_dir = Path("models")
model_smote = xgb.XGBClassifier()
model_smote.load_model(str(models_dir / "bitecheckSMOTE_V1.json"))

model_adasyn = xgb.XGBClassifier()
model_adasyn.load_model(str(models_dir / "bitecheckADASYN_V1.json"))

# Generate predictions
probs_smote = model_smote.predict_proba(X)[:, 1]
probs_adasyn = model_adasyn.predict_proba(X)[:, 1]
probs_ensemble = np.maximum(probs_smote, probs_adasyn)

# Apply decision threshold
preds = (probs_ensemble >= 0.2).astype(int)

# Apply domain-specific rules
# ...
```

## Design Rationale

1. **Supervised Learning from Corrections**: The system learns from human-annotated corrections, using the "Before" state as features and the "After" state to derive labels, capturing the data quality improvement process.

2. **Semantic Text Encoding**: SentenceTransformer embeddings enable the model to understand semantic relationships in product descriptions, improving generalization beyond exact text matching.

3. **Ensemble Methodology**: Combining SMOTE and ADASYN models leverages complementary oversampling strategies, with maximum probability aggregation prioritizing recall for anomaly detection.

4. **Threshold Calibration**: A probability threshold of 0.2 (rather than the default 0.5) balances recall and precision for the operational context.

5. **Hybrid Approach**: Machine learning predictions are supplemented with rule-based heuristics to ensure comprehensive coverage of known anomaly patterns.

## Dependencies

- pandas
- xgboost
- scikit-learn
- sentence-transformers
- imbalanced-learn (SMOTE/ADASYN implementations)
- optuna (hyperparameter optimization)
- pyarrow (Parquet file I/O)
- openpyxl (Excel file processing)
- numpy

## Performance Characteristics

The ensemble system demonstrates effective performance on imbalanced classification tasks, achieving high recall on known error cases while maintaining low false positive rates suitable for production deployment. The maximum probability ensemble strategy provides robustness through model diversity, with complementary oversampling techniques capturing different aspects of the anomaly detection problem space.

Threshold selection represents a fundamental trade-off between recall and precision. Lower thresholds increase anomaly detection coverage but may elevate false positive rates. The implemented threshold (0.2) balances these objectives for the target application domain.

Rule-based components complement machine learning predictions by encoding domain expertise and regulatory requirements, ensuring deterministic handling of known high-risk patterns.
