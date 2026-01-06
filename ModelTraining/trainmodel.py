import pandas as pd
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from pathlib import Path


def train_final_model():

    # Trains final model with optimized hyperparameters and analyzes threshold sensitivity
    # Uses only: text embeddings + nutrient columns (no tag/flag columns)
    data_path = Path(__file__).parent.parent / "Data" / "PostOpData" / "merged_embedded.parquet"
    df = pd.read_parquet(data_path)
    
    X = df.drop(columns=["gtin", "label_is_anomaly"], errors="ignore")
    y = df["label_is_anomaly"]
    
    # Verify feature set
    print(f"Training on {len(X.columns)} features:")
    embedding_cols = [c for c in X.columns if '_emb_' in c]
    nutrient_cols = [c for c in X.columns if c in ['calories', 'total_fat', 'sat_fat', 'trans_fat', 'unsat_fat', 
                                                     'cholesterol', 'sodium', 'carbs', 'dietary_fiber',
                                                     'total_sugars', 'added_sugars', 'protein', 'potassium']]
    print(f"  - Text embeddings: {len(embedding_cols)} features")
    print(f"  - Nutrient columns: {len(nutrient_cols)} features")
    print(f"  - Total features: {len(X.columns)}")
    print(f"  - Training samples: {len(X):,}")
    print(f"  - Anomalies: {y.sum():,} ({y.mean()*100:.2f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- SMOTE (TRAINING ONLY) ---
    print("\nApplying SMOTE to training data...")
    smote = SMOTE(
        sampling_strategy=0.2,  # raise anomalies to ~20% of training data
        random_state=42,
        k_neighbors=5
    )
    
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE:")
    print(f"  - Training samples: {len(X_train_smote):,}")
    print(f"  - Anomalies: {y_train_smote.sum():,} ({y_train_smote.mean()*100:.2f}%)")

    # Load optimized hyperparameters from Optuna
    models_dir = Path(__file__).parent.parent / "models"
    with open(models_dir / "best_hyperparams.json", "r") as f:
        best_params = json.load(f)
    

    # Reduced class weight (SMOTE handles imbalance, so we use 1.0)
    weight = 1.0
    model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=weight,
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42
    )

    print("\nTraining final model with SMOTE...")
    model.fit(X_train_smote, y_train_smote)


    # Get probability predictions (not just binary)
    probs = model.predict_proba(X_test)[:, 1]


    # Analyze different thresholds to find optimal balance
    print("\nSensitivity Analysis (How many errors can we catch?):")
    thresholds = [0.7, 0.5, 0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2]
    for t in thresholds:
        t_preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, t_preds).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_test, t_preds)
        print(f"Threshold {t:.2f}: Recall = {recall:.1%}, Precision = {precision:.1%}, F1 = {f1:.3f}, Total Flagged = {tp+fp}")


    # Save the final model
    model_path = models_dir / "final_anomaly_detector.json"
    model.get_booster().save_model(str(model_path))
    print(f"\nFinal model saved as {model_path}")


if __name__ == "__main__":
    train_final_model()
