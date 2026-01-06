import pandas as pd
import xgboost as xgb
import sys
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Column definitions matching embed_text.py
TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

NUTRIENT_COLS = [
    "calories", "total_fat", "sat_fat", "trans_fat", "unsat_fat",
    "cholesterol", "sodium", "carbs", "dietary_fiber",
    "total_sugars", "added_sugars", "protein", "potassium",
]

# Columns to keep: only text + nutrients + gtin (no tags/flags)
COLS_TO_KEEP = TEXT_COLS + NUTRIENT_COLS + ["gtin"]


def process_and_embed(df):
    # Cleans and embeds raw CSV data to match training feature set
    # Only uses: text columns (embedded) + nutrient columns (no tag/flag columns)
    
    # Keep only columns used in training
    keep = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[keep].copy()
    print(f"Selected {len(keep)} columns: text + nutrients + gtin")
    
    # Convert ID and text columns to string
    for col in ["gtin"] + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Convert nutrient columns to numeric (remove carets, fill missing with -1)
    for col in NUTRIENT_COLS:
        if col in df.columns:
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            df[col] = df[col].fillna(-1)
    
    
    # Remove rows without GTIN or empty ingredients
    df = df.dropna(subset=["gtin"])
    if "ingredients_text" in df.columns:
        df = df[df["ingredients_text"].notna() & (df["ingredients_text"].astype(str).str.strip() != "")]
    
    # Embed text columns using SentenceTransformer
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for col in TEXT_COLS:
        if col in df.columns:
            print(f"Embedding {col}...")
            text_data = df[col].fillna("").astype(str)
            embeddings = model.encode(text_data.tolist())
            embedding_cols = [f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
            embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
            df = pd.concat([df, embedding_df], axis=1)
    
    # Drop original text columns (keep only embeddings)
    df = df.drop(columns=TEXT_COLS, errors="ignore")
    
    # Verify feature set matches training
    final_cols = df.columns.tolist()
    embedding_count = len([c for c in final_cols if '_emb_' in c])
    nutrient_count = len([c for c in final_cols if c in NUTRIENT_COLS])
    print(f"Final features: {embedding_count} embeddings + {nutrient_count} nutrients = {len(final_cols)} total")
    
    # Separate GTIN for reference, return feature matrix
    gtin_column = df["gtin"].copy() if "gtin" in df.columns else None
    X = df.drop(columns=["gtin"], errors="ignore")
    
    return X, gtin_column


def test_model_on_dataset(csv_path, expected_result, dataset_name, threshold=0.3):
    # Tests model on a dataset using custom probability threshold
    
    print(f"\nTesting on: {dataset_name}")
    print(f"Expected: {expected_result} | Using Threshold: {threshold}")
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return 0, None, None
    
    # Load and process data
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")
    
    X, gtin_column = process_and_embed(df)
    print(f"After processing: {len(X):,} rows, {len(X.columns)} features")
    
    # Load trained model
    models_dir = Path(__file__).parent.parent / "models"
    model_path = models_dir / "final_anomaly_detector.json"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 0, None, None
    
    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Generate probabilities
    print("Generating predictions...")
    probs = model.predict_proba(X)[:, 1]
    
    # Apply custom threshold (instead of default 0.5)
    preds = (probs >= threshold).astype(int)
    
    # Calculate metrics
    total_rows = len(preds)
    anomalies_detected = sum(preds)
    anomaly_percentage = (anomalies_detected / total_rows * 100) if total_rows > 0 else 0
    avg_probability = probs.mean()
    
    print(f"\nResults (at {threshold} threshold):")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Anomalies detected: {anomalies_detected:,}")
    print(f"  Catch Rate / FP Rate: {anomaly_percentage:.2f}%")
    print(f"  Average probability: {avg_probability:.3f}")
    
    # Confidence breakdown
    high = sum(probs >= 0.7)
    med = sum((probs >= 0.3) & (probs < 0.7))
    low = sum(probs < 0.3)
    
    print(f"\nConfidence Breakdown:")
    print(f"  High (>=0.7): {high} | Medium (0.3-0.7): {med} | Low (<0.3): {low}")
    
    return anomaly_percentage, probs, preds


def main():
    # Main function to test model on both validation datasets
    data_folder = Path(__file__).parent.parent / "TestData"
    
    # Target threshold balances high recall with low false positives
    TARGET_THRESHOLD = 0.2 
    
    # Test on errors dataset (measures recall/catch rate)
    errors_file = data_folder / "AI Training Data - items data errors.csv"
    catch_rate, _, _ = test_model_on_dataset(errors_file, "100% anomalies", "Items Data Errors", threshold=TARGET_THRESHOLD)
    
    # Test on approved dataset (measures false positive rate)
    approved_file = data_folder / "AI Training Data - approved profiles.csv"
    fp_rate, _, _ = test_model_on_dataset(approved_file, "0% anomalies", "Approved Profiles", threshold=TARGET_THRESHOLD)
    
    print(f"\nFINAL PERFORMANCE SUMMARY")
    print(f"Overall Catch Rate (Recall): {catch_rate:.2f}%")
    print(f"Overall False Positive Rate: {fp_rate:.2f}%")
    
    if fp_rate <= 10.0:
        print("SUCCESS: Threshold met the <10% False Positive requirement!")
    else:
        print(f"ALERT: False Positive rate exceeds 10%. Consider raising threshold above {TARGET_THRESHOLD}.")


if __name__ == "__main__":
    main()
