#!/usr/bin/env python
"""
TF-IDF Baseline for Tune Type Classification
Compare learned embeddings against simple text-based features.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

def main():
    print("=" * 80)
    print("TF-IDF Baseline for Tune Type Classification")
    print("=" * 80)
    print()

    # Load test data
    data_dir = Path("/Volumes/LLModels/Projects/ABC2VEC/data/processed")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    # Filter out empty tune types
    test_df = test_df[test_df['tune_type'].notna() & (test_df['tune_type'] != '')]

    print(f"Loaded {len(test_df)} test tunes")
    print(f"Tune types: {test_df['tune_type'].value_counts().to_dict()}")
    print()

    # Prepare data
    X_text = test_df['abc_body'].values
    y = test_df['tune_type'].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Creating TF-IDF features...")
    print("  - Character n-grams: 2-4")
    print("  - Max features: 5000")
    print()

    # TF-IDF vectorizer with character n-grams
    # This treats ABC notation as text and extracts character patterns
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 4),
        max_features=5000,
        lowercase=False  # ABC notation is case-sensitive
    )

    X_tfidf = vectorizer.fit_transform(X_text)
    print(f"TF-IDF feature matrix shape: {X_tfidf.shape}")
    print()

    # 5-fold cross-validation
    print("Running 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_macro_f1 = []
    fold_weighted_f1 = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_tfidf, y_encoded)):
        X_train, X_val = X_tfidf[train_idx], X_tfidf[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        # Train logistic regression
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        )
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_val)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        weighted_f1 = f1_score(y_val, y_pred, average='weighted')

        fold_accuracies.append(acc)
        fold_macro_f1.append(macro_f1)
        fold_weighted_f1.append(weighted_f1)

        print(f"  Fold {fold_idx + 1}: Acc={acc:.4f}, Macro F1={macro_f1:.4f}, Weighted F1={weighted_f1:.4f}")

    print()
    print("=" * 80)
    print("BASELINE RESULTS (TF-IDF + Logistic Regression)")
    print("=" * 80)
    print(f"Accuracy:     {np.mean(fold_accuracies):.1f}% ± {np.std(fold_accuracies):.1f}%")
    print(f"Macro F1:     {np.mean(fold_macro_f1):.1f}% ± {np.std(fold_macro_f1):.1f}%")
    print(f"Weighted F1:  {np.mean(fold_weighted_f1):.1f}% ± {np.std(fold_weighted_f1):.1f}%")
    print()

    # Train on all data for final report
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_tfidf, y_encoded)
    y_pred_all = clf.predict(X_tfidf)

    print("Detailed Classification Report (Full Test Set):")
    print("-" * 80)
    print(classification_report(y_encoded, y_pred_all, target_names=le.classes_))
    print()

    # Compare to ABC2Vec
    print("=" * 80)
    print("COMPARISON TO ABC2VEC")
    print("=" * 80)
    abc2vec_results = {
        "tune_type_classification_results.json": {
            "accuracy": 78.4,
            "std": 1.2,
            "macro_f1": 62.5,
            "weighted_f1": 78.0
        }
    }

    tfidf_acc = np.mean(fold_accuracies) * 100
    abc2vec_acc = abc2vec_results["tune_type_classification_results.json"]["accuracy"]

    improvement = abc2vec_acc - tfidf_acc

    print(f"TF-IDF Baseline:    {tfidf_acc:.1f}%")
    print(f"ABC2Vec (Learned):  {abc2vec_acc:.1f}%")
    print(f"Improvement:        +{improvement:.1f} percentage points")
    print()

    if improvement > 5:
        print("✓ ABC2Vec shows substantial improvement over text-based baseline")
    elif improvement > 2:
        print("✓ ABC2Vec shows moderate improvement over text-based baseline")
    else:
        print("⚠ ABC2Vec shows marginal improvement over text-based baseline")

    print()
    print("=" * 80)

    # Save results
    results = {
        "baseline_method": "TF-IDF + Logistic Regression",
        "feature_type": "Character n-grams (2-4)",
        "max_features": 5000,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_macro_f1": float(np.mean(fold_macro_f1)),
        "std_macro_f1": float(np.std(fold_macro_f1)),
        "mean_weighted_f1": float(np.mean(fold_weighted_f1)),
        "std_weighted_f1": float(np.std(fold_weighted_f1)),
        "fold_accuracies": [float(x) for x in fold_accuracies],
        "abc2vec_accuracy": abc2vec_acc,
        "improvement_over_baseline": float(improvement)
    }

    output_path = Path("/Volumes/LLModels/Projects/ABC2VEC/checkpoints/tfidf_baseline_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
