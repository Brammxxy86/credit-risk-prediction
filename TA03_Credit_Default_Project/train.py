from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3",
    "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"
]
TARGET_CANDIDATES = [
    "default.payment.next.month", "default payment next month",
    "Y", "target"
]

def clean_column_name(name):
    return str(name).strip().replace(" ", "_").replace(".", "_").upper()

def load_dataset():
    print("Mengambil dataset Default of Credit Card Clients dari UCI...")
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    X.columns = [clean_column_name(c) for c in X.columns]
    y.columns = [clean_column_name(c) for c in y.columns]

    # Samakan beberapa kemungkinan variasi nama kolom target dari sumber.
    target_col = None
    for candidate in TARGET_CANDIDATES:
        normalized = clean_column_name(candidate)
        if normalized in y.columns:
            target_col = normalized
            break
    if target_col is None:
        if y.shape[1] == 1:
            target_col = y.columns[0]
        else:
            raise ValueError(f"Kolom target tidak ditemukan. Kolom target tersedia: {list(y.columns)}")

    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Fitur berikut tidak ditemukan pada dataset UCI: {missing}")

    data = X[FEATURES].copy()
    data["TARGET"] = pd.to_numeric(y[target_col], errors="coerce")
    data = data.dropna(subset=FEATURES + ["TARGET"])
    data["TARGET"] = data["TARGET"].astype(int)
    return data

def main():
    data = load_dataset()
    X = data[FEATURES]
    y = data["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "dataset": "UCI Default of Credit Card Clients (ID 350)",
        "total_rows": int(len(data)),
        "features": FEATURES,
        "target": "default.payment.next.month",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
    }

    joblib.dump(pipeline, MODEL_DIR / "logistic_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Tidak Default", "Default"],
        yticklabels=["Tidak Default", "Default"]
    )
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\n=== HASIL EVALUASI MODEL ===")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"{key}: {metrics[key]:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nModel tersimpan di model/logistic_model.joblib")
    print("Metrik tersimpan di model/metrics.json")

if __name__ == "__main__":
    main()
