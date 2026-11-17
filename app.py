from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "logistic_model.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"

FEATURES = [
    "LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3",
    "BILL_AMT1", "PAY_AMT1", "PAY_AMT2"
]
LABELS = {
    "LIMIT_BAL": "Batas kredit (NT Dollar)",
    "AGE": "Usia",
    "EDUCATION": "Pendidikan (1-4; 0/5/6 jika sesuai data)",
    "MARRIAGE": "Status pernikahan (1-3; 0 jika sesuai data)",
    "PAY_0": "Status pembayaran bulan terakhir",
    "PAY_2": "Status pembayaran 2 bulan sebelumnya",
    "PAY_3": "Status pembayaran 3 bulan sebelumnya",
    "BILL_AMT1": "Tagihan bulan terakhir (NT Dollar)",
    "PAY_AMT1": "Pembayaran bulan terakhir (NT Dollar)",
    "PAY_AMT2": "Pembayaran 2 bulan sebelumnya (NT Dollar)"
}

app = Flask(__name__)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None
    values = {}
    if request.method == "POST":
        try:
            row = {}
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()
                values[feature] = raw
                if raw == "":
                    raise ValueError(f"Kolom {LABELS[feature]} wajib diisi.")
                row[feature] = float(raw)
            if model is None:
                raise RuntimeError("Model belum tersedia. Jalankan python train.py terlebih dahulu.")
            frame = pd.DataFrame([row], columns=FEATURES)
            prediction = int(model.predict(frame)[0])
            probability = float(model.predict_proba(frame)[0][1]) * 100
            result = "Berisiko Default" if prediction == 1 else "Tidak Terindikasi Default"
        except Exception as exc:
            error = str(exc)
    return render_template(
        "index.html", features=FEATURES, labels=LABELS,
        result=result, probability=probability, error=error,
        values=values, metrics=metrics
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
