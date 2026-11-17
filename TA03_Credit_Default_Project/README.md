# TA-03 — Prediksi Risiko Gagal Bayar Kartu Kredit

**Nama:** Bayu Muhammad Ramadhan  
**NIM:** 301240008  
**Kelas:** 5A

Project ini menggunakan dataset resmi **Default of Credit Card Clients** dari UCI Machine Learning Repository (dataset ID 350), dengan Logistic Regression sebagai model klasifikasi biner.

## 1. Dataset
Dataset diambil otomatis dari UCI ketika `train.py` dijalankan, sehingga file Excel tidak wajib disimpan manual di folder project. Dataset berisi 30.000 baris dan 23 fitur menurut halaman resmi UCI.

Sumber: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients  
DOI: https://doi.org/10.24432/C55S3H

## 2. Fitur model
Model memakai 10 fitur:
`LIMIT_BAL`, `AGE`, `EDUCATION`, `MARRIAGE`, `PAY_0`, `PAY_2`, `PAY_3`, `BILL_AMT1`, `PAY_AMT1`, `PAY_AMT2`.

Target: `default.payment.next.month` (0 = tidak default, 1 = default).

## 3. Menjalankan di VS Code
Pastikan Python 3.10+ terpasang. Buka terminal di folder project:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install library:
```bash
pip install -r requirements.txt
```

Training dan evaluasi:
```bash
python train.py
```

Script akan mengambil dataset UCI, membagi data secara stratified (80% train, 20% test), melakukan scaling, melatih Logistic Regression, lalu menyimpan:
- `model/logistic_model.joblib`
- `model/metrics.json`
- `model/confusion_matrix.png`

Jalankan web:
```bash
python app.py
```
Buka http://127.0.0.1:5000

## 4. Deployment Railway
1. Push seluruh folder project ke repository GitHub.
2. Buat project baru di Railway dan hubungkan repository.
3. Railway akan memasang dependensi dari `requirements.txt`.
4. Build command (jika diperlukan): `python train.py`.
5. Start command: `python train.py && gunicorn app:app` (sudah dicantumkan pada Procfile).

**Penting:** Karena model dibuat oleh `train.py`, pastikan training dijalankan saat build/deploy. Jika Railway tidak menjalankan build command, gunakan start command `python train.py && gunicorn app:app` agar file model tersedia sebelum Flask dijalankan. Training membutuhkan koneksi internet untuk mengambil dataset UCI.

## 5. Catatan
- Hasil metrik aktual tercatat pada `model/metrics.json` setelah training berhasil.
- Aplikasi adalah prototipe akademik, bukan alat keputusan kredit.
- Kode mengasumsikan input numerik untuk seluruh fitur yang dipilih.
- Dataset berlisensi CC BY 4.0; cantumkan atribusi UCI dan Yeh (2009).
