# Hands-on 1 — Logging Eksperimen Menggunakan MLflow

## PowerShell (Windows)
1. Buka folder `Hands-on-1` di VS Code.
2. Buka terminal PowerShell di folder ini.
3. Buat environment: `python -m venv .venv`
4. Aktifkan: `.\.venv\Scripts\Activate.ps1`
5. Instal library: `pip install -r requirements.txt`
6. Terminal pertama: `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000`
7. Biarkan server berjalan. Buka terminal PowerShell kedua di folder yang sama, aktifkan environment, lalu jalankan `python src/train.py`.
8. Buka `http://127.0.0.1:5000` di browser. Pilih experiment `breast-cancer-clf`, lalu run `rf-baseline`.

Hasil dapat sedikit berbeda menurut versi package. Catat angka yang benar-benar keluar dari terminal/UI, jangan menyalin angka contoh sebagai hasil sendiri.
