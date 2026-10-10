# Hands-on 2 — End-to-end MLflow + scikit-learn

## Catatan penting
- Jalankan dari PowerShell di folder `Hands-on-2`.
- Dataset berasal dari `sklearn.datasets.load_breast_cancer`, jadi tidak perlu download CSV dari luar.
- Target disesuaikan agar `1 = malignant` (kelas positif) sesuai slide.
- `test.csv` harus dianggap terkunci sampai `evaluate.py` (L6). Jangan menggunakannya untuk memilih/tuning model.
- Urutan kode di bawah mengikuti tahapan L1–L9 pada slide. Beberapa output dapat berbeda antarversi library; gunakan hasil aktual untuk laporan.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Jika PowerShell menolak aktivasi, jalankan terminal Git Bash yang sudah biasa digunakan, atau izinkan skrip untuk sesi ini dengan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` lalu aktivasi ulang.

## Urutan pengerjaan
1. Terminal server (folder ini):
```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```
2. Terminal kedua, aktifkan `.venv`, kemudian jalankan berturut-turut:
```powershell
python src/data.py
python src/train.py
python src/analysis.py
python src/evaluate.py
python src/register.py
```
3. Buka `http://127.0.0.1:5000`, pilih experiment `bc-malignant-detection`, bandingkan run dan catat screenshot/tabel hasil.
4. Jika `register.py` meloloskan model dan alias `champion` tersedia, jalankan server serving pada terminal ketiga:
```powershell
mlflow models serve -m "models:/bc-malignant-classifier@champion" -p 5001 --env-manager local
```
5. Terminal keempat:
```powershell
python src/predict.py
```
6. Jalankan reproduction:
```powershell
python src/reproduce.py
```

## Struktur
- `src/data.py`: L1 data split, target label, hash, menyimpan train/test.
- `src/tracking.py`: helper logging dan 5-fold cross-validation.
- `src/train.py`: L2 baseline, L3 model selection, L4 tuning SVC.
- `src/analysis.py`: L5 kandidat dengan precision CV minimal 0.90.
- `src/evaluate.py`: L6 evaluasi test set.
- `src/register.py`: L7 registry, quality gate, alias challenger/champion.
- `src/predict.py`: L8 REST API dan PyFunc.
- `src/reproduce.py`: L9 data hash dan lineage.

Jika proses registry tidak mempromosikan model, periksa metrik aktual. Jangan memalsukan hasil; dokumentasikan quality gate yang benar-benar terjadi. Jika error kompatibilitas MLflow muncul, kirim screenshot/traceback agar disesuaikan dengan versi terpasang.
