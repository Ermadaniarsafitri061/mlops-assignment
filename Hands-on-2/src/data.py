from pathlib import Path
import hashlib
import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bc-malignant-detection")

df = load_breast_cancer(as_frame=True).frame.copy()
# scikit-learn target: 0=malignant, 1=benign. Flip it so 1=malignant (positive class).
df["target"] = 1 - df["target"]
data_md5 = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
train_df, test_df = train_test_split(
    df, test_size=0.20, stratify=df["target"], random_state=42
)
train_df.to_csv(DATA_DIR / "train.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)
with mlflow.start_run(run_name="L1-data-preparation"):
    mlflow.log_params({"test_size": 0.2, "random_state": 42, "stratify": True})
    mlflow.set_tags({"data_md5": data_md5, "dataset": "sklearn-breast-cancer", "stage": "L1"})
    mlflow.log_metrics({"n_samples": len(df), "n_features": df.shape[1] - 1,
                        "n_train": len(train_df), "n_test": len(test_df)})
    mlflow.log_artifact(str(DATA_DIR / "train.csv"), artifact_path="data")
    mlflow.log_artifact(str(DATA_DIR / "test.csv"), artifact_path="data")
print("Jumlah sampel:", len(df))
print("Jumlah fitur:", df.shape[1] - 1)
print("Train/Test:", len(train_df), "/", len(test_df))
print("Data MD5:", data_md5)
print("Distribusi target (0=benign, 1=malignant):")
print(df["target"].value_counts().sort_index())
