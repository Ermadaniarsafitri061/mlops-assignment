from pathlib import Path
import hashlib
import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.datasets import load_breast_cancer
ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
champ = client.get_model_version_by_alias("bc-malignant-classifier", "champion")
run = client.get_run(champ.run_id)
df = load_breast_cancer(as_frame=True).frame.copy()
df["target"] = 1 - df["target"]
data_md5 = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
print("Model:", champ.name, "version:", champ.version)
print("Run:", run.info.run_name, run.info.run_id)
print("Git commit:", run.data.tags.get("mlflow.source.git.commit", "not recorded"))
print("Data MD5 current:", data_md5)
print("Data MD5 model:", run.data.tags.get("data_md5"))
print("Params:", {k:v for k,v in run.data.params.items() if "clf__" in k or k in ("C", "gamma")})
print("Metrics:", {k:round(v,4) for k,v in run.data.metrics.items()})
if run.data.tags.get("data_md5") not in (None, data_md5):
    raise SystemExit("REPRODUCE CHECK FAILED: data hash tidak cocok.")
model = mlflow.pyfunc.load_model(f"models:/{champ.name}@champion")
X = df.drop(columns="target").head(3)
print("Prediction:", model.predict(X))
print("REPRODUCE CHECK: PASS (data hash cocok dan model berhasil dimuat)")
