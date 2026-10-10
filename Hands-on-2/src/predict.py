import requests
import pandas as pd
from pathlib import Path
import mlflow
ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri("http://127.0.0.1:5000")
test_df = pd.read_csv(ROOT / "data" / "test.csv")
X_test = test_df.drop(columns="target")
payload = {"dataframe_split": X_test.head(3).to_dict(orient="split")}
r = requests.post("http://127.0.0.1:5001/invocations", json=payload, timeout=60)
r.raise_for_status()
print("REST API predictions:", r.json())
model = mlflow.pyfunc.load_model("models:/bc-malignant-classifier@champion")
print("PyFunc predictions:", model.predict(X_test.head(3)))
