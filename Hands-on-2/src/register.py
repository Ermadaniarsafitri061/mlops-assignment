import mlflow
from mlflow import MlflowClient
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
NAME = "bc-malignant-classifier"
exp = mlflow.get_experiment_by_name("bc-malignant-detection")
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
    filter_string="tags.candidate = 'true' and metrics.cv_precision_mean >= 0.90 and tags.test_evaluated = 'true'",
    order_by=["metrics.cv_recall_mean DESC", "metrics.cv_f1_mean DESC"])
if runs.empty: raise SystemExit("Tidak ditemukan kandidat yang sudah dievaluasi. Jalankan evaluate.py terlebih dahulu.")
best = runs.iloc[0]
model_uri = best.get("tags.model_uri")
if not isinstance(model_uri, str) or not model_uri: raise SystemExit("Model URI tidak ditemukan.")
test_precision = float(best.get("metrics.test_precision", 0.0))
test_recall = float(best.get("metrics.test_recall", 0.0))
try:
    champ = client.get_model_version_by_alias(NAME, "champion")
    champ_run = client.get_run(champ.run_id)
    champion_recall = float(champ_run.data.metrics.get("test_recall", -1.0))
except Exception:
    champion_recall = -1.0
mv = mlflow.register_model(model_uri, NAME)
client.set_model_version_tag(NAME, mv.version, "validation_status", "approved" if (test_precision >= 0.90 and test_recall >= champion_recall + 0.005) else "rejected")
client.set_model_version_tag(NAME, mv.version, "data_md5", str(best.get("tags.data_md5", "unknown")))
client.update_model_version(NAME, mv.version, description=f"Candidate={best.get('tags.mlflow.runName')}; test_recall={test_recall:.4f}; test_precision={test_precision:.4f}")
client.set_registered_model_alias(NAME, "challenger", mv.version)
passed = test_precision >= 0.90 and test_recall >= champion_recall + 0.005
if passed:
    client.set_registered_model_alias(NAME, "champion", mv.version)
print("Registry:", NAME, "version:", mv.version)
print("test_recall:", test_recall, "test_precision:", test_precision)
print("Champion recall sebelumnya:", champion_recall)
print("Quality gate:", "PASS" if passed else "NOT PROMOTED (registered as challenger)")
