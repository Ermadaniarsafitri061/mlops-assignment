from pathlib import Path
import mlflow
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri("http://127.0.0.1:5000")
exp = mlflow.get_experiment_by_name("bc-malignant-detection")
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
    filter_string="tags.candidate = 'true' and metrics.cv_precision_mean >= 0.90",
    order_by=["metrics.cv_recall_mean DESC", "metrics.cv_f1_mean DESC"])
if runs.empty: raise SystemExit("Tidak ada kandidat layak. Jalankan train.py dan periksa MLflow UI.")
best = runs.iloc[0]
model_uri = best.get("tags.model_uri")
if not isinstance(model_uri, str) or not model_uri: raise SystemExit("Model URI tidak ditemukan pada run terpilih.")
test_df = pd.read_csv(ROOT / "data" / "test.csv")
X_test, y_test = test_df.drop(columns="target"), test_df["target"]
model = mlflow.sklearn.load_model(model_uri)
y_pred = model.predict(X_test)
score = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
metrics = {"test_recall": recall_score(y_test, y_pred), "test_precision": precision_score(y_test, y_pred),
           "test_f1": f1_score(y_test, y_pred), "test_roc_auc": roc_auc_score(y_test, score)}
with mlflow.start_run(run_id=best["run_id"]):
    mlflow.log_metrics(metrics)
    mlflow.log_input(mlflow.data.from_pandas(test_df, source="data/test.csv", name="bc-test", targets="target"), context="testing")
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["benign", "malignant"])
    mlflow.log_figure(disp.figure_, "plots/confusion_matrix_test.png")
    plt.close(disp.figure_)
    mlflow.set_tag("test_evaluated", "true")
print("Best run:", best.get("tags.mlflow.runName"), best["run_id"])
for k,v in metrics.items(): print(f"{k}: {v:.4f}")
print("Model URI:", model_uri)
