import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
runs = mlflow.search_runs(experiment_names=["bc-malignant-detection"],
    filter_string="tags.candidate = 'true' and metrics.cv_precision_mean >= 0.90",
    order_by=["metrics.cv_recall_mean DESC", "metrics.cv_f1_mean DESC"])
cols = [c for c in ["tags.mlflow.runName", "tags.model_family", "metrics.cv_recall_mean",
                    "metrics.cv_recall_std", "metrics.cv_precision_mean", "metrics.cv_f1_mean", "run_id",
                    "tags.model_uri"] if c in runs.columns]
print(runs[cols].head(10).to_string(index=False))
if runs.empty:
    raise SystemExit("Tidak ada kandidat yang memenuhi precision >= 0.90. Periksa run MLflow.")
best = runs.iloc[0]
print("\nKandidat terpilih:", best.get("tags.mlflow.runName"), best["run_id"])
print("Model URI:", best.get("tags.model_uri"))
runs.to_csv("candidate_comparison.csv", index=False)
print("Tabel lengkap disimpan ke candidate_comparison.csv")
