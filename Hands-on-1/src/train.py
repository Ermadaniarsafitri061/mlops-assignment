import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("breast-cancer-clf")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

params = {"n_estimators": 200, "max_depth": 8, "random_state": 42}

with mlflow.start_run(run_name="rf-baseline"):
    mlflow.set_tags({"stage": "baseline", "author": "Mahasiswa"})
    mlflow.log_params(params)
    model = RandomForestClassifier(**params).fit(X_tr, y_tr)
    proba = model.predict_proba(X_val)[:, 1]
    val_f1 = f1_score(y_val, proba > 0.5)
    val_auc = roc_auc_score(y_val, proba)
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("val_auc", val_auc)
    # MLflow 3: log model as an artifact in this run.
    mlflow.sklearn.log_model(model, name="model", input_example=X_val[:5], skops_trusted_types=["sklearn.tree._tree.Tree"])
    print("Validation samples:", len(X_val))
    print("Parameters:", params)
    print("val_f1:", round(val_f1, 4))
    print("val_auc:", round(val_auc, 4))
