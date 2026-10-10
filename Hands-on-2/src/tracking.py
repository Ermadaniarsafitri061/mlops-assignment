
from pathlib import Path
import hashlib

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from mlflow.models import infer_signature


# ============================================================
# 1. KONFIGURASI PROYEK DAN MLFLOW
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bc-malignant-detection")


# ============================================================
# 2. LOAD DATA TRAINING
# ============================================================

train_df = pd.read_csv(ROOT / "data" / "train.csv")

X_train = train_df.drop(columns="target")
y_train = train_df["target"]

TRAIN_DS = mlflow.data.from_pandas(
    train_df,
    source="data/train.csv",
    name="bc-train",
    targets="target",
)


# ============================================================
# 3. HASH DATA UNTUK REPRODUCIBILITY
#    Pertahankan perhitungan hash yang digunakan data.py
# ============================================================

original_df = load_breast_cancer(as_frame=True).frame
original_df["target"] = 1 - original_df["target"]

DATA_MD5 = hashlib.md5(
    pd.util.hash_pandas_object(
        original_df,
        index=True,
    ).values
).hexdigest()

COMMON_TAGS = {
    "data_md5": DATA_MD5,
    "split_seed": "42",
    "dataset": "sklearn-breast-cancer",
}


# ============================================================
# 4. KONFIGURASI CROSS VALIDATION DAN METRIK
# ============================================================

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

SCORING = ["recall", "precision", "f1", "roc_auc"]


# ============================================================
# 5. FUNGSI EKSPERIMEN DAN LOGGING MLFLOW
# ============================================================

def run_experiment(run_name, pipe, tags=None, note=""):
    tags = tags or {}

    with mlflow.start_run(run_name=run_name) as run:

        # A. Konteks eksperimen
        mlflow.set_tags({
            **COMMON_TAGS,
            "candidate": "true",
            **tags,
        })

        mlflow.set_tag("mlflow.note.content", note)

        mlflow.log_input(
            TRAIN_DS,
            context="training",
        )

        # B. Logging parameter
        mlflow.log_param(
            "pipeline_steps",
            " -> ".join(pipe.named_steps),
        )

        estimator_params = {
            f"clf__{key}": value
            for key, value in pipe[-1].get_params().items()
            if isinstance(
                value,
                (str, int, float, bool, type(None)),
            )
        }

        mlflow.log_params(estimator_params)

        # C. Evaluasi 5-fold Stratified Cross Validation
        cv = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=CV,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=-1,
        )

        for metric in SCORING:
            mlflow.log_metric(
                f"cv_{metric}_mean",
                float(cv[f"test_{metric}"].mean()),
            )

            mlflow.log_metric(
                f"cv_{metric}_std",
                float(cv[f"test_{metric}"].std()),
            )

        mlflow.log_metric(
            "fit_time_s",
            float(cv["fit_time"].mean()),
        )

        # D. Prediksi out-of-fold untuk analisis CV
        pred = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=CV,
            n_jobs=-1,
        )

        report = classification_report(
            y_train,
            pred,
            labels=[0, 1],
            target_names=["benign", "malignant"],
            zero_division=0,
        )

        report_path = ROOT / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")

        mlflow.log_artifact(
            str(report_path),
            artifact_path="reports",
        )

        # E. Confusion matrix dari prediksi cross-validation
        disp = ConfusionMatrixDisplay.from_predictions(
            y_train,
            pred,
            labels=[0, 1],
            display_labels=["benign", "malignant"],
        )

        mlflow.log_figure(
            disp.figure_,
            "plots/confusion_matrix_cv.png",
        )

        plt.close(disp.figure_)

        # F. Latih pipeline pada seluruh training set
        pipe.fit(X_train, y_train)

        # G. Signature dan penyimpanan model ke MLflow
        signature = infer_signature(
            X_train,
            pipe.predict(X_train.head(5)),
        )

        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        # H. Simpan metadata model
        mlflow.set_tag("model_uri", info.model_uri)

        mlflow.set_tag(
            "model_family",
            tags.get("model_family", "unknown"),
        )

        # I. Tampilkan ringkasan hasil
        metrics_summary = {
            metric: round(
                float(cv[f"test_{metric}"].mean()),
                4,
            )
            for metric in SCORING
        }

        print(
            run_name,
            metrics_summary,
            "run_id=",
            run.info.run_id,
        )

        return run.info.run_id, info.model_uri
