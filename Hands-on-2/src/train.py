import json
from pathlib import Path
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tracking import run_experiment, ROOT

# L2 — baseline
run_experiment("00-dummy-most-frequent", Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
               {"model_family": "dummy", "stage": "L2"}, "Baseline sederhana")
run_experiment("01-logreg-noscale", Pipeline([("clf", LogisticRegression(max_iter=5000))]),
               {"model_family": "logistic-regression", "stage": "L2"})
run_experiment("02-logreg-scaled", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
               {"model_family": "logistic-regression-scaled", "stage": "L2"})

# L3 — compare five model families with a shared 5-fold stratified CV
candidates = {
    "knn": KNeighborsClassifier(n_neighbors=5),
    "tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "gb": GradientBoostingClassifier(random_state=42),
    "svc": SVC(kernel="rbf", random_state=42),
}
for family, clf in candidates.items():
    run_experiment(f"03-{family}-default", Pipeline([("scaler", StandardScaler()), ("clf", clf)]),
                   {"model_family": family, "stage": "L3"})

# L4 — GridSearchCV on SVC; log parent run and each tested configuration as a child run.
Xy = __import__("pandas").read_csv(ROOT / "data" / "train.csv")
X_train, y_train = Xy.drop(columns="target"), Xy["target"]
pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=42))])
grid = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, grid, cv=cv, scoring={"recall":"recall", "precision":"precision", "f1":"f1", "roc_auc":"roc_auc"},
                  refit="recall", n_jobs=-1, return_train_score=False)
with mlflow.start_run(run_name="09-svc-gridsearch") as parent:
    mlflow.set_tags({"stage":"L4", "model_family":"svc", "data_md5": __import__("tracking").DATA_MD5})
    mlflow.log_param("search_space", json.dumps(grid))
    mlflow.log_param("cv", "StratifiedKFold(5, shuffle=True, random_state=42)")
    gs.fit(X_train, y_train)
    mlflow.log_metric("best_cv_recall", float(gs.best_score_))
    mlflow.log_params({f"best_{k}": v for k,v in gs.best_params_.items()})
    mlflow.log_dict(grid, "search_space.json")
    # log all 12 parameter combinations as nested child runs for transparent tracking
    results = gs.cv_results_
    for i, p in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"svc-trial-{i+1:02d}", nested=True):
            mlflow.set_tags({"stage":"L4", "candidate":"true", "model_family":"svc", "data_md5":__import__("tracking").DATA_MD5})
            mlflow.log_params(p)
            mlflow.log_metric("cv_recall_mean", float(results["mean_test_recall"][i]))
            mlflow.log_metric("cv_recall_std", float(results["std_test_recall"][i]))
            mlflow.log_metric("cv_precision_mean", float(results["mean_test_precision"][i]))
            mlflow.log_metric("cv_f1_mean", float(results["mean_test_f1"][i]))
            mlflow.log_metric("cv_roc_auc_mean", float(results["mean_test_roc_auc"][i]))
    # Persist the tuned estimator as its own model candidate run so the later selector can load it.
with mlflow.start_run(run_name="10-svc-gridsearch-best") as run:
    mlflow.set_tags({"stage":"L4", "candidate":"true", "model_family":"svc", "data_md5":__import__("tracking").DATA_MD5})
    mlflow.log_params(gs.best_params_)
    for metric in ["recall", "precision", "f1", "roc_auc"]:
        idx = gs.best_index_
        mlflow.log_metric(f"cv_{metric}_mean", float(gs.cv_results_[f"mean_test_{metric}"][idx]))
        mlflow.log_metric(f"cv_{metric}_std", float(gs.cv_results_[f"std_test_{metric}"][idx]))
    info = mlflow.sklearn.log_model(gs.best_estimator_, name="model", input_example=X_train.head(3))
    mlflow.set_tag("model_uri", info.model_uri)
    print("Best SVC params:", gs.best_params_)
    print("Best CV recall:", round(gs.best_score_, 4))
