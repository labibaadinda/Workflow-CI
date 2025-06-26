import os
import sys
import mlflow
import mlflow.xgboost
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "xgboost"])
    import xgboost as xgb


def train_model(file_path):
    print(f"\n[INFO] Membaca dataset dari: {file_path}")
    df = pd.read_csv(file_path)
    print("\n[INFO] Tipe data kolom:")
    print(df.dtypes)

    X = df.drop(['Sleep Disorder'], axis=1)
    y = df['Sleep Disorder']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        'objective': ['multi:softmax'],
        'num_class': [3],
        'eval_metric': ['mlogloss'],
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'lambda': [1],
        'alpha': [1],
        'gamma': [0.1]
    }

    xgb_model = xgb.XGBClassifier(use_label_encoder=False)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='accuracy',
        verbose=2
    )

    with mlflow.start_run() as run:
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Logging metrics to MLflow
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)

        # Save model to local directory (for Docker)
        mlflow.xgboost.save_model(
            best_model,
            path="xgboost_model_dir"
        )

        # Optional: Save backup model file
        joblib.dump(best_model, "xgboost_best_model.joblib")
        mlflow.log_artifact("xgboost_best_model.joblib")

        print("\n--- Evaluation Results ---")
        print(f"Best CV Accuracy: {best_score:.4f}")
        print(f"Test Accuracy   : {acc:.4f}")
        print(f"Precision       : {prec:.4f}")
        print(f"Recall          : {rec:.4f}")
        print(f"F1 Score        : {f1:.4f}")
        print(f"[INFO] MLflow run ID: {run.info.run_id}")
        print(f"[INFO] Model saved to: ./xgboost_model_dir")


if __name__ == "__main__":
    file_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleep-health_life-style_preprocessing.csv")
    )
    train_model(file_path)
