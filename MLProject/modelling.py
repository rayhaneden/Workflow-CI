import argparse
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

# Argument parser untuk MLflow Project
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=150)
parser.add_argument("--max_depth", type=str, default="None")
parser.add_argument("--min_samples_leaf", type=int, default=1)
parser.add_argument("--min_samples_split", type=int, default=2)
args = parser.parse_args()

# Convert max_depth string to proper type
max_depth = None if args.max_depth == "None" else int(args.max_depth)

# Load data
data = pd.read_csv("pokemon_dataset_preprocessing.csv")
X = data.drop(columns=["first_winner"])
y = data["first_winner"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlflow.set_experiment("Pokemon_Battle_Prediction_CI")

with mlflow.start_run():
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred))

    print("Model training selesai dengan parameter optimal hasil tuning!")
