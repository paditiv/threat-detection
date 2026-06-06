
import json, os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime


TRAIN_CSV = r"C:\Users\Aditi Vaibhav Patil\Desktop\Major Project\combined csvs\train_dataset_encoded.csv"
TEST_CSV  = r"C:\Users\Aditi Vaibhav Patil\Desktop\Major Project\combined csvs\train_dataset_encoded.csv"
OUTPUT_DIR = r"C:\Users\Aditi Vaibhav Patil\Desktop\Major Project\ml output"
LABEL_COL = "label"   


os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    if LABEL_COL not in df.columns:
        print(f"[ERROR] '{LABEL_COL}' column not found in {path}")
        sys.exit(1)
    return df

train_df = load_csv(TRAIN_CSV)
test_df  = load_csv(TEST_CSV)


y_train = train_df[LABEL_COL]
X_train = train_df.drop(columns=[LABEL_COL])

y_test  = test_df[LABEL_COL]
X_test  = test_df.drop(columns=[LABEL_COL])


num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]


numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols),
    ],
    remainder="drop"
)


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42
)
lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=None,
    solver="lbfgs",
    multi_class="auto"
)

pipelines = {
    "RandomForest": Pipeline(steps=[("prep", preprocess), ("clf", rf)]),
    "LogReg": Pipeline(steps=[("prep", preprocess), ("clf", lr)])
}

results = {}
best_name, best_f1, best_pipe = None, -1.0, None

for name, pipe in pipelines.items():
    print(f"\n--- Training {name} ---")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    results[name] = {
        "macro_f1": f1,
        "report": report,
        "confusion_matrix": cm
    }
    print(f"{name} Macro F1: {f1:.4f}")
    if f1 > best_f1:
        best_name, best_f1, best_pipe = name, f1, pipe


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{timestamp}.json")
model_path   = os.path.join(OUTPUT_DIR, f"best_model_{best_name}_{timestamp}.joblib")
cols_path    = os.path.join(OUTPUT_DIR, f"feature_columns_{timestamp}.json")

with open(metrics_path, "w") as f:
    json.dump({
        "candidates": results,
        "best_model": best_name,
        "macro_f1": best_f1
    }, f, indent=2)

joblib.dump(best_pipe, model_path)


with open(cols_path, "w") as f:
    json.dump({
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "all_input_cols": X_train.columns.tolist(),
        "label_col": LABEL_COL
    }, f, indent=2)

print("\n=== Training complete ===")
print(f"Best model: {best_name}  |  Macro F1: {best_f1:.4f}")
print(f"Saved metrics: {metrics_path}")
print(f"Saved model:   {model_path}")
print(f"Saved cols:    {cols_path}")
