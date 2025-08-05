import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import time

steps = [
    "Loading datasets",
    "Preparing features and target",
    "Encoding target",
    "Training model",
    "Making predictions",
    "Calculating metrics",
    "Saving metrics",
    "Saving model"
]

with tqdm(total=len(steps)) as pbar:
    # Step 1: Load train and validation datasets
    train_df = pd.read_csv('data/iris_train_poison.csv')
    val_df = pd.read_csv('data/val_iris.csv')
    time.sleep(0.2)
    pbar.update(1)

    # Step 2: Separate features and target
    X_train = train_df.drop(columns=['species'])
    y_train_raw = train_df['species']
    X_val = val_df.drop(columns=['species'])
    y_val_raw = val_df['species']
    time.sleep(0.2)
    pbar.update(1)

    # Step 3: Encode target labels using LabelEncoder
    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)  # assumes all val labels exist in train
    time.sleep(0.2)
    pbar.update(1)

    # Step 4: Train model with regularization
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    time.sleep(0.2)
    pbar.update(1)

    # Step 5: Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    time.sleep(0.2)
    pbar.update(1)

    # Step 6: Compute metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    train_loss = log_loss(y_train, y_train_proba)
    val_loss = log_loss(y_val, y_val_proba)

    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')

    train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
    val_precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)

    train_recall = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average='macro', zero_division=0)

    # AUC using One-vs-Rest
    try:
        train_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr')
        val_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr')
    except ValueError:
        train_auc = val_auc = -1  # Fallback for edge cases (e.g., single class in val)

    time.sleep(0.2)
    pbar.update(1)

    # Step 7: Save metrics
    with open("metrics.txt", "w") as f:
        f.write(f"Train size: {len(X_train)}\n")
        f.write(f"Validation size: {len(X_val)}\n\n")

        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")

        f.write(f"Train Log Loss: {train_loss:.4f}\n")
        f.write(f"Validation Log Loss: {val_loss:.4f}\n\n")

        f.write(f"Train F1 Score: {train_f1:.4f}\n")
        f.write(f"Validation F1 Score: {val_f1:.4f}\n")

        f.write(f"Train Precision: {train_precision:.4f}\n")
        f.write(f"Validation Precision: {val_precision:.4f}\n")

        f.write(f"Train Recall: {train_recall:.4f}\n")
        f.write(f"Validation Recall: {val_recall:.4f}\n")

        f.write(f"Train AUC: {train_auc:.4f}\n")
        f.write(f"Validation AUC: {val_auc:.4f}\n")
    time.sleep(0.2)
    pbar.update(1)

    # Step 8: Save model
    joblib.dump(model, "model.pkl")
    time.sleep(0.2)
    pbar.update(1)

print("✅ Training complete. Metrics and model saved.")
