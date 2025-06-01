import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib
from tqdm import tqdm
import time

steps = [
    "Loading dataset",
    "Preparing features and target",
    "Encoding target",
    "Splitting data",
    "Training model",
    "Making predictions",
    "Calculating metrics",
    "Saving metrics",
    "Saving model"
]

with tqdm(total=len(steps)) as pbar:
    # Step 1: Load dataset
    data_path = 'data/iris.csv'
    df = pd.read_csv(data_path)
    time.sleep(0.2)
    pbar.update(1)

    # Step 2: Features and target
    X = df.drop(columns=['species'])  # assuming 'species' is the target column
    y = df['species']
    time.sleep(0.2)
    pbar.update(1)

    # Step 3: Encode target if necessary
    if y.dtype == object:
        y = pd.factorize(y)[0]
    time.sleep(0.2)
    pbar.update(1)

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    time.sleep(0.2)
    pbar.update(1)

    # Step 5: Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    time.sleep(0.2)
    pbar.update(1)

    # Step 6: Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)
    time.sleep(0.2)
    pbar.update(1)

    # Step 7: Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_loss = log_loss(y_train, y_train_proba)
    test_loss = log_loss(y_test, y_test_proba)
    time.sleep(0.2)
    pbar.update(1)

    # Step 8: Save metrics
    with open("metrics.txt", "w") as f:
        f.write(f"Train size: {len(X_train)}\n")
        f.write(f"Test size: {len(X_test)}\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Train Log Loss: {train_loss:.4f}\n")
        f.write(f"Test Log Loss: {test_loss:.4f}\n")
    time.sleep(0.2)
    pbar.update(1)

    # Step 9: Save model
    joblib.dump(model, "model.pkl")
    time.sleep(0.2)
    pbar.update(1)

print("✅ Training complete. Metrics and model saved.")
