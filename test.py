import unittest
import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load

class TestModelAccuracy(unittest.TestCase):
    def test_model_on_sample(self):
        # Load sample data
        sample_df = pd.read_csv("samples/sample.csv")
        X = sample_df.drop(columns=["species"])
        y_true_raw = sample_df["species"]

        # Encode species labels using same logic as train.py
        y_true, uniques = pd.factorize(y_true_raw)

        # Load trained model
        model = load("model.pkl")

        # Predict
        y_pred = model.predict(X)

        # Calculate accuracy
        acc = accuracy_score(y_true, y_pred)

        # Save metric
        # with open("metrics.txt", "w") as f:
            # f.write(f"Sample Accuracy: {acc:.4f}\n")

        # Assert perfect accuracy
        self.assertEqual(acc, 1.0, f"Expected 100% accuracy, got {acc:.4f}")

if __name__ == "__main__":
    unittest.main()

