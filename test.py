import unittest
import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load

class TestModelAccuracy(unittest.TestCase):

    def test_model_on_sample(self):
        # Load sample data from the correct path
        sample_df = pd.read_csv("samples/sample.csv")
        X = sample_df.drop(columns=["species"])
        y_true = sample_df["species"]

        # Load model
        model = load("model.pkl")

        # Predict
        y_pred = model.predict(X)

        # Calculate and log accuracy
        acc = accuracy_score(y_true, y_pred)
        with open("metrics.csv", "w") as f:
            f.write("accuracy,{:.2f}\n".format(acc))

        # Assert perfect accuracy
        self.assertEqual(acc, 1.0, f"Expected 100% accuracy, but got {acc:.2f}")

if __name__ == "__main__":
    unittest.main()
