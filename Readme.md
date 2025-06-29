# ✅ Iris Model Sanity Test (CI-enabled with DVC)

This repository sets up a CI pipeline for sanity testing a Decision Tree classifier trained on the Iris dataset. All large files such as data and models are tracked via **DVC**, and not committed directly to the repository.

## 📁 Files and Structure

- `train.py` — Script to train the model and generate metrics.
- `test.py` — Unittest script to validate the model on a sample.
- `samples/sample.csv.dvc` — DVC pointer file for test sample.
- `model.pkl.dvc` — DVC pointer file for trained model.
- `data.dvc` — DVC pointer for the full training dataset.
- `metrics.txt` — Training metrics (accuracy, log loss).
- `.github/workflows/sanity.yml` — GitHub Actions CI workflow.
- `.gitignore` — Ignores raw files (`data/`, `model.pkl`, etc.).
- `dvc.yaml`, `dvc.lock` — Pipeline and dependency tracking (optional).

> 📌 **Note**: Actual CSV, model, and dataset files are *not* committed — they are pulled using `dvc pull` from a GCS remote during CI.

## 🔁 CI Workflow Triggers

- **Automatically** runs on every Pull Request to the `main` branch.
- Can also be **manually triggered** via GitHub UI (`workflow_dispatch`).

## 🧪 What the Workflow Does

- Sets up Python environment.
- Installs dependencies and DVC.
- Authenticates with GCP using a secret (`GCP_CREDENTIALS`).
- Pulls the model and sample test set using `dvc pull`.
- Executes `unittest` to validate model accuracy on the sample.
- Fails the build if accuracy < 100%.
- Publishes a report using CML as a PR comment.

## ✅ Requirements

- Python 3.10
- DVC with `dvc-gs` extension
- GitHub repo secret named `GCP_CREDENTIALS` with GCP service account credentials

## 🧩 Note

Ensure all large files are properly tracked using:

```bash
dvc add data/iris.csv
dvc add model.pkl
dvc add samples/sample.csv
dvc push
```

Only the `.dvc` pointer files should be committed to Git.

