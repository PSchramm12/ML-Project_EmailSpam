# ML Project: Email Spam Classification

Course project for AI & Deep Learning: binary email classification (`spam` vs. `ham`) with in-domain benchmarking on UCI Spambase and external validation on Enron-derived emails.

## Project Goals

- Compare multiple supervised models on a standard tabular spam benchmark.
- Evaluate generalization under domain shift (train on Spambase, validate on Enron-derived emails).
- Analyze practical trade-offs with threshold tuning and a soft-voting ensemble.

## Repository Structure

- `src/spambase_project.ipynb` - main notebook with full pipeline and results.
- `docs/TECHNICAL_WRITEUP.md` - technical report (methodology, metrics, limitations).
- `docs/PRESENTATION_STRUCTURE.md` - slide/storyline structure and result highlights.
- `data/` - local data assets used by the notebook (not all files are versioned).
- `images/` - figures and visuals used for documentation/presentation.

## Datasets

### 1) UCI Spambase (training and in-domain evaluation)
- 4,601 samples
- 57 engineered numeric features per email
- Binary target (`1 = spam`, `0 = ham`)

Source: [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase)

### 2) Enron-derived spam dataset (external validation)
- Used only for external testing / transfer analysis
- No model training on Enron labels in the project protocol

Source used in this project: [MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data)

## Methods (High Level)

- Stratified train/test split and stratified cross-validation
- Baseline model (`DummyClassifier`) for minimum performance reference
- Model comparison across:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - KNN
  - SVM
  - MLP (Neural Network)
- Hyperparameter tuning for selected candidates
- Threshold optimization from precision-recall behavior
- Soft-voting ensemble (SVM + MLP in final configuration)
- Robustness checks (including leakage checks and McNemar tests)

## Main Findings (Short)

- Strong in-domain performance on Spambase across multiple models.
- Clear performance drop on Enron external validation, consistent with domain shift.
- SVM and MLP generalize more robustly in the external setting than tree-heavy variants in this setup.
- A tuned SVM+MLP soft-voting ensemble improves external performance over individual models in the reported run.

## Reproducibility

1. Create a Python environment (recommended: Python 3.10+).
2. Install required libraries used in the notebook (for example: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`).
3. Open and run `src/spambase_project.ipynb` from top to bottom.
4. For external validation, ensure Enron CSV access is available as expected by the notebook.

## Notes

- Exact metrics can vary slightly with library versions and reruns.
- Results in docs reflect the executed notebook snapshot referenced by the team.

## Authors

Maintained by the project team in this repository.
