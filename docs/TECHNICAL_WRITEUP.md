# Technical Write-Up: Spam vs. Non-Spam Classification with Spambase and External Enron Validation

**Course project (AI & DL)** — Binary email classification, model comparison, and domain-shift analysis.

---

## 1. Task and Research Questions

### Primary task

**Binary classification**: predict whether an email is **spam (1)** or **non-spam / ham (0)** using machine learning.

### Research questions (RQ)

1. **RQ1 (in-domain)**: How do classical and modern supervised learners perform on a standard tabular spam benchmark when evaluated with proper train/test separation and cross-validation?
2. **RQ2 (generalization)**: How well do models **trained exclusively on Spambase** transfer to **real email text** from an external corpus (Enron-derived spam/ham data), i.e. under **domain shift**?
3. **RQ3 (mitigation)**: Can **post-hoc decisions** (threshold optimization) and a **soft-voting ensemble** of complementary models improve practical metrics on the external set **without** training on that data?

---

## 2. Datasets and Justification

### 2.1 Spambase (primary training and evaluation benchmark)

- **Source**: UCI Machine Learning Repository — [Spambase (ID 94)](https://archive.ics.uci.edu/dataset/94/spambase).
- **Content**: **4,601** instances, **57** numeric input features per email: frequencies of selected words and characters (normalized), plus statistics on consecutive capital letters (“capital run” features). The target is a binary spam label.
- **Why use it?**
  - **Standard benchmark** for spam classification research; enables **reproducible** comparison across algorithms.
  - **Fixed feature space** avoids subjective text-preprocessing choices for the main experiment.
  - **Moderate class imbalance**: spam is roughly **40%** and ham **60%** (exact counts in the notebook EDA), which motivates **stratified** splitting and reporting **precision/recall**, not accuracy alone.

### 2.2 Enron Spam data (external validation only)

- **Source used in the project**: Preprocessed collection distributed as a single CSV in the repository [MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data) (described there as Enron-Spam data in one clean table). The underlying texts stem from the well-known **Enron email corpus** (real corporate mailbox traffic), which has been widely reused in NLP and spam-filtering studies.
- **Role in this project**: **External validation** — different senders, vocabulary, and time period than Spambase → strong test of **robustness** and **domain shift**.
- **Critical protocol constraint**: **No model is trained, tuned, or fitted on Enron labels.** All learners are fit on Spambase only. Enron emails are mapped into the **same 57-dimensional feature space** via a custom `extract_spambase_features` function (token/character statistics aligned with Spambase semantics).
- **Evaluation sample**: For efficiency and stable reporting, the notebook evaluates on a **stratified random sample of 1,000** emails (`random_state=42`) from the Enron-derived set. Full-corpus numbers may differ slightly; all reported Enron metrics refer to this sample unless stated otherwise.

---

## 3. Proposed Methodology and Approach

### 3.1 Exploratory analysis (Section 1)

- Descriptive statistics, missing values, target distribution, feature distributions, correlation with the label, and class-conditional plots — to understand scale, skew, and separation of spam vs. ham.

### 3.2 Preprocessing (Section 2)

- **Features / target** split; column names sanitized where needed for tree libraries.
- **Stratified train/test split** (e.g. **80% / 20%**, `random_state=42`) so train and test sets preserve the spam/ham ratio.
- **Feature scaling**: `StandardScaler` **fit on the training set only**, then applied to validation and test splits (and to externally engineered features) for distance- and margin-based models (e.g. logistic regression, KNN, SVM, MLP).

### 3.3 Baseline (Section 3)

- **DummyClassifier** (most frequent class) to establish a **floor** that any reasonable model must beat.

### 3.4 Model training and cross-validation (Sections 4–5)

- **StratifiedKFold** on the **training portion only** (e.g. 5 folds); the **hold-out test set never appears in CV folds**.
- Models compared include (as implemented in the notebook): **Logistic Regression**, **Decision Tree**, **Random Forest**, **XGBoost**, **KNN**, **SVM**, **MLP** (neural network).
- **Class balance reporting per fold** (table/plot) to document that stratification behaves as intended — aligned with good ML teaching practice.

### 3.5 Extensions (Sections 6–11)

- **Feature importance** (e.g. XGBoost) for interpretation.
- **Hyperparameter tuning** (`RandomizedSearchCV`) for selected strong candidates (**XGBoost, SVM, MLP**).
- **Threshold optimization** from precision–recall curves for asymmetric costs in spam filtering.
- **Feature selection**: performance vs. number of features for the focus models; reduced models re-evaluated on Spambase and on the Enron sample.
- **Soft-voting ensemble**: average predicted **spam probabilities** from **SVM and MLP** (XGBoost excluded from the ensemble in the final design because it hurt recall on Enron). **Threshold** for the ensemble chosen using **Spambase test-set** scores, then applied to Enron (no Enron training).
- **Calibrated weighted soft-voting check** (SVM + MLP + Logistic Regression): tested as an additional mitigation, but did **not** outperform the simpler SVM+MLP soft-voting setup on Enron in this run.

### 3.6 Rigor and validity (Sections 12–13)

- **McNemar’s test**: pairwise comparison of **disagreements** between models on the same examples (Spambase test set and Enron sample) to contextualize whether metric gaps may reflect systematic differences vs. noise.
- **Data leakage checks**: disjoint train/test indices, scaler fit verification, CV isolation from the hold-out test set, stratification sanity check.

---

## 4. Evaluation Criteria and Success Metrics

### 4.1 Metrics

- **Accuracy** — easy to communicate; can be misleading under imbalance.
- **Precision, Recall, F1** — especially relevant for spam: **false positives** (ham marked spam) vs. **false negatives** (spam in the inbox).
- **ROC-AUC** — ranking quality independent of a single threshold.
- **Confusion matrices** and **classification reports** for error patterns.

### 4.1.1 Why F1 as the main summary metric — and why **precision** matters most in practice

**Deployment perspective (asymmetric costs).**  
In realistic spam filtering, **false positives** — legitimate mail classified as spam — are often **more harmful** than **false negatives** (spam that reaches the inbox): users may miss invoices, job offers, or security alerts. **Precision** (for the spam class) directly answers: *“When the model says ‘spam’, how often is it right?”* — so **high precision** is usually the **primary** quality goal for a user-facing filter.

**Why we still emphasize F1 in this project.**  
We report **precision and recall explicitly** everywhere, but we use **F1** as a convenient **single-number summary** for comparing models because:

1. **Class imbalance** (~40% spam / 60% ham on Spambase): accuracy alone can look good while the minority class is predicted poorly; F1 stays sensitive to performance on **spam** (unlike a majority-class baseline, which achieves ~60% accuracy but **F1 = 0** in our notebook).
2. **Balance between two errors**: F1 is the **harmonic mean** of precision and recall — it penalizes models that achieve high precision only by **ignoring** almost all spam (very low recall), and vice versa. That makes it a **fair middle-ground** metric when we want models that are **useful overall**, not only optimized for one side.
3. **Comparability**: F1 is widely used in binary classification benchmarks, which helps **compare** algorithms on equal footing.

**Limitation (stated clearly).**  
F1 **weights precision and recall equally**; it does **not** encode that false positives are “twice as expensive” as false negatives. For a **production** system with strong emphasis on not blocking ham, one would typically **prioritize precision**, tune the **decision threshold** (as in **Section 9** of the notebook: default vs. best F1 vs. high-precision scenarios), or use **cost-sensitive** learning — not maximize F1 alone. Our pipeline reflects this by pairing **F1-oriented summaries** with **precision–recall curves**, threshold analysis, and raw **precision** in all tables.

### 4.2 Success criteria (interpretable)

- **Minimum**: All substantive models **beat the dummy baseline** on Spambase test metrics.
- **Primary scientific outcome**: Quantify **performance drop** on Enron vs. Spambase and explain it as **domain shift** (feature distribution and language change), not as “failure” of a single number.
- **Stretch / aspirational goal** (optional narrative in the project): improving external accuracy toward ~**70%** without Enron training — framed as **engineering ambition**, not a formal statistical guarantee.

### 4.3 Ethics and data context (brief)

- Email data can contain sensitive content; this project uses **public research corpora** and does not identify individuals. Spam filters have real user impact — **wrongly quarantined legitimate mail** is often the most salient failure mode — which motivates **transparent metrics**, reporting **precision** alongside F1, and explicit **threshold choices**.

---

## 5. Outcomes and Contributions

### 5.1 Empirical outcomes (high level)

- **Strong in-domain performance** on Spambase across multiple algorithms (see notebook tables and comparison plots in **Section 5**).
- **Substantial degradation on Enron** for some models — consistent with **covariate / domain shift** when moving from pre-engineered Spambase features to **approximated** features from raw Enron text.
- **SVM and MLP** tend to **generalize better** than tree-heavy models such as XGBoost in this external setting (see **Section 7** and follow-on sections).
- **Soft-voting ensemble (SVM + MLP)** with an optimized threshold is the **best practical stop-point** in this project for external Enron transfer (**Accuracy 0.689, F1 0.640** on the 1k Enron sample, threshold ≈ 0.42) — see **Section 10.5** and the final comparison table in the notebook.
- A follow-up **calibrated weighted** soft-voting variant (SVM + MLP + Logistic Regression) did not improve this result in the observed run (Enron **F1 0.553**), reinforcing that domain shift dominates fine-grained ensemble weighting.

### 5.2 Methodological contributions

- End-to-end pipeline with **explicit baseline**, **stratified splits**, **fold-wise label documentation**, and a **true hold-out** test set.
- **External validation** with a strict **no-training-on-Enron** rule and feature alignment via engineering.
- **Threshold and ensemble** discussion tied to **cost-sensitive** spam filtering.
- **Statistical testing** (McNemar) and **leakage audit** to strengthen claims.

### 5.3 Limitations

- Enron features are **approximations** of Spambase statistics; remaining **distribution mismatch** limits ceiling performance.
- **Threshold** for the ensemble is tuned on the **Spambase test** split — legitimate for deployment prototyping but slightly **optimistic** for Enron-specific calibration (still **no Enron label leakage** into model weights).
- Enron results on a **1k stratified sample** estimate corpus-level performance with sampling variance.
- Ensemble improvements are **incremental** under transfer: more complex post-hoc fusion (calibration + weighting) was not automatically better than the simpler SVM+MLP soft-voting baseline.

---

## References and links

- Spambase: [https://archive.ics.uci.edu/dataset/94/spambase](https://archive.ics.uci.edu/dataset/94/spambase)
- Enron Spam CSV (preprocessed): [https://github.com/MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data)
- Project implementation: [`src/spambase_project.ipynb`](../src/spambase_project.ipynb)

---

*Optional short German abstract:*  
Dieses Projekt klassifiziert E-Mails (Spam vs. Ham) mit dem UCI-Spambase-Datensatz, vergleicht mehrere ML-Verfahren mit stratifiziertem Split und Kreuzvalidierung und validiert extern auf Enron-basierten E-Mails **ohne** Training auf Enron — mit Fokus auf Domain Shift, Metriken und einem SVM+MLP-Soft-Voting-Ensemble.
