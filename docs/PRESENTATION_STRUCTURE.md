# Presentation Structure (~10 minutes)

**Audience**: Mixed technical level — stay high-level; one clear story: *benchmark → good practice → models → domain shift → ensemble insight*.  
**Notebook**: [`src/spambase_project.ipynb`](../src/spambase_project.ipynb) — export figures as PNG/PDF from the matching sections before building slides.

**Timing budget (target ~10:00)**

| Block        | ~Time   |
| ------------ | ------- |
| Intro + RQ   | ~2:00   |
| Data + CV    | ~2:30   |
| Method       | ~1:30   |
| Results      | ~3:00   |
| Wrap-up + Q  | ~1:00   |

### Metrics story (use on Slide 2 and/or 7 — ~30 s)

- **In production**, you mainly want **high precision** for “spam”: *legitimate mail wrongly flagged as spam* (false positives) is usually **more costly** than some spam slipping through (false negatives).
- **In this project**, **F1** is our main **single-number summary** to compare models: it combines precision and recall, handles **imbalance** better than accuracy alone, and avoids “always predict ham” tricks (high accuracy, **F1 = 0** on our baseline).
- **Caveat**: F1 treats precision and recall **equally** — it is **not** the same as “optimize for precision only.” We therefore always show **precision & recall** in tables and use **Section 9** (threshold / high-precision scenarios) for asymmetric costs.

---

## Results snapshot (from executed notebook outputs)

These numbers come from the **saved outputs** in [`src/spambase_project.ipynb`](../src/spambase_project.ipynb). If you **re-run** the notebook (different seeds, library versions, or data download), values may shift slightly — use your fresh outputs for the final slides.

### Spambase — full dataset (EDA, Section 1)

| Class   | Count (approx.) | Share   |
| ------- | --------------- | ------- |
| Ham (0) | 2,788           | ~60.6%  |
| Spam (1)| 1,813           | **39.4%** |

### Train / test & CV (Sections 2 & 4)

- **Split**: stratified **80% train / 20% test**, `random_state=42`.
- **Training-set spam rate**: **39.4%** (matches full data under stratification).
- **Stratified 5-fold CV** (on training data only): every fold had **39.4%** spam in both train and validation partitions; train size **2944**, val size **736** per fold → **holdout test set never appears in CV**.

### Baseline — DummyClassifier (Section 3)

Always predicts majority class (ham) on the **Spambase test set**:

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | **0.606** |
| Precision | 0.000 (no positive spam predictions) |
| Recall    | 0.000 |
| F1        | 0.000 |

→ All real models must beat **~60.6% accuracy** and non-zero F1.

### Spambase — hold-out test metrics (Section 5)

| Model                 | Test Acc | Test F1  | Test ROC-AUC |
| --------------------- | -------- | -------- | ------------ |
| Logistic Regression   | 0.929    | 0.909    | 0.970        |
| Decision Tree         | 0.911    | 0.888    | 0.908        |
| Random Forest         | 0.946    | 0.930    | 0.983        |
| **XGBoost**           | **0.946**| **0.931**| **0.987**    |
| KNN                   | 0.908    | 0.882    | 0.951        |
| SVM                   | 0.927    | 0.906    | 0.967        |
| Neural Network (MLP)  | 0.939    | 0.922    | 0.983        |

**CV (5-fold, train only)** — XGBoost example: CV F1 **0.944 ± 0.010**; other models similarly high and stable (see notebook table).

### Enron — external validation (Section 7)

- **Source**: [MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data) — **33,716** emails in full CSV (per notebook printout).
- **Evaluation sample**: **n = 1000**, stratified, `random_state=42` — **no training on Enron**; features via `extract_spambase_features`.

**All models, default threshold 0.5** (notebook “External Validation Results”):

| Model                 | Accuracy | Precision | Recall  | F1     |
| --------------------- | -------- | --------- | ------- | ------ |
| Logistic Regression   | 0.656    | 0.769     | 0.464   | 0.578  |
| Decision Tree         | 0.578    | 0.818     | 0.220   | 0.347  |
| Random Forest         | 0.567    | 0.952     | 0.157   | 0.270  |
| XGBoost               | 0.595    | 0.906     | 0.228   | 0.364  |
| KNN                   | 0.637    | 0.728     | 0.458   | 0.562  |
| **SVM**               | **0.679**| 0.807     | 0.485   | **0.606** |
| Neural Network (MLP)  | 0.666    | 0.769     | 0.491   | 0.600  |

**Story**: Strong **domain shift** — same models, much lower external performance; **XGBoost** keeps high precision but **very low recall** on Enron (misses spam).

### Hyperparameter tuning on Enron (Section 8.5, optional slide)

Tuned XGBoost / SVM / MLP on Spambase did **not** improve Enron accuracy in this run (e.g. XGBoost 0.595 → 0.572; SVM 0.679 → 0.662) — useful as “tuning on source domain ≠ fixing target shift.”

### Soft-voting ensemble SVM + MLP (Section 10.5)

- **XGBoost excluded** from ensemble (hurts recall on Enron).
- **Threshold** for “Best F1” chosen on **Spambase test** pooled probabilities: **t ≈ 0.42**.

**Enron (ensemble vs. singles, t = 0.42 for ensemble):**

| Model                     | Accuracy | F1      |
| ------------------------- | -------- | ------- |
| XGBoost (t = 0.5)         | 0.595    | 0.364   |
| SVM                       | 0.679    | 0.606   |
| Neural Network (MLP)      | 0.666    | 0.600   |
| **Ensemble SVM+MLP (t≈0.42)** | **0.689** | **0.640** |

**Enron metrics at tuned threshold** (printed in notebook): e.g. Best F1 scenario — Acc **0.689**, Prec **0.780**, Rec **0.542**, F1 **0.640** (vs. default 0.5 scenarios on same plot).

### Calibrated weighted soft-voting check (Section 10.6)

- Additional variant tested: calibrated weighted soft voting with **SVM + MLP + Logistic Regression**.
- Best found config in this run: `t=0.57`, weights `SVM=0.30`, `MLP=0.20`, `LogReg=0.50`.
- Enron result: **Accuracy 0.653, F1 0.553** → **worse** than simple SVM+MLP soft voting.
- Presentation takeaway: **SVM+MLP soft voting (t≈0.42)** remains the best practical external model.

### McNemar’s test (Section 12)

**Spambase test** (XGBoost vs. SVM vs. MLP): XGBoost vs. SVM **significant** (*p* ≈ 0.017); XGBoost vs. MLP and SVM vs. MLP **not significant** at α = 0.05.

**Enron**: XGBoost vs. SVM and XGBoost vs. MLP **highly significant** (***p* < 0.001**); **SVM vs. MLP not significant** (*p* ≈ 0.31) — consistent with similar external performance and ensemble benefit.

### Data leakage check (Section 13)

All four checks **passed** (disjoint train/test, scaler fit on train only, CV excludes test, stratification preserved).

---

## Suggested graphics — master list (export from notebook)

| Priority | Notebook section | Figure / output | Use on slide(s) |
| -------- | ----------------- | --------------- | --------------- |
| 1 | **1** (EDA) | Target distribution (spam vs. ham bar or pie) | 4 |
| 2 | **4** (after baseline) | *Class Distribution per K-Fold* bar chart | 5 |
| 3 | **5.2** | *Model Comparison — Test Set Metrics* bar chart (+ baseline line) | 8 |
| 4 | **5** or **7** | ROC curves (multi-model) if space on slide 8 | 8 (optional) |
| 5 | **7** | External validation bar chart **or** confusion-matrix grid | 9 |
| 6 | **10.5** | Ensemble threshold curves + Enron bar comparison | 9 |
| 6b | **10.6** | Calibrated weighted soft-voting result (optional table only) | 9 / appendix |
| 7 | **9** | Precision–recall or metrics-vs-threshold (one focus model) | 10 (if Option A) |
| 8 | **10.2** | F1 vs. number of features (XGBoost/SVM/MLP) | 10 (if Option B) |
| 9 | **12** | McNemar disagreement heatmaps | 10 (if Option C) |
| 10 | **13** | Data leakage “checks passed” bar (or screenshot table) | 10 (if Option D) |

**Custom (non-notebook)**: simple **pipeline flowchart** (Slide 7) in PowerPoint / Google Slides / draw.io; optional **envelope / shield icon** on Slide 2 (motivation).

---

## Slide 1 — Title (~30 s)

**Slide title (suggested)**  
Spam vs. Non-Spam: Multi-Model Benchmark on Spambase + External Enron Validation

**Bullets (on slide)**

- Course / institution (e.g. TUM — AI & DL)
- Your name
- Date

**Suggested graphic**

- Optional: university or course logo (top corner).
- Optional: simple **binary icon** (e.g. inbox + “spam” tag) — stock icon, not from notebook.

**Speaker notes**

- One sentence: “We compare seven ML models on a standard spam benchmark and stress-test them on real Enron mail **without** training there.”

---

## Slide 2 — Motivation & relevance (~1:00–1:30)

**Slide title (suggested)**  
Why spam classification (and why careful metrics)

**Bullets (on slide)**

- Email is still a main channel for **phishing**, **scams**, and **malware**.
- **False positives hurt users**: legitimate mail marked as spam → missed invoices, job offers, 2FA codes.
- **False negatives** are annoying but often **less critical** than blocking real mail.
- → **Primary product goal**: **high precision** on “spam” (*when we say spam, we should be right*).
- For **this study**: **F1** summarizes **precision + recall** for **fair model ranking** under ~40/60 class balance; we still report **precision/recall** everywhere and **Section 9** for **high-precision thresholds**.
- Models trained on **one corpus** may fail on another → **external validation** (Enron).

**Suggested graphic**

- **Diagram**: two columns “FP: real mail → junk (bad)” vs. “FN: spam in inbox (less ideal)” — hand-drawn in slide tool.
- No notebook export required.

**Speaker notes**

- Plain language first: “wrongly blocked” vs. “missed spam”.
- Closing line: “Seminar uses **F1** to rank models; a **product** would stress **precision** and **thresholds**.”

---

## Slide 3 — Task & research questions (~45 s)

**Slide title (suggested)**  
Task & research questions

**Bullets (on slide)**

- **Task**: binary classification — **spam (1)** vs. **non-spam / ham (0)**.
- **Input**: 57 numeric features (Spambase); Enron mapped to same space via feature engineering.
- **Protocol**: train **only** on Spambase; Enron = **scoring only**.

**On-slide table**

| RQ | Question (short) |
| -- | ---------------- |
| **RQ1** | How well do 7 models perform on Spambase (CV + hold-out test)? |
| **RQ2** | Do they **generalize** to Enron mail **without** retraining? |
| **RQ3** | Can **threshold tuning** + **SVM+MLP ensemble** help on Enron? |

**Suggested graphic**

- Optional: tiny **two-box** diagram “Train: Spambase” → “Test: Spambase + Enron (no fit)”.

**Speaker notes**

- “No gradients, no hyperparameter search on Enron labels — ever.”

---

## Slide 4 — Dataset A: Spambase (~1:00)

**Slide title (suggested)**  
Dataset A: UCI Spambase

**Bullets (on slide)**

- **Source**: UCI Machine Learning Repository — Spambase (ID 94).
- **Size**: **4,601** emails × **57** features.
- **Features**: word/char **frequencies**, **capital run** statistics (no raw text in main table).
- **Why**: reproducible benchmark; same feature space for all models.

**On-slide table (class balance)**

| Class | Label | Count (≈) | Share |
| ----- | ----- | --------- | ----- |
| Ham | 0 | 2,788 | ~60.6% |
| Spam | 1 | 1,813 | **39.4%** |

**Suggested graphic**

- **Export**: Section **1** — bar chart or pie of **spam vs. ham** (`class_counts` plot).
- Optional second mini-plot: **correlation** with target (top features) — only if you want “EDA depth”.

**Speaker notes**

- “Slight imbalance → stratified splits + F1/precision/recall, not accuracy alone.”

---

## Slide 5 — Good ML practice: labels everywhere (~1:00–1:30)

**Slide title (suggested)**  
Stratified splits & CV — label distribution

**Bullets (on slide)**

- **Train / test**: **80% / 20%**, stratified, `random_state=42`.
- **Spam rate** in train and test ≈ **39.4%** (matches full data).
- **5-fold CV** runs **only on the training 80%** — the **~920 test emails never appear in any fold**.
- **Dummy baseline**: always predicts ham → **~60.6% accuracy** but **F1 = 0** → shows why accuracy alone misleads.

**On-slide table (CV folds)**

| Fold | Train spam % | Val spam % | \|Train\| | \|Val\| |
| ---- | ------------ | ---------- | --------- | ------- |
| 1–5 | **39.4%** | **39.4%** | 2944 | 736 |

**Suggested graphic**

- **Export**: Section **4** — *Class Distribution per K-Fold (Stratified 5-Fold CV)* **bar chart** (train vs. validation spam % per fold).

**Speaker notes**

- “This is the slide for ‘we followed good ML practice’ — same prevalence in every fold.”
- Mention **Section 12**: scaler fit on train only, no index overlap.

---

## Slide 6 — Dataset B: Enron (external) (~45 s)

**Slide title (suggested)**  
Dataset B: Enron (external validation only)

**Bullets (on slide)**

- **Source**: [MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data) — real mailbox-style mail.
- **Full corpus** (notebook): **33,716** rows; we score a **stratified sample of 1,000** (`random_state=42`).
- **Feature pipeline**: raw text → `extract_spambase_features` → **same 57 columns** as Spambase.
- **Strict rule**: **no training, no tuning on Enron** — only **apply** fitted models.

**On-slide table**

| Item | Value |
| ---- | ----- |
| Full Enron CSV | ~33.7k emails |
| Evaluation sample | **n = 1000** (stratified) |
| Training on Enron | **No** |

**Suggested graphic**

- **Export** (optional): Section **7** — small **screenshot** of Enron target distribution printout for the 1k sample.
- Or: **concept graphic** “Spambase (tabular) vs. Enron (text → features)” — two arrows into “same 57-D vector”.

**Speaker notes**

- “This is a **domain shift** test, not a second training set.”

---

## Slide 7 — Methodology pipeline (~1:30)

**Slide title (suggested)**  
Methodology overview

**Bullets (on slide)**

- **Models (7)**: Logistic Regression, Decision Tree, Random Forest, **XGBoost**, KNN, **SVM**, **MLP**.
- **Preprocessing**: stratified split; **StandardScaler fit on train only** for scaled models.
- **Evaluation**: CV (5-fold) on train; final metrics on **hold-out test**; then **Sections 8–13** (tuning, thresholds, features, ensemble, McNemar, leakage).
- **Metrics**: Acc, Precision, Recall, **F1**, ROC-AUC; **F1** for headline comparison; **precision** stressed for user-facing cost of false spam flags.

**Suggested graphic**

- **Custom flowchart** (main visual for this slide):

  `Load Spambase → EDA (Sec.1) → Split + scale (Sec.2) → Baseline dummy (Sec.3) → Train + CV (Sec.4–5) → [Tuning Sec.8 | PR/thresholds Sec.9 | Features Sec.10 | Ensemble Sec.10.5] → Score Enron (Sec.7+)`

- Keep **one row**; use **icons** (data, model, checkmark) if you like.

**Speaker notes**

- “CV estimates stability on **train**; the **test set** is the honest once-only check on Spambase.”

---

## Slide 8 — Results: Spambase (~1:00)

**Slide title (suggested)**  
Results: Spambase hold-out test

**Bullets (on slide)**

- **Baseline (DummyClassifier)**: Acc **≈ 0.606**, Precision **0**, Recall **0**, F1 **0** — predicts only majority class (ham).
- All **seven** models beat the baseline on **Acc**, **F1**, and show **strong precision and recall** on Spambase.
- **Headline ranking (F1)**: **XGBoost** and **Random Forest** top (~**0.93**); **MLP** ~**0.92**; weakest still **~0.88** (Decision Tree).
- **Precision on Spambase**: **all models ≥ ~0.88 test precision** — i.e. when they fire “spam”, they are usually correct in-domain (aligns with **precision-first** story for a filter).
- **ROC-AUC**: up to **~0.987** (XGBoost) — excellent ranking quality on Spambase.

**On-slide table — Spambase test (full, for handout or dense slide)**

| Model | Acc | Precision | Recall | F1 | ROC-AUC |
| ----- | --- | --------- | ------ | --- | ------- |
| Logistic Regression | 0.929 | 0.921 | 0.898 | 0.909 | 0.970 |
| Decision Tree | 0.911 | 0.883 | 0.893 | 0.888 | 0.908 |
| Random Forest | 0.946 | 0.951 | 0.909 | 0.930 | 0.983 |
| **XGBoost** | **0.946** | **0.936** | **0.926** | **0.931** | **0.987** |
| KNN | 0.908 | 0.886 | 0.879 | 0.882 | 0.951 |
| SVM | 0.927 | 0.928 | 0.884 | 0.906 | 0.967 |
| Neural Network (MLP) | 0.939 | 0.932 | 0.912 | 0.922 | 0.983 |

*Tip for live talk: show **chart** on slide; put **full table** in appendix PDF.*

**Suggested graphic**

- **Primary export**: Section **5.2** — *Model Comparison — Test Set Metrics* **grouped bar chart** (all five metrics) with **horizontal baseline** at Acc ≈ **0.606**.
- **Optional**: Section **5** ROC overlay — if you want one curve slide instead of table column.

**Speaker notes**

- “On Spambase the problem looks **easy** — high precision **and** high recall. The **stress test** is Enron.”
- Explicitly say: “**High precision across the board** here — good for a spam filter story before we talk domain shift.”

---
## Slide 9 — Results: Enron & domain shift (~1:30)

**Slide title (suggested)**  
Results: Enron (domain shift) + ensemble

**Bullets (on slide)**

- Metrics **drop** vs. Spambase — same models, different **vocabulary / senders / feature alignment**.
- **XGBoost**: still **high precision (~0.91)** on Enron but **very low recall (~0.23)** → misses most spam externally.
- **SVM** best single model on sample: Acc **0.679**, F1 **0.606** (threshold 0.5).
- **Soft-voting ensemble (SVM + MLP)** — XGBoost **excluded** — with **t ≈ 0.42** (chosen on Spambase test): Enron **Acc 0.689**, **F1 0.640** (best in comparison table).
- Additional calibrated weighted soft-voting variant (Sec. 10.6) was tested but **did not improve** external metrics.
- Caveat: threshold tuned on **Spambase test**, not a third holdout.

**On-slide table — Enron (t = 0.5, except ensemble)**

| Model | Acc | Precision | Recall | F1 |
| ----- | --- | --------- | ------ | --- |
| Logistic Regression | 0.656 | 0.769 | 0.464 | 0.578 |
| Decision Tree | 0.578 | 0.818 | 0.220 | 0.347 |
| Random Forest | 0.567 | 0.952 | 0.157 | 0.270 |
| XGBoost | 0.595 | 0.906 | 0.228 | 0.364 |
| KNN | 0.637 | 0.728 | 0.458 | 0.562 |
| **SVM** | **0.679** | 0.807 | 0.485 | **0.606** |
| MLP | 0.666 | 0.769 | 0.491 | 0.600 |
| **Ensemble SVM+MLP** | **0.689** | — | — | **0.640** |

*(Ensemble row: threshold **≈0.42**; precision/recall for “Best F1” scenario in notebook ≈ **0.78 / 0.54** — cite live or add footnote.)*

**Suggested graphic**

- **Primary**: Section **7** — **bar chart** “External validation — all models” **or** **1×7 confusion matrices** (readable if printed large).
- **Secondary**: Section **10.5** — **Enron** panel of ensemble (metrics vs. threshold or final Acc/F1 bars vs. singles).

**Speaker notes**

- “Rankings **change**: tree boosting was king on Spambase; **SVM/MLP** generalize better here.”
- “Ensemble averages **probabilities**; dropping XGBoost stopped dragging **recall** down.”

---

## Slide 10 — One “deep dive” tool (~45 s) — pick **one**

**Slide title (suggested)**  
Deep dive (choose one)

**Bullets (on slide)** — keep only your chosen option:

| Option | Title on slide | Bullets | Suggested graphic |
| ------ | --------------- | ------- | ------------------- |
| **A** | Thresholds & costs | Default vs. **best F1** vs. **high precision** thresholds (Sec. 9); same model, different FP/FN trade-off. | **Export Sec. 9**: metrics vs. threshold **or** PR curve for XGBoost / SVM / MLP. |
| **B** | Feature selection | Fewer features via ranking; per-model optimal *k*; Sec. **10.4** compares 57 vs. reduced on Enron. | **Export Sec. 10.2**: F1 vs. #features (3 lines). |
| **C** | McNemar tests | Spambase: XGB vs. SVM *p*≈0.017; others n.s. Enron: XGB vs. SVM/MLP ***p*<0.001**; SVM vs. MLP n.s. | **Export Sec. 12**: **disagreement heatmaps** (Spambase + Enron). |
| **D** | No data leakage | Four checks: disjoint indices, scaler = train means, CV excludes test, stratification OK. | **Export Sec. 13**: verification figure or table screenshot. |

**Speaker notes**

- “Pick **one** so we stay under 10 minutes.”

---

## Slide 11 — Key takeaways (~45 s)

**Slide title (suggested)**  
Takeaways

**Bullets (on slide)**

1. **Stratification**: **39.4%** spam in **train, test, and every CV fold** (2944 / 736).
2. **Spambase**: dummy **F1 = 0**; all models **F1 ≈ 0.88–0.93** and **high precision (~0.88–0.95)** on test.
3. **Enron**: clear **domain shift**; best single **SVM ~0.68 Acc / 0.61 F1**; **ensemble SVM+MLP ~0.69 / 0.64** with **no Enron training**.
4. **Metrics**: **F1** for **fair ranking**; **precision** first for **real filters**; **Section 9** for **high-precision** operating points.

**On-slide mini-table (optional)**

| Setting | Best headline result |
| ------- | --------------------- |
| Spambase test | XGBoost F1 **~0.93** |
| Enron (singles) | SVM F1 **~0.61** |
| Enron (ensemble) | SVM+MLP F1 **~0.64** |

**Suggested graphic**

- Optional: **one composite** — side-by-side **Spambase vs. Enron** bar (mean F1 drop) — build in slide tool from your numbers.

**Speaker notes**

- “Seminar = **F1** for ranking; product = **precision-first** + thresholds.”

---

## Slide 12 — Limitations, ethics, Q&A (~30 s + buffer)

**Slide title (suggested)**  
Limitations & outlook

**Bullets (on slide)**

- Enron features are **approximations** — not identical to original Spambase extraction → caps **transfer** performance.
- Ensemble **threshold** chosen on **Spambase test** — pragmatic; for publication-grade claims, a **nested** or **third** split would be stricter.
- **Sample**: Enron results on **n = 1000** — point estimates with sampling variance.
- **Ethics**: public research corpora; real products need **privacy**, **consent**, and **user control** over false positives.

**Suggested graphic**

- None required — or a simple **“?” + envelope** icon for Q&A.
- Optional: **thank you** + **email**.

**Speaker notes**

- Offer to open **Section 11 / 12** in the notebook live.

---

## Quick checklist before presenting

- [ ] **Slide 4**: Section **1** — class distribution figure  
- [ ] **Slide 5**: Section **4** — CV fold distribution bar chart  
- [ ] **Slide 7**: Pipeline flowchart (custom)  
- [ ] **Slide 8**: Section **5.2** — model comparison bar chart (+ baseline line)  
- [ ] **Slide 8 appendix**: full Spambase table (optional PDF page)  
- [ ] **Slide 9**: Section **7** Enron bars or confusion matrices + **10.5** ensemble  
- [ ] **Slide 10**: One figure from **9 / 10 / 11 / 12** per chosen option  
- [ ] Rehearse to **~9:30** spoken time  

---

## Mapping: slide ↔ notebook sections

| Slide | Topic | Primary notebook sections |
| ----- | ----- | ------------------------- |
| 4 | Spambase EDA | **1** |
| 5 | Split + CV balance | **2**, **4** |
| 6 | Enron protocol | **7** |
| 7 | Pipeline + metrics story | **2–5** |
| 8 | Spambase results | **5** |
| 9 | Enron + ensemble | **7**, **10.5** |
| 10 | Deep dive | **9** / **10** / **12** / **13** |
| 11–12 | Summary / limits | **14** (+ ethics narrative) |
