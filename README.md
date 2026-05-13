# Manglish Emotion Classification - NLP Model Implementation

## Project Overview
This project implements a **supervised text classification model** to detect emotions in Manglish (code-mixed Malay-English) social media text. The system uses TF-IDF feature extraction combined with multiple baseline classifiers (Naive Bayes, SVM, Logistic Regression).

**Status**: ✅ Working prototype with full evaluation

---

## Research Alignment
- Objective 4.1: This implementation establishes a practical baseline for Malay-English code-mixed emotion detection using TF-IDF on Manglish text. It captures linguistic diversity and pragmatic emotion indicators from the dataset while remaining interpretable and suitable for low-resource settings.
- Objective 4.2: The prototype is evaluated on a small, noisy Malay emotion dataset and explicitly reports the effects of class imbalance and low-resource constraints. This helps assess the method's flexibility and limitations in a realistic, resource-constrained case.
- Scope note: The current model focuses on a baseline TF-IDF approach rather than a full multilingual embedding system. That makes the work reproducible, fast, and well-aligned with the dataset available, while leaving more advanced cross-lingual embedding and translation-based methods for future work.

---

## Dataset
- **File**: `Threads_Manglish_Clean_Labeled.csv`
- **Samples**: 118 posts
- **Classes**: 6 emotion categories (anger, fear, joy, love, neutral, sadness)
- **Columns**: 
  - `text` - Original text
  - `clean_text` - Preprocessed text (used for modeling)
  - `emotion_label` - Ground truth emotion label (target variable)
  - Additional metadata: code_mixing, manglish_score, secondary emotions

---

## Model Architecture

### Pipeline Flow
```
Raw Text
    ↓
Preprocessing (TF-IDF Vectorization)
    ↓
Feature Matrix (118 samples × 296 features)
    ↓
Train/Test Split (80/20)
    ↓
Baseline Model Training
├─ Naive Bayes
├─ SVM (Linear)
└─ Logistic Regression
    ↓
Evaluation & Comparison
```

### Feature Representation
- **Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features**: 5,000 most important tokens
- **N-grams**: Unigrams + Bigrams (1-2)
- **Stop Words**: Removed (English)
- **Feature Dimension**: 296 features
- **Sparsity**: 96.8%

**Why TF-IDF?** 
- Interpretable: easily identify emotion-bearing words
- Efficient: works well with classical ML models
- Proven baseline: standard for text classification
- Computationally fast for rapid experimentation

**Relevance to Objectives**
- Supports objective 4.1 by modeling code-switched Manglish text directly and highlighting emotion-bearing lexical signals.
- Supports objective 4.2 by evaluating performance on a low-resource dataset and making the dataset’s imbalance and noise visible in the results.
- Serves as a strong baseline for later comparison with multilingual embeddings or translation-based methods.

### Models
1. **Naive Bayes** (MultinomialNB)
   - Fast probabilistic classifier
   - Strong baseline for text data
   - Training time: ~0.004s

2. **SVM (Linear)** ⭐ **BEST MODEL**
   - High-performance discriminative classifier
   - Handles high-dimensional sparse data well
   - F1-Score: **0.3800** (macro-averaged)
   - Accuracy: **83.33%**

3. **Logistic Regression**
   - Interpretable coefficients
   - Probabilistic predictions
   - Training time: ~0.031s

---

## Results

### Best Model: SVM (Linear)
| Metric | Score |
|--------|-------|
| Accuracy | 0.8333 |
| Precision (macro) | 0.3636 |
| Recall (macro) | 0.4000 |
| F1-Score (macro) | 0.3800 |

### Per-Class Performance (SVM)
```
              precision    recall  f1-score   support
       anger      0.000     0.000     0.000         2
        fear      0.000     0.000     0.000         1
         joy      1.000     1.000     1.000         2
        love      0.000     0.000     0.000         1
     neutral      0.818     1.000     0.900        18
     sadness      0.000     0.000     0.000         0
```

### Visualizations Generated
- **confusion_matrices.png** - Side-by-side confusion matrices for all 3 models
- **metrics_comparison.png** - Bar chart comparing Accuracy, Precision, Recall, F1
- **feature_importance.png** - Top 20 most important TF-IDF features

---

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Or manually:**
```bash
pip install pandas scikit-learn numpy matplotlib seaborn
```

**Python Version**: 3.10+

### 2. Project Structure
```
NLP ASSIGNMENT 3B/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── nlp_model.py                       # Main implementation
├── Threads_Manglish_Clean_Labeled.csv # Dataset
└── results/                           # Output folder
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    ├── feature_importance.png
    └── summary_report.txt
```

---

## Usage

### Run the Complete Pipeline
```bash
python nlp_model.py
```

**Output:**
- Console: Detailed training logs, metrics for all models
- `results/` folder: Visualizations and summary report

### Expected Execution Time
- Data loading: <1s
- TF-IDF vectorization: <1s
- Model training: ~0.05s (all 3 models)
- Evaluation & visualization: ~2-3s
- **Total**: ~5 seconds

---

## Key Findings

### Model Comparison
- **SVM** performs best (F1: 0.38, Accuracy: 83.33%)
- **Naive Bayes** & **Logistic Regression** tied (F1: 0.17, Accuracy: 75%)
- Class imbalance issue: 68.6% neutral samples bias models toward "neutral" class

### Error Analysis
Most misclassifications are **anger** and **fear** → **neutral** (underrepresented classes)

**Example misclassifications:**
```
Actual: anger → Predicted: neutral
"orang marah i nangis, i marah orang pun i nangis eh malas la i ...."

Actual: fear → Predicted: neutral  
"Orang cakap mainan tidur. Tapi adik saya beberapa bulan sebelum meninggal..."
```

### Important Features (by TF-IDF weight)
Top emotion indicators:
- **joy**: "love", "comel" (cute), "bagus" (good)
- **anger**: "marah" (angry), "fckin"
- **neutral**: "bagels", "astro", "awak" (you)
- **love**: "sayang" (love), "doa" (prayer), "ibu" (mother)

---

## Reproducibility

### Fixed Parameters
```python
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.80
TF-IDF Parameters:
  - max_features: 5000
  - ngram_range: (1, 2)
  - min_df: 2
  - max_df: 0.95
  - sublinear_tf: True
  - stop_words: 'english'
```

### To Reproduce
1. Run: `python nlp_model.py`
2. Same results guaranteed (random seed fixed at 42)
3. Dataset: Use provided `Threads_Manglish_Clean_Labeled.csv`

---

## Limitations & Future Work

### Current Limitations
1. **Small dataset** (118 samples) - especially for 6 classes
2. **Class imbalance** - neutral dominates (68.6%)
3. **Rare classes** - sadness, fear have only 1 sample each
4. **TF-IDF loses context** - doesn't capture word order or semantics
5. **No Malay stop words** - applied only English stop words

### Future Improvements
1. **BERT embeddings** - Contextual word representations
2. **Class balancing** - SMOTE, class weights, or data augmentation
3. **More data collection** - Increase samples, especially for minority emotions
4. **Malay-specific processing** - Language-specific stop words, stemming
5. **Ensemble methods** - Combine multiple models
6. **Hyperparameter optimization** - GridSearch, RandomSearch, Bayesian optimization
7. **Cross-validation** - K-fold CV for more robust evaluation
8. **Neural networks** - LSTM, CNN for sequential dependencies

---

## Evaluation Metrics Explanation

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness - % correct predictions |
| **Precision** | TP/(TP+FP) | Specificity - when model predicts emotion X, how often is it right? |
| **Recall** | TP/(TP+FN) | Sensitivity - of all true emotion X instances, how many does model find? |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean - balances precision & recall |
| **Macro-Average** | Mean of per-class scores | Treats all classes equally (important for imbalanced data) |
| **Weighted-Average** | Weighted by class support | Accounts for class distribution |

**Why macro F1-Score?** Better for imbalanced datasets - doesn't favor dominant "neutral" class

---

## Citation & References

1. **Emotion Detection**
   - Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word-emotion association lexicon. *Computational Intelligence*, 29(3), 436-465.
   - Rosenthal, S., et al. (2017). SemEval-2017 Task 4: Sentiment Analysis in Twitter. In Proceedings of SemEval.

2. **Code-Mixed Text**
   - Jauhiainen, H., Lindén, K., & Jauhiainen, T. (2015). Language identification of extremely similar languages and dialects. In EACL 2015.

3. **Feature Representation**
   - Leskovec, I., Rajaraman, A., & Ullman, J. D. (2014). *Mining of Massive Datasets* (2nd ed.). Cambridge University Press.

---

## Files Description

| File | Purpose |
|------|---------|
| `README.md` | Project overview, setup, usage (this file) |
| `requirements.txt` | Python package dependencies |
| `nlp_model.py` | Complete working implementation |
| `Threads_Manglish_Clean_Labeled.csv` | Dataset |
| `results/` | Outputs folder |

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'sklearn'`
**Solution**: 
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Issue: `FileNotFoundError: Threads_Manglish_Clean_Labeled.csv not found`
**Solution**: Ensure CSV file is in the same directory as `nlp_model.py`

### Issue: Results differ from expected
**Solution**: Verify `RANDOM_SEED = 42` is set in the code (already fixed)

---

## Contact & Support
- **Dataset File**: Included in submission
- **Code**: Fully commented and documented
- **Outputs**: All visualizations saved automatically

---

**Submission Date**: May 2026  
