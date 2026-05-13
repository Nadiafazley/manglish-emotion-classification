"""
NLP Model Implementation: Manglish Emotion Classification
Baseline TF-IDF pipeline for low-resource Malay-English code-mixed emotion detection.
This implementation aligns with objective 4.1 as a practical code-mixed baseline and with objective 4.2 by evaluating a small, noisy Malay emotion dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configuration
DATA_PATH = Path("Threads_Manglish_Clean_Labeled.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# TF-IDF Configuration
TFIDF_PARAMS = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95,
    'sublinear_tf': True,
    'stop_words': 'english'
}

print("="*80)
print("NLP MODEL IMPLEMENTATION: MANGLISH EMOTION CLASSIFICATION")
print("="*80)

# LOAD DATA
print("\n[1] LOADING DATA...")
try:
    df = pd.read_csv(DATA_PATH)
    print("[OK] Dataset loaded: {} samples, {} features".format(df.shape[0], df.shape[1]))
except FileNotFoundError:
    print("[ERROR] {} not found!".format(DATA_PATH))
    exit(1)

print("\nDataset Overview:")
print("  Shape: {}".format(df.shape))
print("  Columns: {}".format(list(df.columns)))

emotion_counts = df['emotion_label'].value_counts()
print("\nEmotion Distribution:")
print(emotion_counts)

print("\nClass Balance:")
for emotion, count in emotion_counts.items():
    percentage = (count / len(df)) * 100
    print("  {:12s}: {:4d} samples ({:5.1f}%)".format(emotion, count, percentage))

# PREPROCESSING & FEATURE ENGINEERING
print("\n[2] PREPROCESSING & FEATURE ENGINEERING...")

X = df['clean_text'].values
y = df['emotion_label'].values

print("[OK] Extracted text data: {} samples".format(len(X)))
print("[OK] Target emotion labels: {} unique classes".format(len(np.unique(y))))

print("\nApplying TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
X_tfidf = vectorizer.fit_transform(X)

print("[OK] TF-IDF feature matrix created: {}".format(X_tfidf.shape))
print("  Sparsity: {:.1f}%".format((1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100))

feature_names = vectorizer.get_feature_names_out()
print("\n[OK] Sample features (first 20):")
print("  {}".format(', '.join(feature_names[:20])))

# TRAIN/TEST SPLIT
print("\n[3] TRAIN/TEST SPLIT...")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

try:
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_tfidf, y_encoded, np.arange(X_tfidf.shape[0]),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    print("[OK] Stratified train/test split (80/20):")
except ValueError:
    print("[WARN] Stratification skipped (rare classes detected)")
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_tfidf, y_encoded, np.arange(X_tfidf.shape[0]),
        test_size=0.2,
        random_state=RANDOM_SEED
    )
    print("[OK] Random train/test split (80/20):")

y_train_labels = label_encoder.inverse_transform(y_train)
y_test_labels = label_encoder.inverse_transform(y_test)

print("  Training set: {} samples ({:.1f}%)".format(X_train.shape[0], (X_train.shape[0]/X_tfidf.shape[0]*100)))
print("  Test set: {} samples ({:.1f}%)".format(X_test.shape[0], (X_test.shape[0]/X_tfidf.shape[0]*100)))

print("\nClass distribution:")
print("  Training: {}".format(pd.Series(y_train_labels).value_counts().to_dict()))
print("  Test: {}".format(pd.Series(y_test_labels).value_counts().to_dict()))

# MODEL TRAINING
print("\n[4] TRAINING BASELINE MODELS...")

models = {
    'Naive Bayes': MultinomialNB(),
    'SVM (Linear)': LinearSVC(random_state=RANDOM_SEED, max_iter=1000),
    'Logistic Regression': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
}

trained_models = {}

for model_name, model in models.items():
    print("\n  Training {}...".format(model_name))
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    trained_models[model_name] = model
    print("  [OK] {} trained in {:.3f}s".format(model_name, elapsed))

# EVALUATION
print("\n[5] MODEL EVALUATION...")

results = {}

for model_name, model in trained_models.items():
    print("\n" + "-"*80)
    print("MODEL: {}".format(model_name))
    print("-"*80)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    class_report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        labels=np.arange(len(label_encoder.classes_)),
        digits=3,
        zero_division=0
    )
    
    cm = confusion_matrix(y_test, y_pred)
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'predictions': y_pred,
        'cm': cm,
        'class_report': class_report
    }
    
    print("\nOVERALL METRICS:")
    print("  Accuracy:  {:.4f}".format(accuracy))
    print("  Precision: {:.4f} (macro-averaged)".format(precision_macro))
    print("  Recall:    {:.4f} (macro-averaged)".format(recall_macro))
    print("  F1-Score:  {:.4f} (macro-averaged)".format(f1_macro))
    
    print("\nPER-CLASS PERFORMANCE:")
    print(class_report)

# MODEL COMPARISON
print("\n[7] MODEL COMPARISON...")
print("-"*80)

comparison_df = pd.DataFrame({
    model_name: {
        'Accuracy': results[model_name]['accuracy'],
        'Precision': results[model_name]['precision'],
        'Recall': results[model_name]['recall'],
        'F1-Score': results[model_name]['f1']
    }
    for model_name in results.keys()
}).T

print("\nCOMPARATIVE RESULTS TABLE:")
print(comparison_df.to_string())

best_model = comparison_df['F1-Score'].idxmax()
print("\n[OK] BEST MODEL (by F1-Score): {}".format(best_model))
print("  F1-Score: {:.4f}".format(results[best_model]['f1']))

# VISUALIZATIONS
print("\n[7] GENERATING VISUALIZATIONS...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices - Baseline Model Comparison', fontsize=16, fontweight='bold')

for idx, (model_name, ax) in enumerate(zip(results.keys(), axes)):
    cm = results[model_name]['cm']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                ax=ax, cbar_kws={'label': 'Normalized Count'})
    ax.set_title('{}\n(F1={:.3f})'.format(model_name, results[model_name]["f1"]), fontweight='bold')
    ax.set_ylabel('Actual Emotion')
    ax.set_xlabel('Predicted Emotion')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
print("[OK] Confusion matrices saved: {}".format(OUTPUT_DIR / 'confusion_matrices.png'))
plt.close()

# Metrics comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison_df.index))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, comparison_df[metric], width, label=metric, color=colors[i], alpha=0.8)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comparison_df.index, fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0, 1.05])
ax.grid(axis='y', alpha=0.3, linestyle='--')

for i, metric in enumerate(metrics):
    for j, v in enumerate(comparison_df[metric]):
        ax.text(j + i*width, v + 0.02, '{:.3f}'.format(v), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Metrics comparison saved: {}".format(OUTPUT_DIR / 'metrics_comparison.png'))
plt.close()

# Feature importance
print("\nExtracting feature importance...")
if 'Logistic Regression' in trained_models:
    model = trained_models['Logistic Regression']
    coefficients = model.coef_
    feature_importance = np.abs(coefficients).mean(axis=0)
    top_n = 20
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    top_features = feature_names[top_indices]
    top_weights = feature_importance[top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_weights, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance (Avg Abs Coefficient)', fontsize=11, fontweight='bold')
    ax.set_title('Top 20 Most Important Features (Logistic Regression)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print("[OK] Feature importance saved: {}".format(OUTPUT_DIR / 'feature_importance.png'))
    plt.close()

# ERROR ANALYSIS
print("\n[8] SAMPLE PREDICTIONS & ERROR ANALYSIS...")

best_model_obj = trained_models[best_model]
y_pred_best = best_model_obj.predict(X_test)

correct_mask = y_pred_best == y_test
incorrect_mask = ~correct_mask

print("\nPrediction Summary:")
print("  Correct: {} / {} ({:.1f}%)".format(correct_mask.sum(), len(y_test), correct_mask.sum()/len(y_test)*100))
print("  Incorrect: {} / {} ({:.1f}%)".format(incorrect_mask.sum(), len(y_test), incorrect_mask.sum()/len(y_test)*100))

if incorrect_mask.sum() > 0:
    print("\nSample Misclassifications (first 5, prefer Manglish examples):")
    incorrect_indices = np.where(incorrect_mask)[0]
    manglish_indices = []

    for test_idx in incorrect_indices:
        original_idx = indices_test[test_idx]
        if df.iloc[original_idx].get('code_mixed', False) is True:
            manglish_indices.append(test_idx)

    if len(manglish_indices) >= 5:
        chosen_indices = manglish_indices[:5]
    else:
        chosen_indices = incorrect_indices[:5]

    for idx, test_idx in enumerate(chosen_indices):
        original_idx = indices_test[test_idx]
        text = df.iloc[original_idx]['text'][:80]
        actual = label_encoder.inverse_transform([y_test[test_idx]])[0]
        predicted = label_encoder.inverse_transform([y_pred_best[test_idx]])[0]
        code_mixed_flag = df.iloc[original_idx].get('code_mixed', False)
        
        print("\n  {}. Text: \"{}...\"".format(idx+1, text))
        print("     Code-mixed: {}".format(code_mixed_flag))
        print("     Actual: {} -> Predicted: {}".format(actual, predicted))

# SUMMARY
print("\n" + "="*80)
print("EXPERIMENT SUMMARY")
print("="*80)

summary = """
DATASET STATISTICS
  * Total samples: {}
  * Train samples: {} (80%)
  * Test samples: {} (20%)
  * Emotion classes: {}
  * Classes: {}

FEATURE ENGINEERING
  * Method: TF-IDF Vectorization
  * Max features: {}
  * N-gram range: {}
  * Feature dimension: {}

MODELS TRAINED
  * Naive Bayes
  * SVM (Linear)
  * Logistic Regression

BEST MODEL: {}
  * Accuracy: {:.4f}
  * Precision: {:.4f}
  * Recall: {:.4f}
  * F1-Score: {:.4f}

REPRODUCIBILITY
  * Random seed: {}
  * Test size: 0.20
  * All results saved to: {}

OUTPUT FILES
  * confusion_matrices.png
  * metrics_comparison.png
  * feature_importance.png
""".format(
    df.shape[0], X_train.shape[0], X_test.shape[0],
    len(label_encoder.classes_), ', '.join(label_encoder.classes_),
    TFIDF_PARAMS['max_features'], TFIDF_PARAMS['ngram_range'], X_tfidf.shape[1],
    best_model,
    results[best_model]['accuracy'], results[best_model]['precision'],
    results[best_model]['recall'], results[best_model]['f1'],
    RANDOM_SEED, OUTPUT_DIR
)

print(summary)

# Save summary
summary_path = OUTPUT_DIR / 'summary_report.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)
    f.write("\n\nDETAILED RESULTS:\n")
    f.write("="*80 + "\n")
    for model_name in results.keys():
        f.write("\n{}:\n".format(model_name))
        f.write(results[model_name]['class_report'])
        f.write("\n" + "="*80 + "\n")

print("[OK] Summary report saved: {}".format(summary_path))

print("\n" + "="*80)
print("[OK] PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("="*80 + "\n")
