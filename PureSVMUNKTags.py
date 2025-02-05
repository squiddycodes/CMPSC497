import optuna
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =============================================================================
# 1. Load the Dataset and Convert to Pandas DataFrames
# =============================================================================
ds = load_dataset("batterydata/pos_tagging")
train_data = ds['train'][:1000]  # Subset for efficiency
test_data = ds['test']

df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# =============================================================================
# 2. Build Word-to-Index and Tag-to-Index Dictionaries
# =============================================================================
word_to_ix = {word: idx for idx, word in enumerate(set(word for sentence in df_train['words'] for word in sentence))}
tag_to_ix = {tag: idx for idx, tag in enumerate(set(tag for tags in df_train['labels'] for tag in tags))}

# Add a special "UNK" tag for unknown labels.
if "UNK" not in tag_to_ix:
    tag_to_ix["UNK"] = len(tag_to_ix)

# =============================================================================
# 3. Create a Fixed Dense Embedding for Each Word
# =============================================================================
EMBEDDING_DIM = 50
np.random.seed(1000)
embedding_dict = {word: np.random.randn(EMBEDDING_DIM) for word in word_to_ix.keys()}

# =============================================================================
# 4. Define a Feature Extraction Function
# =============================================================================
def get_features(word):
    return embedding_dict.get(word, np.zeros(EMBEDDING_DIM))

# =============================================================================
# 5. Flatten the DataFrames
# =============================================================================
def flatten_df(df):
    rows = []
    for _, row in df.iterrows():
        words = row['words']
        labels = row['labels']
        for word, label in zip(words, labels):
            rows.append({'word': word, 'label': label})
    return pd.DataFrame(rows)

train_flat = flatten_df(df_train)
test_flat = flatten_df(df_test)

# If you wish to avoid filtering out test rows, comment out the next line.
# test_flat = test_flat[test_flat['label'].isin(tag_to_ix.keys())]

# =============================================================================
# 6. Build the Feature Matrices and Label Vectors
# =============================================================================
X_train = np.stack(train_flat['word'].apply(get_features).values)
X_test = np.stack(test_flat['word'].apply(get_features).values)

def map_label(label):
    # Use get() to return tag_to_ix["UNK"] if label is not found.
    return tag_to_ix.get(label, tag_to_ix["UNK"])

y_train = train_flat['label'].apply(map_label).values
y_test = test_flat['label'].apply(map_label).values

# =============================================================================
# 7. Standardize the Features
# =============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 8. Define Optuna Optimization for SVM
# =============================================================================
'''
def objective(trial):
    """Objective function for Optuna to optimize SVM hyperparameters."""
    
    # Suggest values for hyperparameters
    C = trial.suggest_loguniform('C', 0.01, 10.0)
    degree = trial.suggest_int('degree', 2, 5)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    coef0 = trial.suggest_uniform('coef0', 0.0, 1.0)

    # Initialize SVM with suggested hyperparameters
    svc = SVC(kernel='poly', C=C, degree=degree, gamma=gamma, coef0=coef0, max_iter=10000, random_state=42)
    
    # Perform cross-validation to get a robust estimate of performance
    accuracy = cross_val_score(svc, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    
    return accuracy  # Optuna will maximize this value

# Create Optuna study and optimize
study = optuna.create_study(direction="maximize")  # Maximizing accuracy
study.optimize(objective, n_trials=50)  # Try 50 different sets of hyperparameters

# Get the best hyperparameters
best_params = study.best_params
print("\nBest Hyperparameters found:", best_params)
#Best Hyperparameters found: {'C': 6.782065779464058, 'degree': 5, 'gamma': 'auto', 'coef0': 0.17131013325827726}
'''

# =============================================================================
# 9. Train SVM with the Best Hyperparameters and Evaluate
# =============================================================================
best = {'C': 6.782065779464058, 'degree': 5, 'gamma': 'auto', 'coef0': 0.17131013325827726}
best_svc = SVC(kernel='poly', **best, max_iter=10000, random_state=42)
best_svc.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = best_svc.predict(X_train_scaled)
y_test_pred = best_svc.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nTrain Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
