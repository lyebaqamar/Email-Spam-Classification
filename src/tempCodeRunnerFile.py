import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Debug check
print("Loaded dataset with shape:", df.shape)

# 2. Inspect dataset
print("Columns:", df.columns)
print("Missing values:\n", df.isnull().sum())

# Keep only relevant columns (dataset has extra unnamed ones)
df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# 3. Preprocess
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].astype(str)

# Handle missing values
df = df.dropna()

print("Class distribution:\n", df['label'].value_counts())

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 5. Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 7. Predictions
y_pred = model.predict(X_test_tfidf)

# 8. Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = f"""
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}

Classification Report:
{classification_report(y_test, y_pred)}
"""

print(results)

# Save results
with open("outputs/evaluation_results.txt", "w") as f:
    f.write(results)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Debug check
print("Loaded dataset with shape:", df.shape)

# 2. Inspect dataset
print("Columns:", df.columns)
print("Missing values:\n", df.isnull().sum())

# Keep only relevant columns (dataset has extra unnamed ones)
df = df[['v1', 'v2']]
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# 3. Preprocess
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].astype(str)

# Handle missing values
df = df.dropna()

print("Class distribution:\n", df['label'].value_counts())

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 5. Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 7. Predictions
y_pred = model.predict(X_test_tfidf)

# 8. Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = f"""
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}

Classification Report:
{classification_report(y_test, y_pred)}
"""

print(results)

# Save results
with open("outputs/evaluation_results.txt", "w") as f:
    f.write(results)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# 9. Extra testing with new samples
sample = ["Congratulations! You won a prize", "Hey, are we meeting tomorrow?"]
sample_tfidf = vectorizer.transform(sample)
predictions = model.predict(sample_tfidf)

for msg, pred in zip(sample, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"Message: {msg} --> Prediction: {label}")
