import os
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# LOCAL PATH SETUP
base_dir = Path("documents")
categories = ["plants", "animals", "astronomy"]

# LOAD DOCUMENTS
def load_documents(base_dir, categories):
    docs_by_category = defaultdict(list)
    for category in categories:
        folder = base_dir / category
        for filename in sorted(os.listdir(folder)):
            filepath = folder / filename
            with open(filepath, "r", encoding="utf-8") as f:
                docs_by_category[category].append(f.read())
    return docs_by_category

category_docs = load_documents(base_dir, categories)

# SPLIT INTO TRAIN/TEST
train_docs, test_docs, train_labels, test_labels = [], [], [], []
for category in categories:
    docs = category_docs[category]
    train_docs.extend(docs[:2])
    train_labels.extend([category] * 2)
    test_docs.append(docs[2])
    test_labels.append(category)

# TF-IDF VECTORIZATION
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.85
)
X_train = vectorizer.fit_transform(train_docs)
X_test = vectorizer.transform(test_docs)

# LABEL ENCODING
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
test_labels_encoded = le.transform(test_labels)

# KMEANS TRAINING
kmeans = KMeans(n_clusters=3, random_state=42, max_iter=300, n_init="auto")
kmeans.fit(X_train)

# PREDICT
predicted_clusters = kmeans.predict(X_test)

# MAP CLUSTERS TO LABELS
cluster_to_label = {}
for cluster_id in range(3):
    indices = [i for i, c in enumerate(kmeans.labels_) if c == cluster_id]
    label_counts = Counter([train_labels[i] for i in indices])
    if label_counts:
        cluster_to_label[cluster_id] = label_counts.most_common(1)[0][0]

# FINAL PREDICTIONS
predicted_labels = [cluster_to_label.get(cluster, "unknown") for cluster in predicted_clusters]

# RESULTS
print("=== Predicted label vs true label ===")
for i, (pred, true) in enumerate(zip(predicted_labels, test_labels)):
    print(f"Document {i+1}: predicted '{pred}', true '{true}'")

# ACCURACY
acc = accuracy_score(test_labels, predicted_labels)
print(f"\n✅ Accuracy: {acc:.2f}")

# AFTER RUNNING TESTED 
# === Predicted label vs true label ===
# Document 1: predicted 'plants', true 'plants'
# Document 2: predicted 'animals', true 'animals'
# Document 3: predicted 'animals', true 'astronomy'

# ✅ Accuracy: 0.67
