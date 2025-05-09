import os
from pathlib import Path
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Paths
base_dir = Path(__file__).parent / "documents"
categories = ["plants", "animals", "astronomy"]

# 2. Load documents
def load_documents():
    category_docs = defaultdict(list)
    for category in categories:
        folder = base_dir / category
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".txt"):
                filepath = folder / filename
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    category_docs[category].append(content)
    return category_docs

category_docs = load_documents()

# 3. Split: 3 train + 1 test per class
train_docs, test_docs = [], []
train_labels, test_labels = [], []

for category in categories:
    docs = category_docs[category]
    if len(docs) < 3:
        raise ValueError(f"Expected at least 3 documents in category '{category}'")
    train_docs.extend(docs[:2])
    test_docs.append(docs[2])
    train_labels.extend([category] * 2)
    test_labels.append(category)

# 4. TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_df=0.9, min_df=1)

X_train = vectorizer.fit_transform(train_docs)
X_test = vectorizer.transform(test_docs)

# 5. KMeans clustering
k = len(categories)
kmeans = KMeans(n_clusters=k, max_iter=300, random_state=42, n_init=10)
kmeans.fit(X_train)

# 6. Predict test documents
test_predictions = kmeans.predict(X_test)
train_predictions = kmeans.predict(X_train)

# 7. Map cluster → label using majority from training
cluster_to_label = {}
for cluster_id in range(k):
    label_counts = Counter(
        train_labels[i]
        for i in range(len(train_labels))
        if train_predictions[i] == cluster_id
    )
    if label_counts:
        cluster_to_label[cluster_id] = label_counts.most_common(1)[0][0]

# 8. Final predicted labels
predicted_labels = [cluster_to_label.get(cluster_id, "unknown") for cluster_id in test_predictions]

# 9. Output
print("=== Predicted label vs true label ===")
for i, (pred, true) in enumerate(zip(predicted_labels, test_labels)):
    print(f"Document {i+1}: predicted '{pred}', true '{true}'")

accuracy = accuracy_score(test_labels, predicted_labels)
print(f"\n✅ Accuracy: {accuracy:.2f}")
