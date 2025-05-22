import os
from pathlib import Path
from collections import defaultdict, Counter
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

base_dir = Path("documents")
categories = ["plants", "animals", "astronomy"]

# 1. LOAD DOCUMENTS
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

# 2. SPLIT INTO TRAIN/TEST
train_docs, test_docs, train_labels, test_labels = [], [], [], []
for category in categories:
    docs = category_docs[category]
    train_docs.extend(docs[:2])       
    train_labels.extend([category] * 2)
    test_docs.append(docs[2])         
    test_labels.append(category)

# 3. TRAIN WORDPIECE TOKENIZER
all_docs = train_docs + test_docs
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordPieceTrainer(vocab_size=3000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train_from_iterator(all_docs, trainer=trainer)

# 4. TOKENIZE DOCUMENTS USING SUBWORDS
def tokenize_texts(tokenizer, texts):
    return [" ".join(tokenizer.encode(text).tokens) for text in texts]

train_tokenized = tokenize_texts(tokenizer, train_docs)
test_tokenized = tokenize_texts(tokenizer, test_docs)

# 5. TF-IDF VECTORIZATION OVER SUBWORDS
vectorizer = TfidfVectorizer(max_df=0.85, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_tokenized)
X_test = vectorizer.transform(test_tokenized)

# 6. ENCODE LABELS
le = LabelEncoder()
y_train = le.fit_transform(train_labels)
y_test = le.transform(test_labels)

# 7. KMEANS TRAINING
kmeans = KMeans(n_clusters=3, random_state=42, max_iter=300, n_init="auto")
kmeans.fit(X_train)

# 8. MAP CLUSTERS TO LABELS
cluster_to_label = {}
for cluster_id in range(3):
    indices = [i for i, c in enumerate(kmeans.labels_) if c == cluster_id]
    label_counts = Counter([train_labels[i] for i in indices])
    if label_counts:
        cluster_to_label[cluster_id] = label_counts.most_common(1)[0][0]

# 9. PREDICT
predicted_clusters = kmeans.predict(X_test)
predicted_labels = [cluster_to_label.get(cluster, "unknown") for cluster in predicted_clusters]

# 10. OUTPUT RESULTS
print("=== Predicted label vs true label ===")
for i, (pred, true) in enumerate(zip(predicted_labels, test_labels)):
    print(f"Document {i+1}: predicted '{pred}', true '{true}'")

# 11. ACCURACY
acc = accuracy_score(test_labels, predicted_labels)
print(f"\n✅ Accuracy: {acc:.2f}")


# Rezultati:
# === Predicted label vs true label ===
# Document 1: predicted 'plants', true 'plants'
# Document 2: predicted 'animals', true 'animals'
# Document 3: predicted 'animals', true 'astronomy'

# ✅ Accuracy: 0.67
