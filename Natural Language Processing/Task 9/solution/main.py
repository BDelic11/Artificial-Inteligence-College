import os
import random
import nltk
import string
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

DATA_DIR = 'documents'

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(stemmed)

def load_documents():
    docs = []
    labels = []
    for category in os.listdir(DATA_DIR):
        cat_path = os.path.join(DATA_DIR, category)
        if os.path.isdir(cat_path):
            for filename in os.listdir(cat_path):
                with open(os.path.join(cat_path, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    docs.append(preprocess(content))
                    labels.append(category)
    return docs, labels

def split_data(docs, labels):
    combined = list(zip(docs, labels))
    random.seed(42)
    random.shuffle(combined)

    train_docs = []
    train_labels = []
    test_docs = []
    test_labels = []

    class_counts = {}
    for doc, label in combined:
        if class_counts.get(label, 0) < 1:
            test_docs.append(doc)
            test_labels.append(label)
            class_counts[label] = class_counts.get(label, 0) + 1
        else:
            train_docs.append(doc)
            train_labels.append(label)

    return train_docs, train_labels, test_docs, test_labels

def map_clusters_to_labels(cluster_labels, true_labels):
    mapping = defaultdict(list)
    for cluster, true in zip(cluster_labels, true_labels):
        mapping[cluster].append(true)

    final_map = {}
    for cluster, labels in mapping.items():
        most_common = Counter(labels).most_common(1)[0][0]
        final_map[cluster] = most_common
    return final_map

def main():
    docs, labels = load_documents()
    train_docs, train_labels, test_docs, test_labels = split_data(docs, labels)

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(train_docs)
    X_test = vectorizer.transform(test_docs)

    kmeans = KMeans(n_clusters=3, max_iter=300, random_state=42, n_init='auto')
    kmeans.fit(X_train)

    cluster_to_label = map_clusters_to_labels(kmeans.labels_, train_labels)

    test_preds = kmeans.predict(X_test)
    mapped_preds = [cluster_to_label.get(pred, 'unknown') for pred in test_preds]

    print("True labels:", test_labels)
    print("Predicted labels:", mapped_preds)
    print("\nClassification Report:")
    print(classification_report(test_labels, mapped_preds))

if __name__ == '__main__':
    main()
