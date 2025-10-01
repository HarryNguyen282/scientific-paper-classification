import re
import math
from typing import List, Tuple, Dict, Any, Literal, Union
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import warnings

warnings.filterwarnings("ignore")

CACHE_DIR = './.cache'

ds = load_dataset("UniverseTBD/arxiv-abstracts-large")
samples = []
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']
for s in ds['train']:
    if len(s['categories'].split(' ')) != 1:
        continue
    current_category = s['categories'].split('.')[0]

    if current_category not in CATEGORIES_TO_SELECT:
        continue

    samples.append(s)

    if len(samples) >= 1000:
        break

preprocessed_samples = []

for s in samples:

    abstract = s['abstract']

    # remove \n characters
    abstract = abstract.strip().replace('\n', " ")

    # remove special characters and digits
    abstract = re.sub(r"[^a-zA-Z\s]", "", abstract)

    # convert to lower case
    abstract = abstract.lower()

    parts = s['categories'].split(' ')
    primary_category = parts[0].split('.')[0]

    preprocessed_samples.append({
        'abstract': abstract,
        'category': primary_category
    })

class EmbeddingVectorizer:
    def __init__(self,
                model_name: str = 'intfloat/multilingual-e5-base',
                normalize: bool = True     
           ):
        self.model = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
        self.normalize = normalize
    
    def _format_input(self, texts: List[str], mode: Literal['query', 'passage']) -> List[str]:
        if mode not in ['query', 'passage']:
            raise ValueError("Mode must be either 'query' or 'passage'")
        return [f"{mode}: {text.strip()}" for text in texts]
    
    def transform(self, texts: List[str], mode: Literal['query', 'passage'] = 'passage') -> List[List[float]]:
        if mode == "raw":
            inputs = texts
        else:
            inputs = self._format_input(texts, mode)

        embeddings = self.model.encode(inputs, normalize_embeddings=self.normalize)
        return embeddings.tolist()
    
    def transform_numpy(self, texts: List[str], mode: Literal['query', 'passage'] = 'query') -> np.array:
        return np.array(self.transform(texts, mode))

labels = set(s['category'] for s in preprocessed_samples)
sorted_labels = sorted(labels)

id_to_label = {i: label for i, label in enumerate(sorted_labels)}
label_to_id = {label: i for i, label in enumerate(sorted_labels)}

# prepare data for training and testing
X = [preprocessed_samples[i]['abstract'] for i in range(len(preprocessed_samples))]
y = [preprocessed_samples[i]['category'] for i in range(len(preprocessed_samples))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# preprocessing

# BOW vectorization
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# TF-IDF vectorization
tf_idf_vectorizer = TfidfVectorizer()
X_train_tfidf = tf_idf_vectorizer.fit_transform(X_train)
X_test_tfidf = tf_idf_vectorizer.transform(X_test)

# embedding vectorization
embedding_vectorizer = EmbeddingVectorizer()
X_train_embd = embedding_vectorizer.transform_numpy(X_train)
X_test_embd = embedding_vectorizer.transform_numpy(X_test)

# Convert all to numpy arrays
X_train_bow, X_test_bow = np.array(X_train_bow.toarray()), np.array(X_test_bow.toarray())
X_train_tfidf, X_test_tfidf = np.array(X_train_tfidf.toarray()), np.array(X_test_tfidf.toarray())

# kmeans clustering
# group the vectors into k clusters then label each cluster based on the most popular label in that cluster

def train_and_test_kmeans(X_train, X_test, y_train, y_test, n=5):
    kmeans = KMeans(n_clusters=n, random_state=42)
    clusters_id = kmeans.fit_predict(X_train)

    cluster_to_label = {}

    for cluster_id in set(clusters_id):
        labels_in_cluster = [y_train[i] for i in range(len(y_train)) if clusters_id[i] == cluster_id]
        most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
        cluster_to_label[cluster_id] = most_common_label

    test_clusters_id = kmeans.predict(X_test)
    y_pred = [cluster_to_label[cluster_id] for cluster_id in test_clusters_id]
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[id_to_label[i] for i in range(len(id_to_label))], output_dict=True)

    return y_pred, accuracy, report

# KNN classification
def train_and_test_knn(X_train, X_test, y_train, y_test, n=5):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[id_to_label[i] for i in range(len(id_to_label))], output_dict=True)
    return y_pred, accuracy, report

# Decision Tree Classification
def train_and_test_tree(X_train, X_test, y_train, y_test, max_depth=None):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    tree.fit(X_train_dense, y_train)
    y_pred = tree.predict(X_test_dense)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[id_to_label[i] for i in range(len(id_to_label))], output_dict=True)
    return y_pred, accuracy, report

# Naive Bayes Classification
def train_and_test_NB(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    
    # NB requires dense input
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    nb.fit(X_train_dense, y_train)
    y_pred = nb.predict(X_test_dense)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[id_to_label[i] for i in range(len(id_to_label))], output_dict=True)
    return y_pred, accuracy, report

def plot_confusion_matrix(y_true, y_pred, labels_list, figure_name="confusion_matrix", normalize="true", save_path=None, show=True):
    # compute cm with exact order provided
    cm = confusion_matrix(y_true, y_pred)

    # normalize cm
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    labels = np.unique(y_true)
    class_names = [labels_list[i] for i in range(len(labels_list)) if labels_list[i] in labels]

    annotation = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            raw = cm[i, j]
            norm = cm_normalized[i, j]
            annotation[i, j] = f"{raw}\n{norm:.2%}"

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=annotation, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(figure_name)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)  
    if show:
        plt.show()

    
def train():

    results = defaultdict(dict)
    
    feature_sets = {
    'BoW': {"X_train": X_train_bow, "X_test": X_test_bow},
    'TF-IDF': {"X_train": X_train_tfidf, "X_test": X_test_tfidf},
    'Embeddings': {"X_train": X_train_embd, "X_test": X_test_embd}
}    
    models = ["KNN", "KMeans", "GaussianNB", "DecisionTree"]

    for model in models:
        print(f"\nClassification using {model} model: ")
        print("-"*20)
        for method, training_set in feature_sets.items():
            print(f"Vectorization method: {method}:")
            X_train = training_set["X_train"]
            X_test = training_set["X_test"]

            if model == "KNN":
                y_pred, acc, cm = train_and_test_knn(X_train, X_test, y_train, y_test)
            elif model == "KMeans":
                y_pred, acc, cm = train_and_test_kmeans(X_train, X_test, y_train, y_test, n=len(np.unique(y_train)))
            elif model == "GaussianNB":
                y_pred, acc, cm = train_and_test_NB(X_train, X_test, y_train, y_test)
            elif model == "DecisionTree":
                y_pred, acc, cm = train_and_test_tree(X_train, X_test, y_train, y_test)

            results[model][method] = acc
            
            print(f"{method}: {acc:.4f}")

            plot_confusion_matrix(y_test, y_pred, sorted_labels, f"Confusion Matrix for {model} and {method}")
    
    return results

results = train()
print(pd.DataFrame(results))