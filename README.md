# 📘 Text Classification with Classical ML Algorithms

This project explores **text classification** using a variety of machine learning algorithms and text preprocessing techniques. The main goal was to **practice implementing classical ML models** for NLP tasks, and to compare their performance across different vectorization methods.

---

## 🎯 Objectives
- Practice **classical machine learning algorithms** for text classification:
  - **KMeans Clustering** (unsupervised)
  - **K-Nearest Neighbors (KNN)**
  - **Gaussian Naive Bayes (GaussianNB)**
  - **Decision Tree Classifier**
- Compare **different text preprocessing / vectorization methods**:
  - **Bag of Words (BoW)**
  - **TF-IDF (Term Frequency – Inverse Document Frequency)**
  - **Embedding Vectorization** (sentence/word embeddings)

---

## ⚙️ Project Workflow

1. **Data Preprocessing**
   - Extracted text samples with associated categories
   - Converted categorical labels to numeric IDs (`label_to_id` and `id_to_label` mappings)
   - Applied different feature extraction techniques:
     - **BoW**
     - **TF-IDF**
     - **Embeddings** (via [SentenceTransformers](https://www.sbert.net/))

2. **Model Training & Evaluation**
   - Implemented `train_and_test_*` functions for each algorithm
   - Created a benchmarking loop to train and evaluate each model on each feature set
   - Plotted confusion matrices for deeper inspection of performance
   - Collected accuracy results into a comparative table

3. **Visualization**
   - Confusion matrices heatmaps
   - Results summary table

---

## 📊 Results

The table below shows the accuracy scores for each algorithm across different vectorization methods:

| Vectorization | KNN   | KMeans | GaussianNB | DecisionTree |
|---------------|-------|--------|------------|--------------|
| **BoW**       | 0.535 | 0.500  | 0.775      | 0.625        |
| **TF-IDF**    | 0.790 | 0.565  | 0.790      | 0.615        |
| **Embeddings**| 0.855 | 0.760  | 0.865      | 0.625        |

📌 **Observations:**
- **Embeddings** consistently provided the best performance across algorithms.
- **GaussianNB** and **KNN** performed best overall with embeddings.
- **KMeans** (unsupervised) lagged behind supervised methods, but embeddings still improved its clustering quality.

