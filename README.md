# CS6365
# Fitness Tracker Recommendation System

# Overview:
A machine learning-based workout recommendation system that uses K-Means clustering and cosine similarity to provide personalized workout suggestions based on user behavior and preferences. Overall, this project built a workout recommendation system that works best when using clustering methods like K-Means and GMM, which group similar workouts well and produce strong recall. The system tends to retrieve most of the workouts users actually do, but it also recommends many they do not choose, which lowers precision. Future work should focus on improving precision by tightening similarity measures and testing new clustering settings so the system can give more accurate and focused recommendations.

The system combines multiple approaches:

* content-based filtering using cosine similarity
* clustering methods such as **K-Means** and **Gaussian Mixture Models (GMM)**
* collaborative filtering using ALS

The goal is to model user workout preferences and recommend workouts that align with their past activity patterns.

---

## Key Insights

The system performs well at identifying relevant workouts:

* **high recall** → successfully retrieves workouts users are likely to perform
* **lower precision** → also recommends some irrelevant workouts

This indicates that clustering-based methods (K-Means, GMM) are effective at capturing user behavior patterns, but future improvements should focus on making recommendations more precise and targeted.

---

## Features

* Data cleaning and normalization of workout datasets
* Feature engineering with numerical scaling and categorical encoding
* Dimensionality reduction using PCA
* User embedding generation from workout history
* Recommendation systems:

  * cosine similarity–based (content filtering)
  * K-Means clustering
  * Gaussian Mixture Models (soft clustering)
  * ALS collaborative filtering
* Evaluation metrics:

  * Precision@K
  * Recall@K
  * NDCG@K
  * ROC-AUC, F1, Accuracy

---

## Repository Structure

```
.
├── data_frame.py              # data cleaning + structuring
├── data_preprocessing.py      # feature engineering
├── data_processing.py        # PCA + embeddings
├── gmm.py                    # GMM-based recommender
├── tester.py                 # ALS collaborative filtering
├── supervised_model.py       # supervised learning models
├── main.py                   # supervised model entry point
├── config.py                 # configuration settings
├── metric_eval.config.py     # evaluation visualization
├── INSTRUCTIONS.md           # detailed workflow guide
```

---

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn implicit
```

---

## How to Run

### 1. Data Processing

```bash
python data_frame.py
python data_preprocessing.py
python data_processing.py
```

### 2. GMM Recommendation System

```bash
python gmm.py
```

### 3. Collaborative Filtering (ALS)

```bash
python tester.py
```

### 4. Supervised Learning Model

```bash
python main.py
```

### 5. Metrics Visualization

```bash
python metric_eval.config.py
```

---

## Methodology

### 1. Data Preparation

* cleaned dataset and standardized column formats
* created user workout session logs

### 2. Feature Engineering

* scaled numerical features
* encoded categorical variables
* handled missing values

### 3. Representation Learning

* reduced dimensionality with PCA
* built user embeddings by averaging workout vectors

### 4. Recommendation Models

* **Content-based filtering:** cosine similarity
* **Clustering:** K-Means and GMM
* **Collaborative filtering:** ALS

### 5. Evaluation

* ranking metrics for recommendation quality
* classification metrics for prediction performance

---

## Results

* clustering methods (K-Means, GMM) provided **strong recall**
* cosine similarity helped refine recommendations
* GMM improved flexibility by allowing soft cluster membership
* overall system retrieves relevant workouts but needs **better precision control**

---

## Future Work

* improve precision by tuning similarity thresholds
* implement Gaussian Mixture Model enhancements
* experiment with hybrid models (content + collaborative)
* incorporate temporal patterns in user behavior
* optimize ranking using learning-to-rank methods

---

## Technologies Used

* Python
* NumPy, Pandas
* Scikit-learn
* Implicit (ALS)
* Matplotlib

---

## Author
* Vania Munjar
* GTID: 903871348
* Georgia Tech – CS 6365

---
