# INSTRUCTIONS.md

## Project

**Fitness Tracker Recommendation System**
CS 6365 – Intelligent Enterprise Computing
Georgia Institute of Technology

---

## Overview

This repository contains the implementation of a **fitness workout recommendation system**. The project focuses on transforming raw workout data into structured representations, building user behavior profiles, and developing multiple recommendation strategies including content-based filtering, clustering, Gaussian Mixture Models (GMM), and collaborative filtering.

The goal of this file is to help an **AI coding assistant (LLM)** or any new contributor quickly understand how to **set up, run, test, and extend the project end-to-end**.

---

## Repository Purpose

This project uses a fitness tracker dataset to build a recommendation system that suggests workouts based on both **workout characteristics** and **user behavior patterns**.

At the current stage, the repository supports:

* loading and cleaning the raw workout dataset
* standardizing column handling across dataset variations
* generating structured workout records
* creating user workout session logs
* preprocessing features for downstream machine learning models
* reducing dimensionality using PCA (optional)
* constructing user embeddings from workout features
* generating recommendations using clustering, GMM, and collaborative filtering
* evaluating ranking and classification performance

---

## Main Files

### `config.py`

Central configuration file for dataset paths and modeling parameters.

Responsibilities:

* defines dataset path
* specifies labeling logic for supervised learning
* configures PCA usage and reproducibility

Key settings include:

* dataset path and preprocessing options
* positive label definition using mood or calorie threshold 

---

### `data_frame.py`

Loads and cleans the dataset, standardizes column names, selects key workout features, creates duration buckets, generates unique workout IDs, and builds user session logs.

Outputs:

* workout catalog (unique workouts)
* user workout interaction logs

---

### `data_preprocessing.py`

Handles the preprocessing pipeline for machine learning.

Responsibilities:

* load dataset and resolve column inconsistencies using aliases
* derive labels (e.g., workout success)
* separate numeric and categorical features
* handle missing values
* apply StandardScaler to numeric features
* apply OneHotEncoder to categorical features

Outputs:

* processed feature matrix `X`
* label vector `y`
* fitted preprocessing pipeline

---

### `data_processing.py`

Applies feature transformations and builds user representations.

Responsibilities:

* apply PCA for dimensionality reduction (if enabled)
* retain maximum variance in features
* generate user embeddings by averaging workout feature vectors

Outputs:

* PCA-transformed feature matrix
* user profile embeddings

---

### `gmm.py`

Implements **Gaussian Mixture Model–based recommendation system**.

Responsibilities:

* preprocess workout features (scaling + encoding)
* construct workout vectors and user embeddings
* split user logs into train/test sets
* fit GMM for soft clustering of user profiles
* compute cluster probabilities per user
* generate recommendations using:

  * cosine similarity
  * cluster-based boosting
* evaluate model using:

  * Precision@K
  * Recall@K
  * NDCG@K
  * classification metrics (ROC-AUC, F1, etc.)

Also includes:

* `recommend_gmm(uid, K)` → generates Top-K recommendations
* `evaluate_ranking(k)` → ranking evaluation
* `evaluate_classification()` → classification evaluation

Implements full pipeline from embeddings → clustering → ranking 

---

### `supervised_model.py`

Implements supervised learning models for predicting workout outcomes or user preferences based on processed features.

---

### `main.py`

Entry point for supervised learning pipeline.

Responsibilities:

* runs model training using configured dataset
* outputs evaluation metrics summary

Execution:

```bash
python main.py
```

Uses configuration from `config.py` and model pipeline from `supervised_model.py` 

---

### `tester.py`

Implements collaborative filtering using ALS and evaluates recommendation quality by generating Top-K recommendations.

---

### `metric_eval.config.py`

Handles visualization of evaluation metrics.

Responsibilities:

* loads computed metrics
* generates plots for:

  * Precision@K
  * Recall@K
  * NDCG@K
* saves visualization to file

Execution produces:

* `metrics_visualization.png` for performance comparison 

---

## Setup Instructions

### Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn implicit
```

### Clone Repository

```bash
git clone https://github.com/vania2004690/CS6365.git
cd CS6365
```

---

## End-to-End Workflow

Follow this pipeline in order:

---

### 1. Data Structuring

```bash
python data_frame.py
```

* Cleans dataset
* Creates workout catalog
* Builds user interaction logs

---

### 2. Feature Preprocessing

```bash
python data_preprocessing.py
```

* Encodes categorical features
* Scales numerical features
* Produces model-ready dataset

---

### 3. Feature Transformation + Embeddings

```bash
python data_processing.py
```

* Applies PCA (if enabled)
* Generates user embeddings

---

### 4. GMM-Based Recommendations

```bash
python gmm.py
```

* Builds user profiles
* Fits Gaussian Mixture Model
* Generates recommendations
* Outputs ranking + classification metrics

---

### 5. Collaborative Filtering (ALS)

```bash
python tester.py
```

* Trains ALS model
* Produces Top-K recommendations

---

### 6. Supervised Learning Pipeline

```bash
python main.py
```

* Trains supervised model
* Outputs classification performance

---

### 7. Metrics Visualization

```bash
python metric_eval.config.py
```

* Plots Precision@K, Recall@K, NDCG@K
* Saves visualization

---

## Evaluation

Evaluation includes both **ranking** and **classification** perspectives:

### Ranking Metrics

* Precision@K
* Recall@K
* NDCG@K

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Expected Outputs

* cleaned and structured dataset
* processed feature matrix
* reduced-dimension feature space (PCA optional)
* user embedding vectors
* cluster assignments (GMM)
* Top-K recommendations per user
* ranking and classification metrics
* performance visualizations
