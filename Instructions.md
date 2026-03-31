Here’s an **expanded version of your existing `INSTRUCTIONS.md`** that *builds on what you already wrote* (keeps your tone/structure, just extends it for full workflow + AI usage):

---

# INSTRUCTIONS.md

## Project

**Fitness Tracker Recommendation System**
CS 6365 – Intelligent Enterprise Computing
Georgia Institute of Technology

---

## Overview

This repository contains the implementation of a **fitness workout recommendation system**. The project focuses on transforming raw workout data into structured representations, building user behavior profiles, and developing multiple recommendation strategies including content-based filtering, clustering, and collaborative filtering.

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
* reducing dimensionality using PCA
* constructing user embeddings from workout features
* generating recommendations using clustering and collaborative filtering

Future stages include:

* Gaussian Mixture Models (GMM) for soft clustering
* ranking evaluation using Precision@K, Recall@K, and NDCG@K
* visualization of model performance and comparisons

---

## Main Files

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

* apply PCA for dimensionality reduction
* retain maximum variance in features
* generate user embeddings by averaging workout feature vectors

Outputs:

* PCA-transformed feature matrix
* user profile embeddings

---

### `supervised_model.py`

Implements supervised learning models for predicting workout outcomes or user preferences based on processed features.

---

### `tester.py`

Implements collaborative filtering using ALS and evaluates recommendation quality by generating Top-K recommendations.

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

* Applies PCA
* Generates user embeddings

---

### 4. Content-Based + Clustering Recommendations

* Uses K-Means clustering
* Uses cosine similarity on embeddings

(Integrated within main workflow)

---

### 5. Collaborative Filtering (ALS)

```bash
python tester.py
```

* Trains ALS model
* Produces Top-K recommendations

---

## Evaluation (Final Stage)

Planned evaluation includes:

* Precision@K
* Recall@K
* NDCG@K

Outputs:

* model comparison metrics
* performance vs K plots
* recommendation quality analysis

---

## Expected Outputs

* cleaned and structured dataset
* processed feature matrix
* reduced-dimension feature space (PCA)
* user embedding vectors
* cluster assignments
* Top-K recommendations per user
* evaluation metrics and visualizations (final stage)

