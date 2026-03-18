# INSTRUCTIONS.md

## Project
Fitness Tracker Recommendation System  
CS 6365 – Intelligent Enterprise Computing  
Georgia Institute of Technology

This repository contains the implementation of a fitness workout recommendation system. The current project focuses on preparing workout data, structuring user workout sessions, and building the preprocessing foundation needed for later recommendation models such as content-based filtering, clustering, and hybrid methods.

The goal of this file is to help an AI coding assistant or any new contributor quickly understand how to set up, run, test, and extend the project.

---

## Repository Purpose

This project uses a fitness tracker dataset to build a recommendation system that can suggest workouts to users based on workout characteristics and user behavior patterns.

At the current stage, the repository mainly supports:
- loading and cleaning the raw workout dataset
- standardizing column handling across dataset variations
- generating structured workout records
- creating user workout session logs
- preprocessing features for downstream machine learning models
- optionally reducing feature dimensionality with PCA

Future stages are expected to include:
- content-based recommendation with cosine similarity
- clustering with K-Means and Gaussian Mixture Models
- collaborative filtering or hybrid recommendation approaches
- ranking evaluation with Precision@K, Recall@K, and NDCG@K

---

## Main Files

### `data_frame.py`
This file loads the workout dataset, cleans column names, keeps the key workout-related fields, creates duration buckets, generates unique workout IDs, and builds user session logs.

In plain terms, it turns the raw dataset into two structured views:
1. a workout catalog with unique workouts
2. a user log showing which workouts each user completed over time

### `data_preprocessing.py`
This file handles the preprocessing pipeline for machine learning.

Its responsibilities include:
- loading the raw CSV
- detecting columns using aliases so the code works across slightly different datasets
- deriving a binary success label based on mood after workout or calories burned
- separating numeric and categorical features
- dropping rows with missing values in the selected features
- scaling numeric columns with `StandardScaler`
- one-hot encoding categorical columns with `OneHotEncoder`
- optionally applying PCA for dimensionality reduction

It returns:
- processed feature matrix `X`
- label vector `y`
- fitted preprocessing pipeline
- metadata about the transformation
