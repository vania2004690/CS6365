"""Preprocess workout features for machine learning models."""
from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import DATA_PATH
from utils import read_dataset, clean_workout_data, add_labels, build_workout_id

def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def load_and_preprocess(path=DATA_PATH):
    df = build_workout_id(add_labels(clean_workout_data(read_dataset(path))))
    drop_cols = {"label", "user_id", "date", "workout_key"}
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", _one_hot_encoder())]), categorical_cols),
    ], remainder="drop")
    X = preprocessor.fit_transform(feature_df)
    y = df["label"].astype(int).values
    return X, y, preprocessor, df

if __name__ == "__main__":
    X, y, _, _ = load_and_preprocess()
    print(f"Processed matrix shape: {X.shape}")
    print(f"Positive labels: {int(y.sum())} / {len(y)}")
