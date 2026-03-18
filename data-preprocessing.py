import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from config import POSITIVE_MOODS, PCA_COMPONENTS

# Aliases for robust column detection across datasets
ALIASES = {
    "workout_type": ["Workout Type", "workout_type", "type"],
    "duration": ["Workout Duration (mins)", "duration", "duration_min", "duration_mins"],
    "calories": ["Calories Burned", "calories", "est_calories"],
    "intensity": ["Workout Intensity", "intensity"],
    "user_id": ["User ID", "user_id", "uid"],
    "heart_rate": ["Heart Rate (bpm)", "heart_rate"],
    "steps": ["Steps Taken", "steps"],
    "distance": ["Distance (km)", "distance_km", "distance"],
    "bmi": ["BMI"],
    "body_fat": ["Body Fat (%)", "body_fat_pct"],
    "water_intake": ["Water Intake (liters)", "water_intake_l"],
    "sleep_hours": ["Sleep Hours", "sleep_hours"],
    "gender": ["Gender", "gender"],
    "location": ["Workout Location", "location"],
    "weather": ["Weather Conditions", "weather"],
    "mood_before": ["Mood Before Workout", "mood_before"],
    "mood_after": ["Mood After Workout", "mood_after"],
    "age": ["Age", "age"],
    "height": ["Height (cm)", "height_cm"],
    "weight": ["Weight (kg)", "weight_kg"],
}

def find_first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def derive_label(df, calories_col, mood_after_col):
    # implicit "accepted" target
    median_cal = df[calories_col].median()
    positive_mood = df[mood_after_col].isin(list(POSITIVE_MOODS)) if mood_after_col else False
    calories_ok = df[calories_col] >= median_cal
    if mood_after_col:
        y = (positive_mood | calories_ok).astype(int)
    else:
        y = calories_ok.astype(int)
    return y

def load_and_preprocess(path, pca_components=PCA_COMPONENTS):
    df = pd.read_csv(path)

    # Map canonical names -> actual columns
    colmap = {k: find_first_present(df, v) for k, v in ALIASES.items()}

    required = ["workout_type", "duration", "calories"]
    missing = [r for r in required if colmap[r] is None]
    if missing:
        raise ValueError(f"Missing required columns in CSV for keys: {missing}. "
                         f"Found columns: {list(df.columns)}" )

    # Derive label
    y = derive_label(df, calories_col=colmap["calories"], mood_after_col=colmap["mood_after"])

    # Build feature set
    # Candidate numeric and categorical columns (choose those present)
    numeric_keys = ["duration", "calories", "heart_rate", "steps", "distance", "bmi",
                    "body_fat", "water_intake", "sleep_hours", "age", "height", "weight"]
    categorical_keys = ["workout_type", "intensity", "gender", "location", "weather",
                        "mood_before"]

    num_cols = [colmap[k] for k in numeric_keys if colmap.get(k) is not None]
    cat_cols = [colmap[k] for k in categorical_keys if colmap.get(k) is not None]

    # Keep a copy of columns used
    cols_used = {"num_cols": num_cols, "cat_cols": cat_cols}

    # Drop rows with NA on columns we actually use
    df_clean = df.dropna(subset=num_cols + cat_cols).reset_index(drop=True)

    X = df_clean[num_cols + cat_cols]
    y = y.loc[df_clean.index]  # align with cleaned rows

    # Pipelines
    numeric_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])

    steps = [("pre", pre)]
    if pca_components is not None and pca_components > 0:
        steps.append(("pca", PCA(n_components=pca_components, random_state=0)))
    pipe = Pipeline(steps)

    X_proc = pipe.fit_transform(X)

    meta = {
        "rows_before": len(df),
        "rows_after_clean": len(df_clean),
        "num_cols_used": num_cols,
        "cat_cols_used": cat_cols,
        "label_rule": "MoodAfter in {Energized, Happy} OR Calories >= median(calories)"
    }

    return X_proc, y.values, pipe, meta

if __name__ == "__main__":
    X, y, pipe, meta = load_and_preprocess("/mnt/data/workout_fitness_tracker_data.csv")
    print("X shape:", X.shape, " y shape:", y.shape)
    print("Meta:", meta)
