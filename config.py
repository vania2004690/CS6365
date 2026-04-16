
# config.py
DATA_PATH = "workout_fitness_tracker_data.csv"

# Labeling rule for implicit "acceptance" (binary classification target)
# We'll mark a row as accepted (1) if either:
#  - Mood After Workout is 'Energized' or 'Happy'
#  - OR Calories Burned is >= dataset median
POSITIVE_MOODS = {"Energized", "Happy"}

# Optional: PCA components (set to None to disable PCA)
PCA_COMPONENTS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
