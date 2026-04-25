from pathlib import Path

BASE_DIR = Path("/Users/vaniamunjar/Desktop/fitness_tracker_recommender")

DATA_PATH = BASE_DIR / "workout_fitness_tracker_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.2

USE_PCA = True
PCA_VARIANCE = 0.95

POSITIVE_CALORIE_THRESHOLD = 300
GMM_COMPONENTS = 5
TOP_K_VALUES = [5, 10, 20]
