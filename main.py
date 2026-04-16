# main.py
from supervised_model import run_supervised_model
from config import DATA_PATH

if __name__ == "__main__":
    model, pipeline, metrics = run_supervised_model(DATA_PATH, model_name="gbt")
    print("\nTraining complete. Summary:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
