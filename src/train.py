import mlflow
import pandas as pd
# import pickle
import os
from lifetimes import BetaGeoFitter, GammaGammaFitter

os.makedirs("models", exist_ok=True)


def train_cltv_models(cltv_df: pd.DataFrame):
    """
    It retrieves CLTV data, trains models, tracks them with MLflow, and saves the models.
    """

    mlflow.set_experiment("Corporate_CLTV_Project")

    with mlflow.start_run() as run:
        # Define the hyperparameters
        bgf_penalizer = 0.001
        ggf_penalizer = 0.01

        # Log the parameters to MLflow
        mlflow.log_param("bgf_penalizer", bgf_penalizer)
        mlflow.log_param("ggf_penalizer", ggf_penalizer)
        mlflow.log_param("train_data_size", len(cltv_df))

        print("Modeller eğitiliyor...")

        # Train the Models
        bgf = BetaGeoFitter(penalizer_coef=bgf_penalizer)
        bgf.fit(cltv_df['cltv_frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])

        ggf = GammaGammaFitter(penalizer_coef=ggf_penalizer)
        ggf.fit(cltv_df['cltv_frequency'], cltv_df['monetary_cltv_avg'])

        # Save Trained Models to Local Disk (We use the library's own method instead of pickling)
        bgf_path = "models/bgf_model.pkl"
        ggf_path = "models/ggf_model.pkl"

        bgf.save_model(bgf_path)
        ggf.save_model(ggf_path)

        # Submit Models to the MLflow Artifact Store
        mlflow.log_artifact(bgf_path, artifact_path="model_artifacts")
        mlflow.log_artifact(ggf_path, artifact_path="model_artifacts")

        print(f"✅ The training is complete and saved to MLflow! Run ID: {run.info.run_id}")

if __name__ == "__main__":
    import numpy as np

    dummy_data = pd.DataFrame({
        'cltv_frequency': np.random.randint(1, 10, 100),
        'recency_cltv_weekly': np.random.randint(1, 50, 100),
        'monetary_cltv_avg': np.random.uniform(10, 100, 100)
    })

    dummy_data['T_weekly'] = dummy_data['recency_cltv_weekly'] + np.random.randint(1, 10, 100)

    print("Training is starting with test data...")
    train_cltv_models(dummy_data)