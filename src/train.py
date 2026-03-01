import mlflow
import pandas as pd
# import pickle
import os
from lifetimes import BetaGeoFitter, GammaGammaFitter

# Kurumsal projelerde model dosyalarının tutulacağı lokal dizin
os.makedirs("models", exist_ok=True)


def train_cltv_models(cltv_df: pd.DataFrame):
    """
    CLTV verisini alır, modelleri eğitir, MLflow ile takip eder ve modelleri kaydeder.
    """
    # MLflow Deneyini Başlat / Seç
    mlflow.set_experiment("Corporate_CLTV_Project")

    with mlflow.start_run() as run:
        # Hiperparametreleri tanımla
        bgf_penalizer = 0.001
        ggf_penalizer = 0.01

        # 1. Parametreleri MLflow'a Logla
        mlflow.log_param("bgf_penalizer", bgf_penalizer)
        mlflow.log_param("ggf_penalizer", ggf_penalizer)
        mlflow.log_param("train_data_size", len(cltv_df))

        print("Modeller eğitiliyor...")

        # 2. Modelleri Eğit
        bgf = BetaGeoFitter(penalizer_coef=bgf_penalizer)
        bgf.fit(cltv_df['cltv_frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])

        ggf = GammaGammaFitter(penalizer_coef=ggf_penalizer)
        ggf.fit(cltv_df['cltv_frequency'], cltv_df['monetary_cltv_avg'])

        # 3. Eğitilen Modelleri Lokal Diske Kaydet (Pickle yerine kütüphanenin kendi metodunu kullanıyoruz)
        bgf_path = "models/bgf_model.pkl"
        ggf_path = "models/ggf_model.pkl"

        bgf.save_model(bgf_path)
        ggf.save_model(ggf_path)

        # 4. Modelleri MLflow Artifact Store'a Gönder
        mlflow.log_artifact(bgf_path, artifact_path="model_artifacts")
        mlflow.log_artifact(ggf_path, artifact_path="model_artifacts")

        print(f"✅ Eğitim tamamlandı ve MLflow'a kaydedildi! Run ID: {run.info.run_id}")


# if __name__ == "__main__":
#     # Gerçek senaryoda burada data_preprocessing.py'den dönen temiz veri setini çağırırız.
#     # Örnek kullanım:
#     # df = pd.read_csv("data/raw_data.csv")
#     # clean_df = prepare_cltv_data(df)
#     # train_cltv_models(clean_df)
#     print("Eğitim pipeline'ı tetiklendi. (Veri bağlantısı gerektirir)")

if __name__ == "__main__":
    import numpy as np

    # Test için 100 satırlık sahte (dummy) bir CLTV verisi oluşturuyoruz
    dummy_data = pd.DataFrame({
        'cltv_frequency': np.random.randint(1, 10, 100),
        'recency_cltv_weekly': np.random.randint(1, 50, 100),
        'monetary_cltv_avg': np.random.uniform(10, 100, 100)
    })
    # T_weekly her zaman recency'den büyük olmalıdır
    dummy_data['T_weekly'] = dummy_data['recency_cltv_weekly'] + np.random.randint(1, 10, 100)

    print("Test verisiyle eğitim başlatılıyor...")
    train_cltv_models(dummy_data)