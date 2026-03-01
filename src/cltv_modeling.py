import pandas as pd
# import pickle
from lifetimes import BetaGeoFitter, GammaGammaFitter


def predict_cltv(cltv_df: pd.DataFrame, secilen_donem: str) -> pd.DataFrame:
    """Eğitilmiş modelleri disken yükleyip canlı veri üzerinde tahmin (inference) yapar."""

    # 1. Kayıtlı modelleri yükle (Eğitim adımı atlandı, süre ve kaynak tasarrufu sağlandı)
    # Boş modelleri oluştur ve üstüne kaydedilmiş ağırlıkları yükle
    bgf = BetaGeoFitter()
    bgf.load_model("models/bgf_model.pkl")

    ggf = GammaGammaFitter()
    ggf.load_model("models/ggf_model.pkl")

    # 2. Tahminleme (Inference) İşlemi
    periyot_map = {"3 Ay": 12, "6 Ay": 24, "9 Ay": 36, "12 Ay": 52}
    t = periyot_map[secilen_donem]
    col_ciro = f"{secilen_donem}_Beklenen_Ciro"

    cltv_df[col_ciro] = ggf.customer_lifetime_value(
        bgf,
        cltv_df['cltv_frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'],
        cltv_df['monetary_cltv_avg'],
        time=t / 4,
        freq="W",
        discount_rate=0.01
    )

    # Segmentasyon
    try:
        cltv_df["Segment"] = pd.qcut(cltv_df[col_ciro], 4, labels=["D", "C", "B", "A"])
    except ValueError:
        cltv_df["Segment"] = "A"

    return cltv_df, col_ciro