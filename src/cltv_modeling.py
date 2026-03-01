import pandas as pd
# import pickle
from lifetimes import BetaGeoFitter, GammaGammaFitter


def predict_cltv(cltv_df: pd.DataFrame, secilen_donem: str) -> pd.DataFrame:
    """It loads trained models from disk and performs inference on live data."""

    bgf = BetaGeoFitter()
    bgf.load_model("models/bgf_model.pkl")

    ggf = GammaGammaFitter()
    ggf.load_model("models/ggf_model.pkl")

    # 2. Tahminleme (Inference) İşlemi
    periyot_map = {"3 Month": 12, "6 Month": 24, "9 Month": 36, "12 Month": 52}
    t = periyot_map[secilen_donem]
    col_ciro = f"{secilen_donem}_Expected_Turnover"

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

    try:
        cltv_df["Segment"] = pd.qcut(cltv_df[col_ciro], 4, labels=["D", "C", "B", "A"])
    except ValueError:
        cltv_df["Segment"] = "A"

    return cltv_df, col_ciro