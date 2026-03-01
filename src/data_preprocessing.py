import pandas as pd
import datetime as dt


def check_outliers_thresholds(dataframe: pd.DataFrame, col_name: str, q1=0.25, q3=0.75):

    q1_val = dataframe[col_name].quantile(q1)
    q3_val = dataframe[col_name].quantile(q3)
    iqr = q3_val - q1_val
    up_limit = (q3_val + 1.5 * iqr).round().astype(int)
    low_limit = (q1_val - 1.5 * iqr).round().astype(int)
    return low_limit, up_limit


def replace_outliers(dataframe: pd.DataFrame, col_name: str):

    low_limit, up_limit = check_outliers_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    return dataframe


def calculate_rfm(df: pd.DataFrame, customer_id_col: str, date_col: str, freq_cols: list, monetary_cols: list) -> pd.DataFrame:

    df['Total_Transaction'] = df[freq_cols].sum(axis=1)
    df['Total_Price'] = df[monetary_cols].sum(axis=1)
    df[date_col] = pd.to_datetime(df[date_col])

    analysis_date = df[date_col].max() + dt.timedelta(days=2)

    # RFM
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda date: (analysis_date - date.max()).days,
        'Total_Transaction': lambda num: num.sum(),
        'Total_Price': lambda price: price.sum()
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm[(rfm['Monetary'] > 0) & (rfm['Recency'] > 0)].copy()

    # Skorlama
    rfm['recency_score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
    rfm['frequency_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
    rfm['monetary_score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])
    rfm['RF_SCORE'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

    # Segmentasyon
    seg_map = {
        r'[1-2][1-2]': 'hibernating', r'[1-2][3-4]': 'at_Risk', r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep', r'33': 'need_attention', r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promissing', r'51': 'new_customers', r'[4-5][2-3]': 'potential_loyalists', r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
    return rfm.reset_index()