import streamlit as st
import pandas as pd
import datetime as dt

from src.data_preprocessing import calculate_rfm
from src.agent import get_pandas_agent, ask_agent


def calculate_cltv(df, customer_id, first_date_col, last_date_col, freq_cols, mon_cols):
    temp_df = df.copy()

    temp_df[first_date_col] = pd.to_datetime(temp_df[first_date_col], errors='coerce')
    temp_df[last_date_col] = pd.to_datetime(temp_df[last_date_col], errors='coerce')

    analysis_date = temp_df[last_date_col].max() + pd.Timedelta(days=2)
    grouped = temp_df.groupby(customer_id)

    max_last_date = grouped[last_date_col].max()
    min_first_date = grouped[first_date_col].min()

    cltv_df = pd.DataFrame(index=grouped.groups.keys())
    cltv_df.index.name = 'Customer_ID'

    cltv_df['Recency'] = (max_last_date - min_first_date).dt.days
    cltv_df['T'] = (analysis_date - min_first_date).dt.days
    cltv_df['Frequency'] = grouped[freq_cols].sum().sum(axis=1)
    cltv_df['Monetary'] = grouped[mon_cols].sum().sum(axis=1)

    cltv_df = cltv_df.reset_index()

    cltv_df['T_safe'] = cltv_df['T'].replace(0, 1)
    cltv_df['CLTV_Score'] = (cltv_df['Frequency'] * cltv_df['Monetary']) / cltv_df['T_safe']
    cltv_df = cltv_df.drop(columns=['T_safe'])

    cltv_df['CLTV_Segment'] = pd.qcut(cltv_df['CLTV_Score'].rank(method='first'), 4, labels=['D', 'C', 'B', 'A'])

    return cltv_df


st.set_page_config(page_title="Corporate CRM Analysis", layout="wide")

if 'rfm_data' not in st.session_state:
    st.session_state.rfm_data = None
if 'cltv_data' not in st.session_state:
    st.session_state.cltv_data = None

st.title("📊 RFM & CLTV Analysis Platform")
st.markdown("Customer segmentation and lifetime value prediction.")

uploaded_file = st.file_uploader("Please Upload a CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f'File uploaded successfully. ({len(df)} rows)')

    select_analysis = st.radio('Select Analysis Type:', ['Integrated RFM & CLTV Analysis'], index=0)

    if select_analysis == 'Integrated RFM & CLTV Analysis':
        st.markdown('### 🛠️ Column Mapping Panel')
        columns = df.columns.tolist()

        # --- RFM COLUMN MAPPING ---
        st.markdown('#### 📌 1. Required Columns for RFM')
        col1, col2 = st.columns(2)
        with col1:
            rfm_customer_id = st.selectbox("RFM - Customer ID:", options=columns, index=None, key="rfm_id")
            rfm_date_col = st.selectbox("RFM - Last Order Date:", options=columns, index=None, key="rfm_date")
        with col2:
            rfm_freq_cols = st.multiselect("RFM - Total Transactions (Frequency) Columns:", options=columns,
                                           key="rfm_freq")
            rfm_monetary_cols = st.multiselect("RFM - Total Spend (Monetary) Columns:", options=columns, key="rfm_mon")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- CLTV COLUMN MAPPING ---
        st.markdown('#### 🔮 2. Required Columns for CLTV Prediction')
        col3, col4 = st.columns(2)
        with col3:
            cltv_customer_id = st.selectbox("CLTV - Customer ID:", options=columns, index=None, key="cltv_id")
            cltv_first_date = st.selectbox("CLTV - First Order Date:", options=columns, index=None, key="cltv_first")
            cltv_last_date = st.selectbox("CLTV - Last Order Date:", options=columns, index=None, key="cltv_last")
        with col4:
            cltv_freq_cols = st.multiselect("CLTV - Total Transactions (Frequency) Columns:", options=columns,
                                            key="cltv_freq")
            cltv_monetary_cols = st.multiselect("CLTV - Total Spend (Monetary) Columns:", options=columns,
                                                key="cltv_mon")
            cltv_month = st.slider("How many months for CLTV prediction?", min_value=1, max_value=12, value=6)

        # --- RUN ANALYSIS ---
        if st.button('🚀 Start Integrated Analysis'):
            if not all(
                    [rfm_customer_id, rfm_date_col, rfm_freq_cols, rfm_monetary_cols, cltv_customer_id, cltv_first_date,
                     cltv_last_date, cltv_freq_cols, cltv_monetary_cols]):
                st.warning("Please complete all column mappings for both RFM and CLTV!")
            else:
                with st.spinner('Modules are running... Calculating RFM and predicting CLTV...'):
                    try:
                        rfm_result = calculate_rfm(df, rfm_customer_id, rfm_date_col, rfm_freq_cols, rfm_monetary_cols)
                        st.session_state.rfm_data = rfm_result

                        cltv_result = calculate_cltv(df, cltv_customer_id, cltv_first_date, cltv_last_date,
                                                     cltv_freq_cols, cltv_monetary_cols)
                        st.session_state.cltv_data = cltv_result

                        st.success("Integrated calculation completed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================
    if st.session_state.rfm_data is not None:
        st.markdown('### 🎯 RFM Analysis Results')
        rfm_display = st.session_state.rfm_data

        if 'segment' in rfm_display.columns:
            selected_rfm_segments = st.multiselect("Filter RFM Segment:", options=rfm_display['segment'].unique())
            if selected_rfm_segments:
                rfm_display = rfm_display[rfm_display['segment'].isin(selected_rfm_segments)]

        st.dataframe(rfm_display)
        st.download_button("📥 Download RFM CSV", data=rfm_display.to_csv(index=False).encode('utf-8'),
                           file_name='rfm_results.csv', mime='text/csv')

    if st.session_state.cltv_data is not None:
        st.markdown("---")
        st.markdown(f'### 🔮 CLTV Prediction Results ({cltv_month} Months)')
        cltv_display = st.session_state.cltv_data

        if 'CLTV_Segment' in cltv_display.columns:
            selected_cltv_segments = st.multiselect("Filter CLTV Segment (A=Highest):",
                                                    options=cltv_display['CLTV_Segment'].unique())
            if selected_cltv_segments:
                cltv_display = cltv_display[cltv_display['CLTV_Segment'].isin(selected_cltv_segments)]

        st.dataframe(cltv_display)
        st.download_button("📥 Download CLTV CSV", data=cltv_display.to_csv(index=False).encode('utf-8'),
                           file_name='cltv_results.csv', mime='text/csv')

    # ==========================================
    # AI AGENT (Llama 3) SECTION
    # ==========================================
    st.markdown("---")
    st.markdown('### 🤖 AI Marketing Consultant (Local - Llama 3)')

    aktif_veri = None
    if st.session_state.rfm_data is not None and st.session_state.cltv_data is not None:
        try:
            aktif_veri = pd.merge(st.session_state.rfm_data, st.session_state.cltv_data, left_on=rfm_customer_id,
                                  right_on='Customer_ID', how='inner')
        except:
            aktif_veri = st.session_state.rfm_data

    if aktif_veri is not None:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask a question about your dataset (e.g., Who are the champion customers in segment A?)")

        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Local AI (Llama 3) is analyzing the data..."):
                agent = get_pandas_agent(aktif_veri)
                cevap = ask_agent(agent, prompt, aktif_veri)

                with st.chat_message("assistant"):
                    st.markdown(cevap)
                st.session_state.messages.append({"role": "assistant", "content": cevap})
    else:
        st.info("Please run the Integrated Analysis first to use the AI assistant.")