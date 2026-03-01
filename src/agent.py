import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def get_pandas_agent(df: pd.DataFrame):
    llm = ChatOllama(model="llama3", temperature=0.1)
    return llm


def ask_agent(llm, query: str, df: pd.DataFrame = None) -> str:
    if df is None:
        return "⚠️ Error: The dataset was not passed to the agent."

    try:
        context = f"Total rows in dataset: {df.shape[0]}\n\n"
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if 'segment' in df.columns:
            context += f"--- RFM SEGMENT COUNTS ---\n{df['segment'].value_counts().to_string()}\n\n"

        # 2. CLTV Segment Özeti
        if 'CLTV_Segment' in df.columns:
            context += f"--- CLTV SEGMENT COUNTS ---\n{df['CLTV_Segment'].value_counts().to_string()}\n\n"

        if 'segment' in df.columns and 'CLTV_Segment' in df.columns:
            cross_tab = pd.crosstab(df['segment'], df['CLTV_Segment'])
            context += f"--- CUSTOMER DISTRIBUTION (RFM vs CLTV Segments) ---\n{cross_tab.to_string()}\n\n"

        if 'segment' in df.columns and numeric_cols:
            grouped_mean = df.groupby('segment')[numeric_cols].mean().round(2)
            context += f"--- AVERAGE VALUES BY RFM SEGMENT ---\n{grouped_mean.to_string()}\n\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Scientist and CRM Consultant.
            You have been provided with a statistical summary of a customer dataset.
            Answer the user's questions STRICTLY based on the provided context data.
            DO NOT try to write python code. Analyze the tables and provide strategic, clear answers.
            IMPORTANT RULES: 
            1. Respond in the language the user speaks to you.
            2. Make your answers professional, actionable, and structured."""),
            ("user", "Context Data:\n{context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": query})

        return response.content

    except Exception as e:
        return f"The local agent encountered an error: {str(e)}"