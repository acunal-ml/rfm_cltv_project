import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def get_pandas_agent(df: pd.DataFrame):
    llm = ChatOllama(model="llama3", temperature=0.1)
    return llm


def ask_agent(llm, query: str, df: pd.DataFrame = None) -> str:
    if df is None:
        return "⚠️ Error: The dataset was not passed to the agent. Please ensure 'aktif_veri' is passed as the 3rd parameter to ask_agent in app.py."

    try:
        seg_col = 'segment' if 'segment' in df.columns else 'Segment' if 'Segment' in df.columns else None

        context = f"Total rows in dataset: {df.shape[0]}\n\n"
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if seg_col and numeric_cols:
            segment_counts = df[seg_col].value_counts()
            grouped_sum = df.groupby(seg_col)[numeric_cols].sum().round(2)
            grouped_mean = df.groupby(seg_col)[numeric_cols].mean().round(2)

            context += f"--- SEGMENT COUNTS (Number of Customers) ---\n{segment_counts.to_string()}\n\n"
            context += f"--- SUMS BY SEGMENT (Total Values) ---\n{grouped_sum.to_string()}\n\n"
            context += f"--- MEANS BY SEGMENT (Average Values) ---\n{grouped_mean.to_string()}\n\n"
        else:
            context += f"--- DATA SUMMARY ---\n{df.describe().round(2).to_string()}\n\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Data Scientist and CRM Consultant.
            You have been provided with a statistical summary of a customer dataset.
            Answer the user's questions STRICTLY based on the provided context data.
            DO NOT try to write python code. Analyze the tables and provide strategic, clear answers.
            IMPORTANT RULES: 
            1. Respond in the language the user speaks to you (if they ask in English, use English; if in Turkish, use Turkish).
            2. Make your answers professional, actionable, and structured (use bullet points if necessary)."""),
            ("user", "Context Data:\n{context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": query})

        return response.content

    except Exception as e:
        return f"The local agent encountered an error: {str(e)}"