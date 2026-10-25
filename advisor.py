import os
from openai import OpenAI
import streamlit as st

# Securely load API key
try:
    GROK_KEY = st.secrets["GROK_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    GROK_KEY = os.getenv("GROK_API_KEY")

def get_grok_advice(df_summary, prediction, user_question):
    if not GROK_KEY:
        return "Grok API key is missing. Please check your secrets."
        
    client = OpenAI(
        api_key=GROK_KEY,
        base_url="https://api.x.ai/v1"
    )

    system_prompt = f"""
    You are an expert financial advisor integrated into FinVision AI. 
    Here is the user's financial data summary: {df_summary}
    The machine learning model predicts next month's expenses will be: ${prediction:,.2f}.
    
    Answer the user's question clearly. Explain the prediction based on the data. 
    Provide 2-3 bullet points of actionable advice. Keep it concise and professional.
    """

    try:
        response = client.chat.completions.create(
            model="grok-beta", # Uses the latest available beta model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Grok: {str(e)}"
