import os
import json
import cv2
import base64
import google.generativeai as genai
import streamlit as st

# Securely load API key (Works on Streamlit Cloud and Locally)
try:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_KEY)

def extract_receipt(image_path):
    # Preprocess image (grayscale, helps the AI focus)
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not read image file."}
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, buffer = cv2.imencode('.jpg', gray)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = """
    Extract the following fields from this receipt image and return ONLY a valid JSON object.
    Do not include markdown formatting or extra text. Just the JSON.
    Keys required:
    - "date": (YYYY-MM-DD)
    - "merchant": (String, store name)
    - "amount": (Float, total amount)
    - "category": (String: choose from Software, Travel, Food, Office Supplies, Utilities, Other)
    """
    
    try:
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_b64}])
        # Clean the response to ensure it's pure JSON
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}
