import streamlit as st
import pandas as pd
import plotly.express as px
from extractor import extract_receipt
from processor import clean_data, detect_anomalies, predict_cashflow
from advisor import get_grok_advice

# --- PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="FinVision AI", layout="wide", page_icon="📊")
st.markdown("""
    <style>
    .main { background-color: #f9f9fc; }
    h1 { color: #1f4287; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.05); }
    div[data-testid="stMetricValue"] { color: #1f4287; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 FinVision AI: Smart Expense Auditor")
st.markdown("Upload your data, extract receipts with AI, detect anomalies, and predict future cashflow.")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80) # Placeholder logo
    st.header("1. Upload Data")
    csv_file = st.file_uploader("Historical Data (CSV)", type="csv")
    image_file = st.file_uploader("New Receipt (Image)", type=["jpg", "png"])
    
    if csv_file:
        raw_df = pd.read_csv(csv_file)
        
        st.header("2. Map Columns")
        st.info("Tell the AI which columns to use.")
        date_col = st.selectbox("Date Column", raw_df.columns)
        amount_col = st.selectbox("Amount Column", raw_df.columns)
        cat_col = st.selectbox("Category Column", raw_df.columns)
        
        st.header("3. Settings")
        contamination = st.slider("Anomaly Sensitivity", 0.01, 0.20, 0.05, help="Higher means more items are flagged as anomalies.")

# --- MAIN APP LOGIC ---
if csv_file:
    # 1. Process Data
    df = raw_df.rename(columns={date_col: 'date', amount_col: 'amount', cat_col: 'category'})
    df = clean_data(df)
    df = detect_anomalies(df, contamination)

    # 2. Handle Receipt Upload
    if image_file:
        with st.sidebar:
            with st.spinner("Gemini is reading receipt..."):
                with open("temp.jpg", "wb") as f:
                    f.write(image_file.getbuffer())
                data = extract_receipt("temp.jpg")
                
                if "error" not in data:
                    st.success("Extracted Successfully!")
                    st.json(data)
                    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                    df = clean_data(df)
                else:
                    st.error("Failed to read receipt.")

    # 3. Create Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Interactive Dashboard", "⚠️ Anomalies", "🤖 Grok AI Advisor"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.subheader("Financial Overview")
        
        # Interactive Date Filter
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.date_input("Filter by Date Range", [min_date, max_date])
        
        if len(date_range) == 2:
            mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
            filtered_df = df[mask]
        else:
            filtered_df = df

        # Dynamic KPIs
        prediction, msg = predict_cashflow(filtered_df)
        total_spend = filtered_df['amount'].sum()
        top_cat = filtered_df.groupby('category')['amount'].sum().idxmax() if not filtered_df.empty else "N/A"

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spend (Filtered)", f"${total_spend:,.2f}")
        col2.metric("Top Category", top_cat)
        if prediction:
            col3.metric("Predicted Next Month", f"${prediction:,.2f}", delta=f"Based on ML", delta_color="normal")
        else:
            col3.metric("Predicted Next Month", "N/A", help=msg)

        st.markdown("---")

        # Interactive Charts
        c1, c2 = st.columns(2)
        with c1:
            fig_donut = px.pie(filtered_df, values='amount', names='category', hole=0.4, title="Spend by Category")
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with c2:
            fig_line = px.line(filtered_df, x='date', y='amount', title="Spend Over Time", markers=True)
            fig_line.update_layout(xaxis_title="Date", yaxis_title="Amount ($)")
            st.plotly_chart(fig_line, use_container_width=True)

    # --- TAB 2: ANOMALIES ---
    with tab2:
        st.subheader("Suspicious Transactions")
        anomalies = df[df['is_anomaly'] == True]
        
        if not anomalies.empty:
            st.warning(f"Found {len(anomalies)} unusual transactions based on Isolation Forest.")
            st.dataframe(anomalies[['date', 'merchant', 'amount', 'category']], use_container_width=True)
            
            # Bar chart of anomalies
            fig_anom = px.bar(anomalies, x='date', y='amount', color='merchant', title="Anomaly Timeline")
            st.plotly_chart(fig_anom, use_container_width=True)
        else:
            st.success("No anomalies detected in this dataset.")

    # --- TAB 3: GROK ADVISOR ---
    with tab3:
        st.subheader("Chat with Grok")
        st.markdown("Ask questions about your spending patterns or the prediction.")
        
        # Prepare summary for Grok
        summary = f"Total spend: ${total_spend:,.2f}. Top category: {top_cat}. Data range: {min_date} to {max_date}."
        
        question = st.text_input("E.g., 'Why is my spending so high?' or 'Where can I save money?'")
        
        if question:
            with st.spinner("Grok is analyzing your data..."):
                advice = get_grok_advice(summary, prediction if prediction else 0, question)
                st.info(advice)

else:
    st.info("👈 Please upload a CSV file in the sidebar to get started.")
