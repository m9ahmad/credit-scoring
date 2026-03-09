import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- 1. CONFIG ---
st.set_page_config(page_title="Corporate Credit Risk AI", layout="wide", page_icon="🛡️")

# --- 2. ADVANCED STYLING ---
st.html("""
<style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #00ffcc; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1e2130; border-radius: 5px; color: white; }
    .stTabs [aria-selected="true"] { background-color: #4B5563; border-bottom: 2px solid #00ffcc; }
    .main { background-color: #0e1117; }
</style>
""")

# --- 3. DATA ENGINE ---
@st.cache_data
def load_risk_data():
    try:
        df = pd.read_csv("financial_risk_early_warning_dataset.csv")
        # Ensure numeric columns are clean
        numeric_cols = ['Current_Ratio', 'Debt_to_Equity', 'Financial_Risk_Score', 'Net_Profit_Margin', 'Quick_Ratio', 'ROA', 'Interest_Coverage']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=['Financial_Risk_Score'])
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        return pd.DataFrame()

df = load_risk_data()

# --- 4. APP LOGIC ---
if not df.empty:
    st.title("🛡️ Corporate Credit Risk Early Warning System")
    st.caption("Multivariate Risk Assessment & Predictive Solvency Modeling")

    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.header("🔍 Portfolio Filters")
        all_industries = sorted(df['Industry_Type'].unique())
        industry = st.multiselect("Industry Type", all_industries, default=all_industries[:3])
        
        all_conditions = sorted(df['Market_Condition'].unique())
        condition = st.selectbox("Market Condition", all_conditions)
        
        filtered_df = df[(df['Industry_Type'].isin(industry)) & (df['Market_Condition'] == condition)].copy()

    # --- KPI HEADER ---
    k1, k2, k3, k4 = st.columns(4)
    if not filtered_df.empty:
        k1.metric("Avg Risk Score", f"{filtered_df['Financial_Risk_Score'].mean():.2f}")
        k2.metric("High Risk Clients", len(filtered_df[filtered_df['Risk_Category'] == 'High']))
        k3.metric("Avg Debt/Equity", f"{filtered_df['Debt_to_Equity'].mean():.2f}")
        k4.metric("Liquidity (Avg QR)", f"{filtered_df['Quick_Ratio'].mean():.2f}")
    else:
        st.warning("No data matches the selected filters.")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Risk Analytics", "🤖 Prediction Engine", "🧪 Stress Test Lab"])

    # --- TAB 1: ANALYTICS ---
    with tab1:
        if not filtered_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Risk Distribution by Industry")
                fig_box = px.box(filtered_df, x='Industry_Type', y='Financial_Risk_Score', color='Risk_Category', template="plotly_dark")
                st.plotly_chart(fig_box, use_container_width=True)
            
            with c2:
                st.subheader("Liquidity vs. Solvency Heatmap")
                # FIX: Use absolute value for size to prevent Plotly ValueError with negative Profit Margins
                plot_df = filtered_df.copy()
                plot_df['Abs_Margin'] = plot_df['Net_Profit_Margin'].abs().fillna(0) + 1 
                
                fig_scatter = px.scatter(plot_df, x='Current_Ratio', y='Debt_to_Equity', 
                                         color='Financial_Risk_Score', size='Abs_Margin',
                                         hover_name='Firm_ID', template="plotly_dark",
                                         color_continuous_scale='RdYlGn_r',
                                         labels={'Abs_Margin': 'Profitability (Scale)'})
                st.plotly_chart(fig_scatter, use_container_width=True)

    # --- TAB 2: PREDICTION ENGINE ---
    with tab2:
        st.subheader("Machine Learning: Risk Driver Analysis")
        features = ['Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity', 'ROA', 'Net_Profit_Margin', 'Interest_Coverage']
        
        # Filter for rows that have all features
        ml_df = df.dropna(subset=features + ['Financial_Risk_Score'])
        
        if not ml_df.empty:
            X = ml_df[features]
            y = ml_df['Financial_Risk_Score']
            
            model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
            importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
            
            col_text, col_plot = st.columns([1, 2])
            with col_text:
                st.write("Identified weighted impact of financial ratios on the **Financial Risk Score**.")
                st.table(importances)
            with col_plot:
                fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', 
                                 title="Feature Importance (Random Forest)", template="plotly_dark")
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.error("Insufficient data for Machine Learning model training.")

    # --- TAB 3: STRESS TEST LAB ---
    with tab3:
        st.subheader("🧪 Client Stress Test Simulator")
        st.write("Input financial metrics to simulate a Credit Risk Rating.")
        
        s1, s2, s3 = st.columns(3)
        with s1:
            in_cur = st.number_input("Current Ratio", 0.0, 10.0, 1.5)
            in_roa = st.number_input("ROA (%)", -50.0, 50.0, 5.0)
        with s2:
            in_de = st.number_input("Debt-to-Equity", 0.0, 20.0, 2.0)
            in_margin = st.number_input("Net Profit Margin (%)", -100.0, 100.0, 10.0)
        with s3:
            in_cov = st.number_input("Interest Coverage", 0.0, 50.0, 3.0)
            in_growth = st.number_input("Revenue Growth (%)", -100.0, 100.0, 5.0)

        if st.button("Calculate Predicted Risk"):
            # Simulation Heuristic (Weighted Scoring)
            # High Debt and Low Liquidity/Profitability increase the score
            sim_score = (in_de * 12) + (abs(min(in_roa, 0)) * 2) - (in_cur * 6) - (in_cov * 2) + 40
            
            risk_res = "High" if sim_score > 65 else "Medium" if sim_score > 35 else "Low"
            color = "#ff4b4b" if risk_res == "High" else "#ffa500" if risk_res == "Medium" else "#00ffcc"
            
            st.markdown(f"<h2 style='text-align: center;'>Risk Category: <span style='color:{color}'>{risk_res}</span></h2>", unsafe_allow_html=True)
            st.progress(min(max(sim_score/100, 0.0), 1.0))
            st.caption(f"Calculated Financial Risk Index: {sim_score:.2f}")

else:
    st.info("Please upload the dataset to the repository to begin analysis.")
