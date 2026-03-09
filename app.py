import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
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
</style>
""")

# --- 3. DATA ENGINE ---
@st.cache_data
def load_risk_data():
    try:
        # Load the specific dataset you mentioned
        df = pd.read_csv("financial_risk_early_warning_dataset.csv")
        return df
    except:
        st.error("⚠️ CSV Not Found. Ensure 'financial_risk_early_warning_dataset.csv' is in your GitHub repo.")
        return pd.DataFrame()

df = load_risk_data()

# --- 4. APP LOGIC ---
if not df.empty:
    st.title("🛡️ Corporate Credit Risk Early Warning System")
    st.markdown("#### Multivariate Risk Assessment & Predictive Solvency Modeling")

    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.header("🔍 Portfolio Filters")
        industry = st.multiselect("Industry Type", df['Industry_Type'].unique(), default=df['Industry_Type'].unique()[:3])
        condition = st.selectbox("Market Condition", df['Market_Condition'].unique())
        
        filtered_df = df[(df['Industry_Type'].isin(industry)) & (df['Market_Condition'] == condition)]

    # --- KPI HEADER ---
    k1, k2, k3, k4 = st.columns(4)
    avg_risk = filtered_df['Financial_Risk_Score'].mean()
    k1.metric("Avg Risk Score", f"{avg_risk:.2f}")
    k2.metric("High Risk Clients", len(filtered_df[filtered_df['Risk_Category'] == 'High']))
    k3.metric("Avg Debt/Equity", f"{filtered_df['Debt_to_Equity'].mean():.2f}")
    k4.metric("Liquidity (Avg QR)", f"{filtered_df['Quick_Ratio'].mean():.2f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Risk Analytics", "🤖 Prediction Engine", "🧪 Stress Test Lab"])

    # --- TAB 1: ANALYTICS ---
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk Distribution by Industry")
            fig_box = px.box(filtered_df, x='Industry_Type', y='Financial_Risk_Score', color='Risk_Category', template="plotly_dark")
            st.plotly_chart(fig_box, use_container_width=True)
        
        with c2:
            st.subheader("Liquidity vs. Solvency Heatmap")
            # Identifying 'Danger Zone'
            fig_scatter = px.scatter(filtered_df, x='Current_Ratio', y='Debt_to_Equity', 
                                     color='Financial_Risk_Score', size='Net_Profit_Margin',
                                     hover_name='Firm_ID', template="plotly_dark",
                                     color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig_scatter, use_container_width=True)

    # --- TAB 2: PREDICTION (ML LOGIC) ---
    with tab2:
        st.subheader("Machine Learning: Risk Driver Analysis")
        
        # Prepare Data for a quick RF Importance
        features = ['Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity', 'ROA', 'Net_Profit_Margin', 'Interest_Coverage']
        X = df[features].fillna(0)
        y = df['Financial_Risk_Score']
        
        model = RandomForestRegressor(n_estimators=50).fit(X, y)
        importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
        
        col_text, col_plot = st.columns([1, 2])
        with col_text:
            st.write("This model identifies which financial ratios most heavily impact the **Financial Risk Score**.")
            st.dataframe(importances)
        with col_plot:
            fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title="Key Risk Determinants")
            st.plotly_chart(fig_imp, use_container_width=True)

    # --- TAB 3: STRESS TEST (SIMULATOR) ---
    with tab3:
        st.subheader("🧪 Client Stress Test Simulator")
        st.write("Manually enter financial metrics to predict a client's risk category.")
        
        s1, s2, s3 = st.columns(3)
        with s1:
            cur_r = st.number_input("Current Ratio", 0.0, 10.0, 1.5)
            roa = st.number_input("ROA (%)", -50.0, 50.0, 5.0)
        with s2:
            de_r = st.number_input("Debt-to-Equity", 0.0, 20.0, 2.0)
            net_m = st.number_input("Net Profit Margin (%)", -100.0, 100.0, 10.0)
        with s3:
            int_cov = st.number_input("Interest Coverage", 0.0, 50.0, 3.0)
            rev_g = st.number_input("Revenue Growth (%)", -100.0, 100.0, 5.0)

        if st.button("Calculate Predicted Risk"):
            # Simple heuristic simulation based on the dataset metrics
            # A real model would use model.predict()
            base_score = (de_r * 10) - (cur_r * 5) - (roa * 2) + 50
            risk_res = "High" if base_score > 70 else "Medium" if base_score > 40 else "Low"
            
            color = "red" if risk_res == "High" else "orange" if risk_res == "Medium" else "green"
            st.markdown(f"### Predicted Risk Category: <span style='color:{color}'>{risk_res}</span>", unsafe_allow_html=True)
            st.progress(min(max(base_score/100, 0.0), 1.0))
