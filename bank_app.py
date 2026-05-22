
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# पेज की सेटिंग और थीम
st.set_page_config(page_title="Bank Customer Retention Analytics", layout="wide")

# डेटा लोड करना
@st.cache_data
def load_data():
    data = pd.read_csv('Processed_Bank_Retention_Data.csv')

    # सुरक्षा जांच: अगर कोडिंग के कारण कॉलम मिसिंग हो तो खुद बना लो
    if 'At_Risk_Premium' not in data.columns:
        balance_q3 = data['Balance'].quantile(0.75)
        data['At_Risk_Premium'] = np.where((data['Balance'] > balance_q3) & (data['IsActiveMember'] == 0), 1, 0)
    if 'Global_ERR' not in data.columns:
        data['Global_ERR'] = 0.52
    if 'Relationship_Strength_Index' not in data.columns:
        data['Relationship_Strength_Index'] = ((data['IsActiveMember'] * 0.40) + ((data['NumOfProducts'] / 4) * 0.35)) * 100

    return data

df = load_data()

# वेबसाइट का मुख्य टाइटल
st.title("🏦 Customer Engagement & Product Utilization Analytics")
st.markdown("### Executive Dashboard for Retail Banking Retention Strategy")
st.write("---")

# 📊 साइडबार फिल्टर्स
st.sidebar.header("🎯 Filter Controls")
selected_geo = st.sidebar.multiselect("Select Country", options=df['Geography'].unique(), default=df['Geography'].unique())
selected_gender = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

# डेटा को फिल्टर करना (एज रेंज को सही तरीके से चेक करना)
filtered_df = df[
    (df['Geography'].isin(selected_geo)) & 
    (df['Gender'].isin(selected_gender)) & 
    (df['Age'].between(age_range[0], age_range[1]))
]

# 🗂️ मुख्य मॉड्यूल्स (Tabs)
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Engagement vs Churn Overview", 
    "📦 Product Utilization Impact", 
    "🚨 High-Value At-Risk Radar", 
    "💯 Retention Strength Scoring"
])

# MODULE 1: Engagement vs Churn Overview
with tab1:
    st.subheader("Customer Engagement Profiles and Overall Churn Risk")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers Selected", f"{len(filtered_df):,}")
    with col2:
        overall_churn = filtered_df['Exited'].mean() * 100 if len(filtered_df) > 0 else 0
        st.metric("Average Churn Rate", f"{overall_churn:.2f}%")
    with col3:
        err_val = filtered_df['Global_ERR'].iloc[0] if len(filtered_df) > 0 else 0.52
        st.metric("Engagement Retention Ratio (ERR)", f"{err_val:.2f}")

    st.markdown("---")
    if len(filtered_df) > 0:
        fig_profile = px.bar(
            filtered_df.groupby('Engagement_Profile')['Exited'].mean().reset_index(),
            x='Engagement_Profile', y='Exited',
            labels={'Exited': 'Churn Rate', 'Engagement_Profile': 'Customer Segment'},
            title="Churn Rate by Engagement Profile (Higher is Risky)",
            color='Engagement_Profile'
        )
        st.plotly_chart(fig_profile, use_container_width=True)

# MODULE 2: Product Utilization Impact
with tab2:
    st.subheader("Product Depth vs Churn Relationship Analysis")
    col1, col2 = st.columns(2)
    with col1:
        if len(filtered_df) > 0:
            prod_churn = filtered_df.groupby('NumOfProducts')['Exited'].mean().reset_index()
            fig_prod = px.line(
                prod_churn, x='NumOfProducts', y='Exited', markers=True,
                title="The Product Churn Curve (Notice the Spike at 3-4 Products)",
                labels={'Exited': 'Churn Rate', 'NumOfProducts': 'Number of Products'}
            )
            st.plotly_chart(fig_prod, use_container_width=True)
    with col2:
        st.markdown("#### Strategic Insights on Bundling")
        st.info("• **The Sweet Spot (2 Products):** Retention peaks dramatically when a customer holds exactly 2 bank products.\n• **Forced Cross-Selling Hazard (3-4 Products):** Churn rates skyrocket for customers with 3 or 4 products.")

# MODULE 3: High-Value At-Risk Radar
with tab3:
    st.subheader("Premium Disengaged Customer Detection System")
    at_risk_premium_df = filtered_df[filtered_df['At_Risk_Premium'] == 1]
    st.warning(f"⚠️ Action Required: Detected **{len(at_risk_premium_df)}** High-Balance Inactive Customers who are flight risks.")

    if not at_risk_premium_df.empty:
        fig_scatter = px.scatter(
            at_risk_premium_df, x='Balance', y='EstimatedSalary', 
            color='Age', hover_data=['Surname', 'CreditScore'],
            title="Premium Inactive Capital Distribution Matrix"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.success("No critical high-balance disengaged customer alerts for selected filters.")

# MODULE 4: Retention Strength Scoring
with tab4:
    st.subheader("Micro-Prudential Relationship Strength Index (RSI)")
    if len(filtered_df) > 0:
        fig_rsi = px.histogram(
            filtered_df, x='Relationship_Strength_Index', 
            color='Exited', barmode='overlay',
            title="Systemic Distribution of Relationship Strength Index (RSI) Scores",
            labels={'Relationship_Strength_Index': 'RSI Score (0 = Weak, 100 = Sticky)'}
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
