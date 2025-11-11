
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="💰",
    layout="wide"
)


@st.cache_resource
def load_assets():
    model = pickle.load(open("sales_prediction_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_assets()


st.title(" Advanced Sales Prediction Dashboard")
st.markdown("""
Welcome to the **Sales Forecasting App** — powered by a trained **Random Forest Regressor**.  
Adjust ad budgets, visualize predictions in real-time, and explore model insights!
""")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header(" Input Parameters")
    
    tv = st.slider("TV Advertising Budget ($)", 0.0, 300.0, 150.0, step=10.0)
    radio = st.slider("Radio Advertising Budget ($)", 0.0, 50.0, 25.0, step=1.0)
    newspaper = st.slider("Newspaper Advertising Budget ($)", 0.0, 120.0, 20.0, step=5.0)
    
    input_df = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    scaled_input = scaler.transform(input_df)
    predicted_sales = model.predict(scaled_input)[0]

    st.metric(label=" Predicted Sales", value=f"{predicted_sales:.2f} units")

    # Pie chart for budget distribution
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
    ax1.pie(input_df.iloc[0], labels=input_df.columns, autopct='%1.1f%%', colors=['#1f77b4','#2ca02c','#ff7f0e'])
    ax1.set_title("Ad Budget Distribution")
    st.pyplot(fig1)

with col2:
    st.header(" Prediction Overview")
    
    # Bar chart for input budget
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.barplot(x=input_df.columns, y=input_df.iloc[0], palette="viridis", ax=ax2)
    ax2.set_title("Advertising Budgets")
    st.pyplot(fig2)
    
    # What-if analysis: vary one feature
    st.subheader(" What-if Analysis (TV Budget Effect)")
    tv_values = np.linspace(0, 300, 50)
    temp_df = pd.DataFrame({'TV': tv_values, 'Radio': [radio]*50, 'Newspaper': [newspaper]*50})
    scaled_temp = scaler.transform(temp_df)
    preds = model.predict(scaled_temp)
    
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.plot(tv_values, preds, color='teal', linewidth=2)
    ax3.set_xlabel("TV Advertising Budget ($)")
    ax3.set_ylabel("Predicted Sales (units)")
    ax3.set_title("Impact of TV Spend on Sales")
    st.pyplot(fig3)

st.divider()

st.header(" Model Insights")

try:
    # Extract feature importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': ['TV', 'Radio', 'Newspaper'],
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig4, ax4 = plt.subplots(figsize=(6, 3))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='mako', ax=ax4)
    ax4.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig4)

    st.markdown("**💡 Insight:** The higher the bar, the more influence that advertising channel has on total sales.")
except Exception as e:
    st.warning(f"Feature importance unavailable: {e}")


st.divider()
st.header(" Explore Your Own Dataset (Optional)")
uploaded_file = st.file_uploader("Upload a CSV file with 'TV', 'Radio', 'Newspaper' columns", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    st.subheader(" Correlation Heatmap")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)

st.markdown("---")
st.caption("Made with  by Ritesh using Streamlit and Machine Learning")
