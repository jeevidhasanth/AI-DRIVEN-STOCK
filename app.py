
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="TCS Stock Forecast", layout="centered")
st.image("https://companieslogo.com/img/orig/TCS.NS_BIG.D-51dd0dc3.png", width=150)
st.markdown("<h1 style='text-align: center; color: darkblue;'>AI-Driven TCS Stock Forecast</h1>", unsafe_allow_html=True)

df = pd.read_csv("TCS_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df = df[df['Close'] > 0]

products = sorted(df['Product'].unique())
selected_product = st.selectbox("Select TCS Product", products)

min_year = df['Year'].min()
future_year = st.selectbox("Select Year to Predict", list(range(min_year, 2031)))

df_filtered = df[df['Product'] == selected_product]
X = df_filtered[['Year']]
y = df_filtered['Close']
model = LinearRegression()
model.fit(X, y)
predicted = model.predict([[future_year]])[0]

st.markdown(f"<h3 style='color:#007acc;'>Predicted Close Price for {selected_product} in {future_year}: ₹{predicted:.2f}</h3>", unsafe_allow_html=True)

avg_close = df_filtered.groupby("Year")["Close"].mean().reset_index()
fig = px.scatter(avg_close, x="Year", y="Close", trendline="lowess",
                 title="Close Price Trend (LOWESS)", color_discrete_sequence=["#004aad"])
st.plotly_chart(fig)

level = "Low" if predicted < 1000 else "Medium" if predicted < 2500 else "High"
bar_df = pd.DataFrame({
    "Level": ["Low", "Medium", "High"],
    "Value": [1 if level == l else 0 for l in ["Low", "Medium", "High"]]
})
fig_bar = px.bar(bar_df, x="Level", y="Value", color="Level",
                 color_discrete_map={"Low": "red", "Medium": "orange", "High": "green"},
                 title="Prediction Category")
st.plotly_chart(fig_bar)

st.success(f"Prediction Category: {level}")
