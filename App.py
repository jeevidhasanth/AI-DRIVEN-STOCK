import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression

df = pd.read_csv("TCS_stock_history.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df = df[df['Close'].notnull() & (df['Close'] > 0)]
df['Year'] = df['Date'].dt.year
df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
df = df[(df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)]
df.reset_index(drop=True, inplace=True)

selected_product = "TCS Digital"
future_year = 2027
df_product = df[df['Product'] == selected_product]

X = df_product[['Year']]
y = df_product['Close']
model = LinearRegression()
model.fit(X, y)
predicted = model.predict([[future_year]])[0]

strength = {"Low": 0, "Medium": 0, "High": 0}
if predicted < 1000:
    strength["Low"] = 1
elif predicted < 2500:
    strength["Medium"] = 1
else:
    strength["High"] = 1

print(f"Predicted Close Price: ₹{predicted:.2f}")
print("Prediction Outcome (0/1 Format):")
print(strength)

fig1 = px.box(df, x="Product", y="Close", color="Product", title="Box Plot of Closing Prices by Product")
fig1.write_image("box_plot.png")

sns.displot(df['Close'], kde=True, bins=30, color="skyblue")
plt.title("Distribution of Closing Prices")
plt.xlabel("Closing Price (INR)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("distribution_plot.png")
