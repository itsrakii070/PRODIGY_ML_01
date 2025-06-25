# PRODIGY_ML_01
# 🏡 House Price Prediction Using Linear Regression

This project implements a **Linear Regression** model to predict house prices based on:
- Square footage
- Number of bedrooms
- Number of bathrooms

---

## 📊 Example Features

| Feature        | Description                       |
|----------------|-----------------------------------|
| `square_feet`  | Total area of the house in sqft   |
| `bedrooms`     | Number of bedrooms                |
| `bathrooms`    | Number of bathrooms               |
| `price`        | Target variable (house price in $)|

---

## 🚀 How It Works

1. Input features (sqft, beds, baths) are fed to a linear regression model.
2. The model learns weights for each feature to best predict house prices.
3. The model is evaluated using metrics like **Mean Squared Error (MSE)** and **R² score**.

---

## 🧠 Tech Stack

- Python 🐍
- scikit-learn
- pandas
- matplotlib (optional for visualization)

---

## 🧪 Sample Run

```bash
$ python house_price_prediction.py

Predicted Prices: [290000.]
Mean Squared Error: 100000000.0
R² Score: 0.95
