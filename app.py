# app.py
import streamlit as st
import pandas as pd
import joblib  # For loading the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Title
st.title("📈 Stock Market Price Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(data.tail())

    # Preprocessing (same as your notebook)
    # Example only: Replace with your real feature engineering
    if 'Close' in data.columns:
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data.dropna(inplace=True)

        # Features and Target
        X = data[['MA10', 'MA50']]
        y = data['Close']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predict on latest data
        prediction = rf_model.predict(X_test)

        # Display Predictions vs Actual
        result_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': prediction})
        st.write("Predictions vs Actual")
        st.write(result_df)

        # Evaluate model performance
        mse = mean_squared_error(y_test, prediction)
        mae = mean_absolute_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)

        st.subheader("Model Performance Metrics")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"R² Score: {r2}")

        # Plot actual vs predicted stock prices
        st.subheader("Predicted vs Actual Stock Prices")
        st.line_chart(result_df)

    else:
        st.error("Make sure your CSV has a 'Close' column.")
