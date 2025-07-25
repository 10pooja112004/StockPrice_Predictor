import streamlit as st
import pandas as pd
import joblib  # For loading the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import ta  # For technical analysis

# Title
st.title("📈 Stock Market Price Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(data.tail())

    # Preprocessing (same as your notebook)
    if 'Close' in data.columns:
        # Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

        # Technical Indicators (RSI, MACD)
        close_series = data['Close'].squeeze()  # Make sure it's a Series, not DataFrame

        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(close=close_series, window=14)
        data['RSI'] = rsi_indicator.rsi()

        # MACD
        macd = ta.trend.MACD(close=close_series)
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_diff'] = macd.macd_diff()

        # Drop NaN values created after calculations
        data.dropna(inplace=True)

        # Features and Target
        features = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff']
        X = data[features]
        y = data['Close']

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Load Pretrained Model (if available)
        try:
            rf_model = joblib.load("stock_price_predictor_model.pkl")
            st.write("Model loaded successfully!")
        except FileNotFoundError:
            st.write("Model not found, training a new one...")
            # Train the model if not available
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            joblib.dump(rf_model, "stock_price_predictor_model.pkl")  # Save the model for future use

        # Predict on latest data
        prediction = rf_model.predict(X_test)

        # Show predictions
        st.subheader("📊 Predictions vs Actual")
        result_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': prediction})
        st.write(result_df.head())

        # Plot
        st.line_chart(result_df)

    else:
        st.error("Make sure your CSV has a 'Close' column.")
