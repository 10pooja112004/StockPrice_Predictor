import yfinance as yf

# Download AAPL stock data and save to CSV
yf.download('AAPL', start='2023-01-01', end='2024-01-01').to_csv('AAPL_stock_data.csv')
