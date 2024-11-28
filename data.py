from environments.CryptoEnvRL import CryptoEnvRL
import pandas as pd

# Load cryptocurrency price series
file_path = "data/BTC_USD.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
price_series = df["Close"].values

print(price_series[:10])  # Inspect the first 10 elements

