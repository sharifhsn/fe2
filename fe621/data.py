import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from curl_cffi import requests
import os

session = requests.Session(impersonate="chrome")

# Define the ticker and date range
ticker = "SPY"
today = datetime.now()
yesterday = today - timedelta(days=1)

# Fetch option chain data for both days
def get_option_chain_data(ticker, date):
    stock = yf.Ticker(ticker, session=session)
    options = stock.option_chain(date)
    return options.calls, options.puts

# Get available expiration dates
stock = yf.Ticker(ticker)
expiration_dates = stock.options

# Filter expiration dates within 30 days
filtered_dates = [
    date for date in expiration_dates
    if datetime.strptime(date, "%Y-%m-%d") <= today + timedelta(days=30)
]

# Fetch option chain data for filtered dates
option_data = {}
for date in filtered_dates:
    calls, puts = get_option_chain_data(ticker, date)
    option_data[date] = {"calls": calls, "puts": puts}

# Print the option data for verification
print(option_data)
# Save the option data to CSV files in the data directory

# Ensure the data directory exists
data_dir = "/home/sharif/Code/fe2/fe621/data"
os.makedirs(data_dir, exist_ok=True)

for date, data in option_data.items():
    calls_file = os.path.join(data_dir, f"calls_{date}.csv")
    puts_file = os.path.join(data_dir, f"puts_{date}.csv")
    
    data["calls"].to_csv(calls_file, index=False)
    data["puts"].to_csv(puts_file, index=False)