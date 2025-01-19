import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('Tesla.csv')
dataframe['Date'] = pd.to_datetime(dataframe['Date'])
dataframe = dataframe.sort_values(by='Date')

# RSI - Relative Strength Index
# Is a techncial indicator that shows recent price changes of a stock and whether it's
#  overbought sold. Used to generate buy and sell signals and identify trend reversals. 
# RSI < 30 --> buys signal and RSI > 70 --> sell signal


def calculate_rsi(data, window):
    delta = data['Close'].diff(1)
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    data['RSI'] = rsi
    return data

# Apply the RSI calculation with a 14-day window
dataframe = calculate_rsi(dataframe, window=14)

# Plot Close price and RSI
fig, axs = plt.subplots(2, figsize=(10, 8), sharex=True)

# Plot Close price
axs[0].plot(dataframe['Date'], dataframe['Close'], label='Close', color='blue', marker='o')
axs[0].set_title('Close Price')
axs[0].set_ylabel('Price')
axs[0].legend()
axs[0].grid()

# Plot RSI
axs[1].plot(dataframe['Date'], dataframe['RSI'], label='RSI', color='red', marker='o')
axs[1].axhline(70, color='green', linestyle='--', label='Overbought (70)')
axs[1].axhline(30, color='orange', linestyle='--', label='Oversold (30)')
axs[1].set_title('Relative Strength Index (RSI)')
axs[1].set_ylabel('RSI')
axs[1].legend()
axs[1].grid()


plt.xlabel('Date')
plt.tight_layout()
plt.show()
