import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('Tesla.csv')

dataframe['Date'] = pd.to_datetime(dataframe['Date'])

dataframe = dataframe.sort_values(by='Date')

# Moving average
dataframe['Moving Average 50 days'] = dataframe['Close'].rolling(window=50).mean()
dataframe['Moving Average 200 days'] = dataframe['Close'].rolling(window=200).mean()


# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(dataframe['Date'], dataframe['Close'], label='Close', marker='o')
plt.plot(dataframe['Date'], dataframe['Moving Average 50 days'], label='Moving Average 50 days', marker='o')
plt.plot(dataframe['Date'], dataframe['Moving Average 200 days'], label='Moving Average 200 days', marker='o')


# labels
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Close Price vs. Moving Average')
plt.legend()
plt.grid()

plt.show()