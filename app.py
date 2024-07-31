import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import keras
import tensorflow as tf
import streamlit as st
import datetime
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

st.title('Stock Price Prediction')

ticker = st.text_input('Enter the stock ticker:', 'AAPL')
end_date = datetime.date.today().strftime('%Y-%m-%d')
df = yf.download(ticker, start="2000-01-01", end=end_date) 
stock_name = yf.Ticker(ticker).info['longName']
st.write(f"Stock Name: {stock_name}")

# Get more stock info
info = yf.Ticker(ticker).info

# Display more stock data
st.subheader('Stock Information')
st.write(f"52 Week High: {info['fiftyTwoWeekHigh']}")
st.write(f"52 Week Low: {info['fiftyTwoWeekLow']}")
st.write(f"Market Cap: {info['marketCap']}")
st.write(f"PE Ratio: {info['trailingPE']}")
st.write(f"Dividend Yield: {info['dividendYield']}")

# Calculate the 100-day moving average
df['100MA'] = df['Adj Close'].rolling(window=100).mean()

# Calculate the 200-day moving average
df['200MA'] = df['Adj Close'].rolling(window=200).mean()

# describe data
st.subheader('Data from 2000 to 2021')
st.write(df.describe())

# visualize data
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Adj Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
st.pyplot(fig)

# time window data visualization
user_input = st.text_input('Enter the number of years to visualize:', 5)
number_of_years = int(user_input)
number_of_days = number_of_years * 252
st.subheader(f'Closing Price with MA for the last {number_of_years} years')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Adj Close'].tail(number_of_days))
plt.plot(df['100MA'].tail(number_of_days))
plt.plot(df['200MA'].tail(number_of_days))
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(loc='upper left')
st.pyplot(fig)

# Heatmap of the correlation matrix
st.subheader('Heatmap of Correlation Matrix')
corr = df.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap='coolwarm')
st.pyplot(fig)

# Risk analysis
st.subheader('Risk Analysis')

# Calculate the daily returns
df['return'] = df['Adj Close'].pct_change()

# Calculate the sharpe ratio
average_return = df['return'].mean()
return_std = df['return'].std()
sharpe_ratio = average_return / return_std

# Explain Sharpe Ratio
st.write("""
The Sharpe Ratio is a measure for calculating risk-adjusted return, and this ratio is a measure of the excess return (or Risk Premium) per unit of deviation in an investment asset or a trading strategy. 
""")
st.write(f"Sharpe Ratio: {sharpe_ratio}")

# Explain what the current Sharpe Ratio means
if sharpe_ratio < 1:
    st.write("A Sharpe Ratio less than 1 is generally considered sub-optimal. The current Sharpe Ratio indicates that the risk taken is not justified by the return.")
elif sharpe_ratio < 2:
    st.write("A Sharpe Ratio between 1 and 2 is generally considered acceptable. The current Sharpe Ratio indicates that the return is adequate for the level of risk taken.")
else:
    st.write("A Sharpe Ratio greater than 2 is generally considered excellent. The current Sharpe Ratio indicates that the return is high relative to the level of risk taken.")

# Initialize model with a default value
model = ''

# Create three columns
col1, col2, col3, col4 = st.columns(4)

# Create a button in each column for model selection
if col1.button('LSTM'):
    model = 'LSTM'
elif col2.button('Linear Regression'):
    model = 'Linear Regression'
elif col3.button('Random Forest'):
    model = 'Random Forest'
elif col4.button('Sentiment Analysis'):
    model = 'Sentiment Analysis'

# Prepare data for LSTM model
df['Next_Close'] = df['Adj Close'].shift(-1)
df = df.dropna()

# Define the features and target
X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', '100MA', '200MA']].values
y = df['Next_Close'].values

# Reshape y for the scaler 
y = y.reshape(-1, 1)

# Scale the features and target
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# Split the data into training and testing sets
split_point = int(len(X) * 0.8)  # 80% of the data will be used for training
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Save the dates corresponding to the test set
date_index = df.index[split_point:]

# Reshape the data to fit the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# Use the selected model
if model == 'LSTM':
    # LSTM Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    
    # Initialize the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 7)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    # Compile and fit the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invert predictions back to normal values
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform(y_train)
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform(y_test)

    # Create a DataFrame with actual and predicted values
    df = pd.DataFrame({
        'Actual': y_test.flatten(),
        'Predicted': test_predict.flatten()
    })

    # Display the DataFrame in Streamlit
    st.dataframe(df)

    # Plot actual vs predicted prices
    st.subheader('LSTM Predicted vs Actual Price')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(date_index, y_test, label='Actual Price')
    plt.plot(date_index, test_predict, label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left')
    st.pyplot(fig2)

    # Predict the next day's price
    # Get the last sequence from data
    last_sequence = X[-1].reshape(1, 1, -1)

    # Predict the next step
    next_prediction = model.predict(last_sequence)

    # Invert prediction back to normal value
    next_prediction = scaler.inverse_transform(next_prediction)

    # the last date in the index
    last_date = date_index[-1]

    next_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)

    st.subheader(f"The forecasted price for {next_date.strftime('%Y-%m-%d')} is: {next_prediction[0][0]}")

    # RMSE
    rmse = sqrt(mean_squared_error(y_test, test_predict))
    st.write(f"Root Mean Squared Error for the LSTM model is: {rmse}")

elif model == 'Linear Regression':
    # Linear Regression Model
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Initialize the Linear Regression model
    lr_model = LinearRegression()

    # Reshape the data to fit the Linear Regression model
    X_train_lr, X_test_lr = X_train.reshape(X_train.shape[0], X_train.shape[2]), X_test.reshape(X_test.shape[0], X_test.shape[2])
    
    # Fit the model on the training data
    lr_model.fit(X_train_lr, y_train)

    # Make predictions on the testing data
    lr_predictions = lr_model.predict(X_test_lr)

    # Invert predictions back to normal values
    lr_predictions = scaler.inverse_transform(lr_predictions)

    # Invert y_test back to normal values
    y_test_inv = scaler.inverse_transform(y_test)

    # Calculate RMSE before inverse transformation
    mse = mean_squared_error(y_test_inv, lr_predictions)
    rmse = np.sqrt(mse)

    # Create a DataFrame with actual and predicted values
    df = pd.DataFrame({
        'Actual': y_test_inv.flatten(),
        'Predicted': lr_predictions.flatten()
    })

    # Display the DataFrame in Streamlit
    st.dataframe(df)


    # Plot actual vs predicted prices
    st.subheader('Linear Regression Predicted vs Actual Price')
    fig3 = plt.figure(figsize=(12,6))
    plt.plot(date_index, y_test_inv, label='Actual Price')
    plt.plot(date_index, lr_predictions, label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left')
    st.pyplot(fig3)

    # Evaluate the model
    st.subheader(f'Next Day Prediction: {lr_predictions[-1]}')
    st.write(f"Root Mean Squared Error for Linear Regression: {rmse}")

elif model == 'Random Forest':
    from sklearn.ensemble import RandomForestRegressor

    scaler = MinMaxScaler()
    print(df.head())  

    # Initialize the Random Forest Regressor model
    rf_model = RandomForestRegressor()

    # Reshape the data to fit the Random Forest Regressor model
    X = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', '100MA', '200MA']]
    X = X[:int(len(df)-1)] 
    y = df['Next_Close']
    y = y[:int(len(df)-1)]
        
    # Fit the model on the training data
    rf_model.fit(X, y)

    # Make predictions on the testing data
    rf_predictions = rf_model.predict(X)

    print(f'The model score is: {rf_model.score(X, y)}')

    # make predictions
    new_data = df[['Open', 'High', 'Low', 'Adj Close', 'Volume', '100MA', '200MA']].tail(1)
    prediction = rf_model.predict(new_data)
    st.write(f"Next day prediction: {float(prediction)}")
    st.write(f"The actual price for the next day is: {float(df['Next_Close'].tail(1))}")

    # Calculate RMSE before inverse transformation
    mse = mean_squared_error(y, rf_predictions)
    rmse = np.sqrt(mse)

    # Create a DataFrame for the actual and predicted values
    df_plot = pd.DataFrame({'Actual': y, 'Predicted': rf_predictions})

    # Get the current date
    current_date = datetime.datetime.now()

    # Calculate the date 4 years ago
    four_years_ago = current_date - datetime.timedelta(days=4*365)

    # Convert the index to datetime if it's not already
    df_plot.index = pd.to_datetime(df_plot.index)

    # Filter df_plot to only include the last 4 years of data
    df_plot = df_plot[df_plot.index >= four_years_ago]

    # Plot the actual and predicted values
    fig, ax = plt.subplots(figsize=(14,8))
    ax.plot(df_plot['Actual'], label='Actual')
    ax.plot(df_plot['Predicted'], label='Predicted')
    plt.title('Actual vs Predicted Close Prices (Last 4 Years)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Pass the figure to st.pyplot()
    st.pyplot(fig)

    st.write(f"Root Mean Squared Error for Random Forest: {rmse}")

# sentiment analysis
elif model == 'Sentiment Analysis':
    st.subheader('Sentiment Analysis')
    # API to fetch news data
    # GNews API
    from gnewsclient import gnewsclient
    from textblob import TextBlob

    # Create a gnewsclient object
    client = gnewsclient.NewsClient(language='english', location='united states', topic=stock_name, max_results=5)

    # Fetch the news articles
    news_list = client.get_news()

    # Initialize a variable to hold the total sentiment
    total_sentiment = 0

    # Loop through the news articles
    for news in news_list:
        # Get the text of the news article
        title = news.get('title', '')
        description = news.get('description', '')
        text = title + ' ' + description

        # Create a TextBlob object
        blob = TextBlob(text)

        # Get the sentiment of the text
        sentiment = blob.sentiment.polarity

        # Add the sentiment to the total sentiment
        total_sentiment += sentiment

        # Print the title and sentiment of each news article
        st.write(f"Title: {title}")
        st.write(f"Sentiment: {sentiment}")

    # Calculate the average sentiment
    average_sentiment = total_sentiment / len(news_list)

    # Print the average sentiment
    st.write('Average sentiment:', average_sentiment)

# streamlit run /Users/safwanislam/Desktop/stockprediction/app.py 