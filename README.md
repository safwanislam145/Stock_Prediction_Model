# Stock_Prediction_Model

Overview
This Streamlit app predicts stock prices using different machine learning models, including LSTM, Linear Regression, Random Forest, and Sentiment Analysis. The application fetches historical stock data using the Yahoo Finance API and performs various analyses, including moving averages, correlation heatmaps, and risk assessments using the Sharpe Ratio. Users can input a stock ticker, visualize historical data, and select a machine learning model to predict future stock prices.

Features
Stock Data Retrieval: Fetches historical stock data from Yahoo Finance starting from January 1, 2000, to the current date.
Stock Information Display: Shows important stock information such as 52-week high/low, market cap, PE ratio, and dividend yield.
Data Visualization: Plots the closing price over time, moving averages, and a correlation heatmap of the stock data.
Risk Analysis: Calculates and explains the Sharpe Ratio for the stock.
Machine Learning Models: Users can select from LSTM, Linear Regression, Random Forest, or Sentiment Analysis to predict future stock prices.
Sentiment Analysis: Analyzes the sentiment of recent news articles related to the selected stock.

Installation

1. Clone the repository:
git clone https://https://github.com/safwanislam145/Stock_Prediction_Model.git

2. Navigate to the project directory:
cd stock_prediction_model

3. Install the required dependencies:
pip install -r requirements.txt

4. Run the Streamlit app:
streamlit run app.py


Usage
Enter the Stock Ticker:

In the text input field, enter the stock ticker symbol (e.g., AAPL for Apple Inc.).
View Stock Information:

The app will display important information about the stock, including its 52-week high/low, market cap, PE ratio, and dividend yield.
Visualize Historical Data:

The app will display the closing price of the stock over time, moving averages, and a correlation heatmap.
Risk Analysis:

View the Sharpe Ratio for the stock and an explanation of what the ratio means.

Select a Prediction Model:

Choose between LSTM, Linear Regression, Random Forest, or Sentiment Analysis by clicking the respective button.
The selected model will train on the historical data and predict future stock prices.
The app will display the actual vs. predicted prices and the Root Mean Squared Error (RMSE) of the model.
Sentiment Analysis:

If Sentiment Analysis is selected, the app will fetch recent news articles related to the stock and analyze their sentiment.

Libraries and Tools
numpy
pandas
matplotlib
yfinance
seaborn
keras
tensorflow
streamlit
sklearn
textblob
gnewsclient
vaderSentiment

Notes
Adjust the model training epochs and batch size as needed to balance accuracy and performance.
The default start date for historical data is January 1, 2000. Modify this in the code if needed.
