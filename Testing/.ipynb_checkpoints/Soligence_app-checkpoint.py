#!/usr/bin/env python
# coding: utf-8

# 
# ### Soligence App

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import mplfinance as mpf
from datetime import datetime
import matplotlib.dates as mdates
import feedparser
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load cryptocurrency price data
data = pd.read_csv('Cleaned_combined_crypto_data.csv')
selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')

def home_section():
    st.title("SOLiGence")
    st.write(
        "Solent Intelligence (SOLiGence) is a leading financial multinational organisation that deals"
        " with stock and shares, saving and investments.")

    # Streamlit app
    st.header('Welcome to SOLiGence')
    st.write("Your Intelligent Coin Trading Platform")

    # Brief Description
    st.write("Empower your cryptocurrency trading decisions with AI-driven insights and real-time data.")

    # Call to Action (CTA) Button
    if st.button("Get Started"):
        st.write("Let's explore the world of cryptocurrency trading together!")

        # Background Image
        st.image('sal.jpg', use_column_width=True)
        if 'section' not in st.session_state:
            st.session_state.section = "Home"

        # News and Updates
        st.write("News and Updates:")
        st.info("Stay tuned for the latest updates and trends in the cryptocurrency market!")

        # Testimonials
        st.write("What Our Users Say:")
        st.write("The SOLiGence app transformed how I approach cryptocurrency trading. Highly recommended!")

        # Contact Information
        st.write("Contact Us:")
        st.write("For inquiries, email us at info@soligence.com")

        # Social Media Links
        st.write("Connect with Us:")
        st.markdown("[Twitter](https://twitter.com/SOLiGenceApp) [LinkedIn](https://linkedin.com/company/soligence)")

        # Privacy and Disclaimer
        st.write("Privacy Policy | Disclaimer")



def about_us():
    st.title("About Solent Intelligence Ltd.")
    st.write(
        "The scale of this organization's operation is impressive, with millions of subscribers and over 150 billion "
        "pounds worth of investments. This emphasizes the substantial influence that data-driven decisions can have "
        "on managing such a significant amount of assets. The app's focus on implementing an Intelligent Coin Trading "
        "(IST) platform, specifically tailored for crypto coin predictions, resonates deeply with me. The idea of "
        "utilizing AI to predict cryptocurrency prices on different timeframes, such as daily, weekly, monthly, "
        "and quarterly, truly piques my interest. This approach aligns perfectly with my desire to explore how AI can "
        "shape and enhance our daily lives. The app's ability to recommend trading opportunities by analyzing "
        "AI-generated predictions showcases the tangible applications of data science in the financial world. "
        "Considering a more neutral perspective, while the concept of the app is exciting, there are potential "
        "challenges that need to be acknowledged. Cryptocurrency markets are known for their volatility, and even the "
        "most sophisticated AI predictions might not always be entirely accurate. Users relying solely on these "
        "predictions could face risks if market conditions change unexpectedly.")

def dataset():
    st.title("Crypto Dataset of Five Coins")

    # Sidebar options
    st.sidebar.subheader("Dataset Options")

    # Sorting
    sort_column = st.sidebar.multiselect("Sort by:", data.columns)
    ascending = st.sidebar.checkbox("Ascending")
    sorted_data = data.sort_values(by=sort_column, ascending=ascending)

    # Filtering
    selected_crypto = st.sidebar.selectbox("Filter by cryptocurrency:", ["All"] + data['Crypto'].unique())
    if "All" not in selected_crypto:
        sorted_data = sorted_data[sorted_data['Crypto'].isin(selected_crypto)]

    # Pagination
    page_size = st.sidebar.number_input("Items per page:", min_value=1, value=10)
    page_number = st.sidebar.number_input("Page number:", min_value=1, value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    paginated_data = sorted_data.iloc[start_idx:end_idx]

    # Display the dataset using a dataframe
    st.subheader("Dataset Overview")
    st.dataframe(paginated_data)

    # Provide an option to show a table view
    show_table = st.checkbox("Show as Table")

    # Display the dataset as a table if the checkbox is selected
    if show_table:
        st.subheader("Dataset Table View")
        st.table(paginated_data)
        



def plot_average_price_trend(data, interval):
    """
    Plot the average price trend of cryptocurrencies based on the specified interval.

    Parameters:
    - crypto_data (pd.DataFrame): DataFrame containing cryptocurrency data with 'Date' as index and 'Close' prices as columns.
    - interval (str): Interval for resampling the data ('D' for daily, 'W' for weekly, 'M' for monthly).
    """
    # Set 'Date' column as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Resample the data based on the specified interval and calculate the mean
    if interval == 'Daily':
        resampled_data = data['Close']
        interval_label = 'Daily'
    elif interval == 'Weekly':
        resampled_data = data['Close'].resample('W').mean()
        interval_label = 'Weekly'
    elif interval == 'Monthly':
        resampled_data = data['Close'].resample('M').mean()
        interval_label = 'Monthly'
    else:
        st.error("Invalid interval. Please choose either 'D' for daily, 'W' for weekly, or 'M' for monthly.")
        return

    # Plot the trend using date index and average price
    plt.figure(figsize=(10, 6))
    plt.plot(resampled_data.index, resampled_data, color='blue', marker='o', linestyle='-')

    # Adding labels and title
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Average Price', fontsize=14)
    plt.title(f'Average {interval_label} Cryptocurrency Price Trend', fontsize=16)

    # Set x-axis date format
    if interval == 'M':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        if interval == 'W':
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
        elif interval == 'D':
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    st.pyplot(plt)



def plot_boxplot(data, coin):
    """
    Plot a boxplot of the High, Low, Open, Close, and Volume for a selected coin.

    Parameters:
    - data (pd.DataFrame): DataFrame containing cryptocurrency data.
    - coin (str): The cryptocurrency symbol (coin name) to plot.
    """
    # Filter data for the selected coin
    coin_data = data[data['Crypto'] == coin]

    # Extract columns for boxplot
    boxplot_data = [coin_data['Low'], coin_data['Close'], coin_data['Open'], coin_data['High']]

    # Plot boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data, patch_artist=True, notch=True, vert=True, showfliers=True)
    plt.title(f"Boxplot of High, Low, Open, Close for {coin}")
    plt.xlabel("Metrics")
    plt.ylabel("Price")
    plt.xticks([1, 2, 3, 4], ['Low', 'Close', 'Open', 'High'])

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)


    

# Example usage:
# Assuming 'data' is your DataFrame containing cryptocurrency data with 'Crypto' as the column for coin names
# Replace 'data' with your actual DataFrame name
# Replace 'BTC-GBP' with the cryptocurrency symbol (coin name) you want to plot


    
    

def pivot_data(data):
    """
    Pivot the DataFrame to have the date as the index and coins as columns.

    Parameters:
    - crypto_data (pd.DataFrame): DataFrame containing cryptocurrency data.

    Returns:
    - pd.DataFrame: Pivot DataFrame with the date as index and coins as columns.
    """
    pivoted_data = data.pivot(index='Date', columns='Crypto', values='Close')
    return pivoted_data

def analyze_coin_correlation(data):
    """
    Analyzes and displays the top four positively and negatively correlated cryptocurrencies 
    with a user-selected coin.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the historical data of cryptocurrencies.
    """
    st.header("Coin Correlation Analysis")
    st.write("Streamlit interface for analyzing and displaying the top four positively and negatively correlated cryptocurrencies with user-selected coins.")

    coins_list = data.columns.tolist()
    st.write("Available coins:", ', '.join(coins_list))

    coin_selected = st.selectbox("Select the coin you want to analyze:", coins_list)

    selected_coin_prices = data[coin_selected]

    # Calculate correlations with the selected coin
    correlations = data.corrwith(selected_coin_prices)

    # Sort correlations in descending order
    sorted_correlations = correlations.sort_values(ascending=False)

    # Exclude the selected coin itself for positive and negative correlations
    sorted_correlations = sorted_correlations.drop(coin_selected)

    positively_correlated = sorted_correlations.head(4)
    negatively_correlated = sorted_correlations.tail(4)

    st.write(f"Top four positively correlated cryptocurrencies with {coin_selected}:")
    st.write(positively_correlated)

    st.write(f"\nTop four negatively correlated cryptocurrencies with {coin_selected}:")
    st.write(negatively_correlated)


        
def plot_moving_average(crypto_data):
    """
    Plots the moving average for a user-selected cryptocurrency.

    Parameters:
    - crypto_data (pd.DataFrame): DataFrame containing the cryptocurrency data.
    """
    st.title("Moving Average Analysis")
    st.write("Plots the moving average for a user-selected cryptocurrency.")

    # Get the list of available cryptocurrencies
    available_coins = crypto_data['Crypto'].unique()

    # Streamlit user input for selecting the coin
    coin_selected = st.selectbox("Select the cryptocurrency you want to analyze:", available_coins)

    # User input for selecting the window size
    window_size = st.radio("Select the window size for the moving average:", ['Short (30-day MA)', 'Medium (60-day MA)', 'Long (90-day MA)'])

    # Map window size selection to the corresponding window value
    window_mapping = {'Short (30-day MA)': 30, 'Medium (60-day MA)': 60, 'Long (90-day MA)': 90}
    window = window_mapping[window_size]

    # Filter data for the selected coin
    selected_data = crypto_data[crypto_data['Crypto'] == coin_selected].copy()

    # Calculate the moving average based on the selected window size
    selected_data['MA'] = selected_data['Close'].rolling(window=window).mean()
    window_title = f'{window}-day MA'

    # Plotting
    st.write(f"Price Chart for {coin_selected} with {window_title}")
    plt.figure(figsize=(10, 6))
    plt.plot(selected_data.index, selected_data['Close'], label=f'{coin_selected} Price')
    plt.plot(selected_data.index, selected_data['MA'], label=window_title)
    plt.title(f'{coin_selected} Price and Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

        


# def plot_crypto_prices_visualization(data, compare_option, coins, metrics, start_date, end_date):
#     """
#     Plots the price metrics of selected cryptocurrencies over a specified interval, with optional comparisons.

#     Parameters:
#     - compare_option (str): Option for comparison. Can be 'coins' to compare metrics between coins, 'metrics' to compare within a coin, or 'none' for no comparison.
#     - coins (list): List of cryptocurrencies to plot.
#     - metrics (list): List of price metrics to plot (e.g., 'Close', 'Open', 'High', 'Low', 'Volume').
#     - start_date (str): Start date of the interval in 'YYYY-MM-DD' format.
#     - end_date (str): End date of the interval in 'YYYY-MM-DD' format.
#     """
#     # Convert start date and end date strings to datetime objects
#     start_date = datetime.strptime(start_date, '%Y-%m-%d')
#     end_date = datetime.strptime(end_date, '%Y-%m-%d')

#     st.title('Explore historical cryptocurrency prices.')
#     st.header('Cryptocurrency Price Metrics Visualization')
#     st.write("Plots the price metrics of selected cryptocurrencies over a specified interval, with optional comparisons")

#     # Filter data for the selected coins and interval
#     selected_data = data[(data['Crypto'].isin(coins)) & (data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]

#     # Plot the selected price metrics
#     plt.figure(figsize=(12, 6))

#     if compare_option == 'coins':
#         for metric in metrics:
#             for coin in coins:
#                 coin_data = selected_data[selected_data['Crypto'] == coin]
#                 plt.plot(coin_data.index, coin_data[metric], label=f"{coin} {metric}")
#     elif compare_option == 'metrics':
#         for coin in coins:
#             for metric in metrics:
#                 coin_data = selected_data[selected_data['Crypto'] == coin]
#                 plt.plot(coin_data.index, coin_data[metric], label=f"{coin} {metric}")
#     else:
#         for coin in coins:
#             for metric in metrics:
#                 coin_data = selected_data[selected_data['Crypto'] == coin]
#                 plt.plot(coin_data.index, coin_data[metric], label=f"{coin} {metric}")

#     # Add labels and title
#     plt.title('Cryptocurrency Prices Over Time')
#     plt.xlabel('Date')
#     plt.ylabel('Price/Volume')
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.grid(True)

#     # Show plot
#     st.pyplot(plt)

#first metric

def plot_crypto_metrics(data, coin, metrics, start_date, end_date):
    """
    Plots cryptocurrency coins for specified metrics within a date range.

    Parameters:
    - data (pd.DataFrame): DataFrame containing cryptocurrency data with 'Date' as index and coin names as columns.
    - coin (str): Cryptocurrency symbol.
    - metrics (list): List of price metrics to plot.
    - start_date (str): Start date of the interval in 'YYYY-MM-DD' format.
    - end_date (str): End date of the interval in 'YYYY-MM-DD' format.
    """
    # Ensure 'Date' column is in datetime format and set it as the index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
    
    plt.figure(figsize=(10, 6))
    
    # Convert start_date and end_date to datetime objects and set timezone to UTC
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Filter data for the specified coin and date range
    coin_data = data[(data['Crypto'] == coin) & (data.index >= start_date) & (data.index <= end_date)]
    
    # Print coin_data for debugging
    print(coin_data)
    
    if coin_data.empty:
        print(f"No data available for {coin} within the specified date range.")
        return
    
    for metric in metrics:
        plt.plot(coin_data.index, coin_data[metric], label=f"{coin} {metric}")
    plt.title(f'Cryptocurrency Prices for {coin} ({start_date} to {end_date})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    
    
# second Coins

def plot_crypto_coins(data, coins, metric, start_date, end_date):
    """
    Plots cryptocurrency coins for specified metric within a date range.

    Parameters:
    - crypto_data (pd.DataFrame): DataFrame containing cryptocurrency data with 'Date' as index and coin names as columns.
    - coins (list): List of cryptocurrency symbols.
    - metric (str): Price metric to plot.
    - start_date (str): Start date of the interval in 'YYYY-MM-DD' format.
    - end_date (str): End date of the interval in 'YYYY-MM-DD' format.
    """
    # Ensure 'Date' column is in datetime format and set it as the index
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
    plt.figure(figsize=(10, 6))
    # Convert start_date and end_date to datetime objects and set timezone to UTC
    start_date = pd.to_datetime(start_date).tz_localize('UTC')
    end_date = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Filter data for the specified coin and date range
    
    
    for coin in coins:
        coin_data = data[(data['Crypto'] == coin) & (data.index >= start_date) & (data.index <= end_date)]
        plt.plot(coin_data.index, coin_data[metric], label=f"{coin} {metric}")
    plt.title(f'Cryptocurrency {metric} Prices for {", ".join(coins)} ({start_date} to {end_date})')
    plt.xlabel('Date')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

# def plot_candlestick_chart(coin='BTC-GBP', period='D'):
#     st.subheader('Cryptocurrency Price Visualization by Interval')
#     # Filter the DataFrame for the selected coin
#     coin_data = data[data['Crypto'] == coin].copy()

#     # Make sure the index is a DatetimeIndex
#     coin_data.index = pd.to_datetime(coin_data.index)

#     # Resample data based on the selected period
#     if period.upper() == 'W':
#         resampled_data = coin_data.resample('W').agg({'Open': 'first',
#                                                       'High': 'max',
#                                                       'Low': 'min',
#                                                       'Close': 'last',
#                                                       'Volume': 'sum'})
#     elif period.upper() == 'M':
#         resampled_data = coin_data.resample('M').agg({'Open': 'first',
#                                                       'High': 'max',
#                                                       'Low': 'min',
#                                                       'Close': 'last',
#                                                       'Volume': 'sum'})
#     else:  # Default to daily data if period is not weekly or monthly
#         resampled_data = coin_data

#     # Plotting
#     st.title(f'{coin} {period.upper()} Candlestick Chart')
#     fig, ax = plt.subplots()
#     mpf.plot(resampled_data, type='candle', style='charles', ax=ax,
#              title=f'{coin} {period.upper()} Candlestick Chart', warn_too_much_data=1000)
#     st.pyplot(fig)

def plot_candlestick_chart(data, coin='BTC-GBP', period='D'):
    st.subheader('Cryptocurrency Price Visualization by Interval')
    # Filter the DataFrame for the selected coin
    coin_data = data[data['Crypto'] == coin].copy()

    # Make sure the index is a DatetimeIndex
    coin_data.index = pd.to_datetime(coin_data.index)

    # Resample data based on the selected period
    if period.upper() == 'W':
        resampled_data = coin_data.resample('W').agg({'Open': 'first',
                                                      'High': 'max',
                                                      'Low': 'min',
                                                      'Close': 'last',
                                                      'Volume': 'sum'})
    elif period.upper() == 'M':
        resampled_data = coin_data.resample('M').agg({'Open': 'first',
                                                      'High': 'max',
                                                      'Low': 'min',
                                                      'Close': 'last',
                                                      'Volume': 'sum'})
    else:  # Default to daily data if period is not weekly or monthly
        resampled_data = coin_data

    # Plotting
    st.title(f'{coin} {period.upper()} Candlestick Chart')
    fig, ax = plt.subplots()
    mpf.plot(resampled_data, type='candle', style='charles', ax=ax,
             warn_too_much_data=1000)
    ax.set_title(f'{coin} {period.upper()} Candlestick Chart')
    st.pyplot(fig)



# # Example usage
# import plotly.graph_objects as go

# def plot_candlestick_chart(data, coin='BTC-GBP', period='D'):
#     st.subheader('Cryptocurrency Price Visualization by Interval')
    
#     # Filter the DataFrame for the selected coin
#     coin_data = data[data['Crypto'] == coin].copy()

#     # Make sure the index is a DatetimeIndex
#     coin_data.index = pd.to_datetime(coin_data.index)

#     # Resample data based on the selected period
#     if period.upper() == 'W':
#         resampled_data = coin_data.resample('W').agg({'Open': 'first',
#                                                       'High': 'max',
#                                                       'Low': 'min',
#                                                       'Close': 'last',
#                                                       'Volume': 'sum'})
#     elif period.upper() == 'M':
#         resampled_data = coin_data.resample('M').agg({'Open': 'first',
#                                                       'High': 'max',
#                                                       'Low': 'min',
#                                                       'Close': 'last',
#                                                       'Volume': 'sum'})
#     else:  # Default to daily data if period is not weekly or monthly
#         resampled_data = coin_data

#     # Create candlestick chart
#     fig = go.Figure(data=[go.Candlestick(x=resampled_data.index,
#                                          open=resampled_data['Open'],
#                                          high=resampled_data['High'],
#                                          low=resampled_data['Low'],
#                                          close=resampled_data['Close'])])

#     # Add volume bars
#     fig.add_trace(go.Bar(x=resampled_data.index,
#                          y=resampled_data['Volume'],
#                          marker_color='rgba(0, 0, 255, 0.5)',
#                          opacity=0.5,
#                          name='Volume',
#                          secondary_y=True))  # Set secondary_y to True for volume bars

#     # Update figure layout
#     fig.update_layout(title=f'{coin} {period.upper()} Candlestick Chart',
#                       xaxis_title='Date',
#                       yaxis_title='Price',
#                       template='plotly_dark')

#     # Show plot
#     st.plotly_chart(fig)



# Example usage
# Assuming 'data' is your DataFrame with cryptocurrency data
# Make sure to load data before using this function
# You can use st.cache() to load the data once and cache it
# data = load_data()




def visualize_market_state(data):
    """
    Predicts the market state (up or down) for a group of chosen cryptocurrencies.

    Parameters:
    - crypto_data (pd.DataFrame): DataFrame containing the cryptocurrency data with date as index.

    Returns:
    - market_state (pd.Series): Series indicating the market state for each day.
    """

    # User input for selecting cryptocurrencies
    st.sidebar.title("Market State Prediction")
    available_coins = data['Crypto'].unique()
    coin_selected = st.sidebar.selectbox("Select cryptocurrencies to analyze", available_coins)

    # Filter data for the selected coins
    # selected_data = data[data['Crypto'].isin(coins_selected)].copy()
    selected_data = data[data['Crypto'] == coin_selected].copy()
    
    # Convert 'Close' column to numeric, dropping rows with non-numeric values
    selected_data['Close'] = pd.to_numeric(selected_data['Close'], errors='coerce')
    selected_data.dropna(subset=['Close'], inplace=True)

    # Calculate daily price changes for each coin
    price_changes = selected_data['Close'].pct_change()

    # Replace infinite values with NaN and drop rows with NaN values
    price_changes.replace([np.inf, -np.inf], np.nan, inplace=True)
    price_changes.dropna(inplace=True)

    # Predict market state (up or down) based on overall price movement of selected coins
    market_state = np.sign(price_changes).astype(float)
    
    # Display explanation
    st.write("\nExplanation:")
    st.write("Down: A value of -1.0 suggests that the market is predicted to decrease or go down on that particular day.")
    st.write("UP: A value of 1.0 suggests that the market is predicted to increase or go up on that particular day.")

    # Visualize market state
    fig, ax = plt.subplots(figsize=(10, 6))
    selected_data['Close'].plot(ax=ax, label='Close Price')
    ax.scatter(selected_data.index, selected_data['Close'], c='g', label='Up Market', marker='^', alpha=0.5)
    ax.scatter(selected_data.index, selected_data['Close'], c='r', label='Down Market', marker='v', alpha=0.5)
    ax.set_title(f'{coin_selected} Market State Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.show()
    st.pyplot(fig)
    

def predict_highs_lows(data):
    """
    Predicts possible highs and lows of a chosen cryptocurrency.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the cryptocurrency data.

    Returns:
    - predicted_highs (pd.DataFrame): DataFrame indicating possible highs.
    - predicted_lows (pd.DataFrame): DataFrame indicating possible lows.
    """
    # Extract available coins from the data
    available_coins = data['Crypto'].unique()

    # User input for selecting the coin
    coin_selected = st.sidebar.selectbox("Select the cryptocurrency to analyze", available_coins)

    # Filter data for the selected coin
    selected_data = data[data['Crypto'] == coin_selected].copy()

    # Calculate percentiles for highs and lows
    high_threshold = np.percentile(selected_data['High'], 90)
    low_threshold = np.percentile(selected_data['Low'], 10)

    # Predict highs and lows based on thresholds
    predicted_highs = selected_data['Close'] > high_threshold
    predicted_lows = selected_data['Close'] < low_threshold

    # Visualize predicted highs and lows
    st.subheader(f'Predicted Highs and Lows for {coin_selected}')
    plt.figure(figsize=(10, 6))
    plt.plot(selected_data.index, selected_data['Close'], label='Close Price')
    plt.scatter(selected_data.index[predicted_highs], selected_data['Close'][predicted_highs], color='red', label='Predicted Highs')
    plt.scatter(selected_data.index[predicted_lows], selected_data['Close'][predicted_lows], color='green', label='Predicted Lows')
    plt.title(f'Predicted Highs and Lows for {coin_selected}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Format output as DataFrame
    predicted_highs_df = pd.DataFrame({'Date': selected_data.index, 'Predicted Highs': predicted_highs})
    predicted_lows_df = pd.DataFrame({'Date': selected_data.index, 'Predicted Lows': predicted_lows})

    return predicted_highs_df, predicted_lows_df
    
    
    
import feedparser
import streamlit as st

def get_top_crypto_news(crypto, num_stories=5):
    # URL of the RSS feed from Cryptoslate
    cryptoslate_feed_url = 'https://cryptoslate.com/feed/'

    # Parse the RSS feed from Cryptoslate
    cryptoslate_feed = feedparser.parse(cryptoslate_feed_url)

    # Filter news stories based on the chosen cryptocurrency from Cryptoslate
    crypto_news_cryptoslate = [entry for entry in cryptoslate_feed.entries if crypto.lower() in entry.title.lower()]

    # URL of the RSS feed from CoinDesk
    coindesk_feed_url = 'https://feeds.feedburner.com/CoinDesk'

    # Parse the RSS feed from CoinDesk
    coindesk_feed = feedparser.parse(coindesk_feed_url)

    # Filter news stories based on the chosen cryptocurrency from CoinDesk
    crypto_news_coindesk = [entry for entry in coindesk_feed.entries if crypto.lower() in entry.title.lower()]

    # Combine news stories from both sources
    crypto_news_combined = crypto_news_cryptoslate + crypto_news_coindesk

    # Sort combined news stories by date
    crypto_news_combined.sort(key=lambda x: x.published_parsed, reverse=True)

    # Display the top news stories
    if crypto_news_combined:
        st.write(f"Top {num_stories} news stories about {crypto}:")
        for i, entry in enumerate(crypto_news_combined[:num_stories], start=1):
            st.write(f"{i}. [{entry.title}]({entry.link})")
    else:
        st.write(f"No news found for {crypto}.")


        
# # Load and preprocess data
# def load_selected_data():
#         try:
#             selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')
#             return selected_data
#         except FileNotFoundError:
#             st.error("Failed to load selected data. File not found.")
#             return None

# selected_data = load_selected_data()

    
def display_selected(selected_data):
    st.title("Boxplot of the 4 selected coins from K-mean clustering")
    # selected_data = load_data()  # Assuming load_data() loads your DataFrame

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))  # Adjust nrows and ncols as per your dataset
    colors = ['skyblue', 'lightgreen', 'tan', 'pink']  # List of colors for each boxplot

    for ax, color, (columnName, columnData) in zip(axes, colors, selected_data.iteritems()):
        # Plot boxplot with patch_artist=True to allow for color filling
        bp = ax.boxplot(columnData, patch_artist=True, notch=True, vert=True, showfliers=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)  # Set color for each box
        ax.set_title(columnName)  # Set the title for each subplot to the name of the coin

    fig.suptitle('Boxplot of the 4 selected coins from K-mean clustering', fontsize=22, x=0.5)  # Center the main title
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the top to make space for the suptitle

    st.pyplot(fig)


    
        


# if side_bars == "Predictions":
#     st.title('Cryptocurrency Price Volatility Example')
#     st.write("Cryptocurrency markets are known for their complexities and volatility.")

#     # Select cryptocurrencies for prediction
#     selected_crypto = st.multiselect("Select cryptocurrencies:", data['Crypto'].unique())

#     # Prepare data for Linear Regression
#     plt.figure(figsize=(10, 6))
#     for crypto in selected_crypto:
#         crypto_data = data[data['Crypto'] == crypto]
#         x = np.array(crypto_data.index).reshape(-1, 1)
#         y = crypto_data['Price']

#         # Linear Regression model
#         lr = LinearRegression()
#         lr.fit(x, y)
#         predicted = lr.predict(x)

#         # Display predicted values and corresponding dates
#         st.write(f"Predicted Values and Corresponding Dates for {crypto}:")
#         prediction_df = pd.DataFrame({'Date': crypto_data['Date'], 'Predicted Price': predicted})
#         st.write(prediction_df)

#         # Display scatter plot with Linear Regression line
#         plt.scatter(x, y, label=f'Actual Prices ({crypto})')
#         plt.plot(x, predicted, label=f'Linear Regression ({crypto})')

#         plt.xlabel('Days')
#         plt.ylabel('Price')
#         plt.title(f'Cryptocurrency Price Volatility Prediction for {", ".join(selected_crypto)}')
#         plt.legend()
#         st.pyplot(plt)

#     # crypto_data = data[data['Crypto'] == 'Bitcoin']
#     # Select cryptocurrency for prediction
#     selected_crypto = st.selectbox("Select a cryptocurrency:", data['Crypto'].unique())
#     crypto_data = data[data['Crypto'] == selected_crypto]

#     # Prepare data for Linear Regression
#     x = np.array(crypto_data.index).reshape(-1, 1)
#     y = crypto_data['Price']

#     # Linear Regression model
#     lr = LinearRegression()
#     lr.fit(x, y)
#     predicted = lr.predict(x)

#     # Display predicted values and corresponding dates
#     st.write(f"Predicted Values and Corresponding Dates for {selected_crypto}:")
#     prediction_df = pd.DataFrame({'Date': crypto_data['Date'], 'Predicted Price': predicted})
#     st.write(prediction_df)

#     st.write(f"Visualizing Cryptocurrency Price Prediction for {selected_crypto}:")
#     # Display scatter plot with Linear Regression line
#     plt.figure(figsize=(10, 6))
#     plt.scatter(x, y, label='Actual Prices', color='blue')
#     plt.plot(x, predicted, label='Linear Regression', color='red')
#     plt.xlabel('Days')
#     plt.ylabel('Price')
#     plt.title(f'Cryptocurrency Price Volatility Prediction for {selected_crypto}')
#     plt.legend()
#     st.pyplot(plt)

#     # Prepare data for Linear Regression
#     x = np.array(crypto_data.index).reshape(-1, 1)
#     y = crypto_data['Price']

#     # Linear Regression model
#     lr = LinearRegression()
#     lr.fit(x, y)
#     predicted = lr.predict(x)

#    # Display predicted values and corresponding dates
#     st.write("Predicted Values and Corresponding Dates:")
#     st.write(pd.DataFrame({'Date': crypto_data['Date'], 'Predicted Price': predicted}))

#     st.write("Let's visualize the limitations of a Linear Regression model.")
#     # Display scatter plot with Linear Regression line
#     plt.figure(figsize=(10, 6))
#     plt.scatter(x, y, label='Actual Prices', color='blue')
#     plt.plot(x, predicted, label='Linear Regression', color='red')
#     plt.xlabel('Days')
#     plt.ylabel('Price')
#     plt.title('Cryptocurrency Price Volatility')
#     plt.legend()
#     st.pyplot(plt)


    
# Sidebar navigation
side_bars = st.sidebar.radio("Navigation", ["Home", "About Us", "Dataset","Coin Correlation","Moving Average", "Visualizations","Predictions","NEWS"])


# Conditionals for sidebar navigation
if side_bars == "Home":
    home_section()
elif side_bars == "About Us":
    about_us()
elif side_bars == "Dataset":
    Selection = st.sidebar.radio("Selection",['Dataset','Visualization of Data'])
    if Selection == 'Dataset':
        dataset()
    elif Selection == 'Visualization of Data':
        Vis= st.sidebar.radio("Selection",["Plot Average Price Trend","BoxPlot"])
        
        if Vis == "Plot Average Price Trend":
            interval = st.sidebar.radio("Select Interval:", ['Daily', 'Weekly', 'Monthly'])

            # Plot the selected average price trend
            if st.sidebar.button("Plot Average Price Trend"):
                plot_average_price_trend(data, interval)
        elif Vis == "BoxPlot":
            # Display coin selection dropdown
            selected_coin = st.sidebar.selectbox("Select a cryptocurrency:", data['Crypto'].unique())

            # Plot boxplot for the selected coin
            if st.sidebar.button("Plot BoxPlot"):
                plot_boxplot(data, selected_coin)
        
        
elif side_bars == "Coin Correlation":

    # Pivot the data
    pivoted_data = pivot_data(data)

    # Example usage
    st.title('Coin Correlation Analysis')
    analyze_coin_correlation(pivoted_data)


elif side_bars == "Moving Average":
    plot_moving_average(data)
elif side_bars == "Visualizations":
    st.title("Welcome to the World of Data Visualisation for Crypto Data")
    st.image('sal.jpg', use_column_width=True)
    # Visualization section
    visualization_option = st.sidebar.radio("Select Visualization:", ['Price Comparison', 'Candlestick Chart','Market State Visualization',"Predicted Highs and Lows"])

    if visualization_option == 'Price Comparison':
        vis_option = st.sidebar.radio("Select how you would like compare the data with Visualization:", ['Metrics', 'Coins'])
        if vis_option == 'Metrics':
            coin = st.sidebar.selectbox("Select the cryptocurrency you want to plot:", data['Crypto'].unique())
            metrics = st.sidebar.multiselect("Select price metrics you want to plot (e.g., Close, Open, High, Low):",['Close', 'Open', 'High', 'Low'])
            start_date = st.sidebar.date_input("Enter the start date of the interval:")
            end_date = st.sidebar.date_input("Enter the end date of the interval:")

            if st.sidebar.button("Plot Cryptocurrency Prices", key="plot_button"):
                plot_crypto_metrics(data, coin.upper(), metrics, str(start_date), str(end_date))

        elif vis_option == 'Coins':
            coin = st.sidebar.multiselect("Enter the cryptocurrencies you want to plot ( e.g., BTC-GBP,ETH-USD):",data['Crypto'].unique())
            metric = st.sidebar.selectbox("Enter the price metric you want to plot (e.g., Close, Open, High, Low):",['Close', 'Open', 'High', 'Low'])
            start_date = st.sidebar.date_input("Enter the start date of the interval:")
            end_date = st.sidebar.date_input("Enter the end date of the interval:")

            # Plot the selected cryptocurrency coins for the specified metric within the date range
            if st.sidebar.button("Cryptocurrencies Prices Comparison"):
                plot_crypto_coins(data, coin, metric, str(start_date), str(end_date))


    elif visualization_option == 'Candlestick Chart':
#         # User input for visualization
#         coin_visualization = st.sidebar.selectbox("Select a coin for visualization:", data['Crypto'].unique(), index=0)
#         period_visualization = st.sidebar.selectbox("Select period for visualization:", ['D', 'W', 'M'], index=0)

#         # Plot the selected cryptocurrency chart
#         if st.sidebar.button("Plot Cryptocurrency Chart"):
#             plot_candlestick_chart(coin_visualization, period_visualization)
        # User input for coin and period
        available_coins = data['Crypto'].unique()
        coin = st.sidebar.selectbox("Select a coin symbol:", available_coins, index=0)
        period = st.sidebar.selectbox("Select a period:", ['Daily', 'Weekly', 'Monthly'], index=0)

        if period == 'Daily':
            period = 'D'
        elif period == 'Weekly':
            period = 'W'
        elif period == 'Monthly':
            period = 'M'

        plot_candlestick_chart(data, coin, period)


            
    elif visualization_option == 'Market State Visualization':
        visualize_market_state(data)
        
    elif visualization_option == "Predicted Highs and Lows":
        predicted_highs, predicted_lows = predict_highs_lows(data)
        
elif side_bars == "Predictions":
    st.write("You are on the Predictions page!")
    st.header("About the Prediction Data")
    st.write("The prediction data is from performing PCA to reduce dimentionality of the data with n_component of 10 and clustering the data with K-means into four clusters and selecting the best from each cluster using the centroid (You can visualize the selected data below)")
     # Display selected data
    if selected_data is not None:
        if st.button("Display Selected Data"):
            st.dataframe(selected_data)
        else:
            st.write("Click the button above to display the selected data.")
    else:
        st.write("Selected data is not available. Please check the file path and data format.")
 
    st.write("Here you can make prediction and visualize forecast for the selected coins using one or all of the avaliable models\n 1. SVR\n 2. XGBoost\n 3. Gradient Boosting\n 4. LSTM")
    display_selected(selected_data)
   
    
    
    
    

        
elif side_bars == 'NEWS':
    # Ask the user for the cryptocurrency they want to see news about
    chosen_crypto = st.text_input("Enter the cryptocurrency you want to see news about:", "Bitcoin").strip().upper()

    # Fetch and display the top cryptocurrency news stories
    get_top_crypto_news(chosen_crypto)



# elif side_bars == "Predictions":
#     predictions()
# elif side_bars == "NEWS":
#     news()
# # SOLiGence - Solent Intelligence

# Welcome to SOLiGence, a leading financial multinational organization specializing in stock and shares, savings, and investments.

# ## About Us

# At SOLiGence, we are committed to providing top-tier financial services and solutions to our clients. With our expert team and years of experience in the industry, we strive to offer the best investment opportunities, personalized advice, and comprehensive solutions to help you achieve your financial goals.

# ## Our Services

# 1. **Stock and Shares**: We offer a wide range of options for investing in the stock market. Our team of experts analyzes market trends and provides insights to help you make informed decisions.

# 2. **Savings**: We understand the importance of savings for a secure future. Our tailored savings plans ensure that you can build a financial cushion to meet unexpected expenses.

# 3. **Investments**: Whether you're a novice investor or an experienced one, we provide investment opportunities that align with your risk tolerance and financial objectives.

# ## Why Choose SOLiGence?

# - **Expertise**: Our team consists of financial experts who are well-versed in market trends and investment strategies.

# - **Personalized Approach**: We understand that every client has unique financial needs. Our solutions are customized to suit your individual goals.

# - **Transparency**: We believe in transparency in all our transactions and provide clear insights into investment performance and market developments.

# ## Contact Us

# For more information about our services or to schedule a consultation, please contact us at:

# - Website: [www.soligence.com](https://www.soligence.com)
# - Email: info@soligence.com
# - Phone: +1-123-456-7890

# Join SOLiGence today and let us help you build a brighter financial future!

