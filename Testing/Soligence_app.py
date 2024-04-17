# importing necceassary modules 
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from ta.trend import SMAIndicator
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Adding PCA import
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import mplfinance as mpf
import matplotlib.dates as mdates
import feedparser
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import tensorflow as tf
import itertools

# Load cryptocurrency price data
data = pd.read_csv('Cleaned_combined_crypto_data.csv')
selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')

# building functions for use in the app
# home page

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


# about the app

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

    
# dataset page    

def dataset():
    
    st.header("Crypto Dataset of Five Coins")
    
    # Sidebar options
    st.sidebar.subheader("Dataset Options")
    
    # Sorting
    sort_column = st.sidebar.multiselect("Sort by:", data.columns)
    ascending = st.sidebar.checkbox("Ascending")
    sorted_data = data.sort_values(by=sort_column, ascending=ascending)

    # Filtering
    selected_crypto = st.sidebar.selectbox("Filter by cryptocurrency:", data['Crypto'].unique())
    sorted_data = sorted_data[sorted_data['Crypto'].isin([selected_crypto])]

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


# plotting the average price trend    
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



# Candlestick and volumne of the data

def plot_candlestick_chart(coin='BTC-GBP', period='D'):
    # Filter the DataFrame for the selected coin
    coin_data = crypto_data[crypto_data['Crypto'] == coin].copy()
    
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
    fig = go.Figure(data=[go.Candlestick(x=resampled_data.index,
                                         open=resampled_data['Open'],
                                         high=resampled_data['High'],
                                         low=resampled_data['Low'],
                                         close=resampled_data['Close'],
                                         name='Candlestick'),
                          go.Bar(x=resampled_data.index,
                                 y=resampled_data['Volume'],
                                 name='Volume',
                                 marker_color='rgba(0, 0, 0, 0.5)')])

    fig.update_layout(title=f'{coin} {period.upper()} Chart',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)

    st.plotly_chart(fig)

# visualizing market state

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
    
# getting crypto news by scrapping        

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


# generating the four coins from through pca and clustering

def generate_selected_data():
    # Assuming 'crypto_data' is loaded and formatted correctly
    crypto_data = pd.read_csv("Cleaned_combined_crypto_data.csv", index_col="Date", parse_dates=True)
    
    # Pivot and preprocess the data
    pivoted_data = crypto_data.pivot(columns='Crypto', values='Close')
    pivoted_data.fillna(0, inplace=True)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivoted_data)

    # Perform PCA with a reasonable number of components
    pca = PCA(n_components=10)  # Adjusted for example, consider your variance ratio to decide
    pca_result = pca.fit_transform(scaled_data)

    # Compute component loadings (transposed PCA components)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(1, 11)], index=pivoted_data.columns)

    # Perform K-means clustering on the loadings
    kmeans = KMeans(n_clusters=4, random_state=0)
    cluster_labels = kmeans.fit_predict(loadings)

    # Assign cluster labels to the coins
    loadings['Cluster'] = cluster_labels

    # Select a representative coin from each cluster
    representative_coins = pd.DataFrame()
    for i in range(4):
        cluster = loadings[loadings['Cluster'] == i]
        # Calculate the distance to the cluster center
        center = kmeans.cluster_centers_[i]
        cluster['distance_to_center'] = cluster.apply(lambda x: np.linalg.norm(x[:-1] - center), axis=1)
        # Append the coin with the minimum distance to the center
        representative_coins = representative_coins.append(cluster.loc[cluster['distance_to_center'].idxmin()])

    # Extract the selected coins based on the index of representative_coins
    selected_data = pivoted_data[representative_coins.index]

    # Save the selected data to a CSV file
    selected_data.to_csv("Selected_coins.csv")

    return selected_data
selected_data = generate_selected_data()  

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


# plotting the four coins against each other to see their relationships     

def plot_coin_scatter(selected_data):
    # Get all combinations of coins
    coin_combinations = list(itertools.combinations(selected_data.columns, 2))

    # Plot scatter plots for each pair of coins
    plt.figure(figsize=(15, 10))
    for i, (coin1, coin2) in enumerate(coin_combinations, start=1):
        plt.subplot(3, 3, i)
        plt.scatter(selected_data[coin1], selected_data[coin2])
        plt.xlabel(coin1)
        plt.ylabel(coin2)
        plt.title(f"{coin1} vs {coin2}")

    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit



# making prediction from the trained and saved models
# Function to evaluate models for selected coins
def evaluate_models_selected_coin(selected_data, column_index, chosen_model='all'):
    coin_name = selected_data.columns[column_index] 

    # Add lagged features for 1 to 3 days
    for lag in range(1, 4):
        selected_data.loc[:, f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)

    # Drop rows with NaN values created due to shifting
    selected_data.dropna(inplace=True)

    # Features will be the lagged values, and the target will be the current price of the coin
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize dictionary to hold models
    models = {
        'GRADIENT BOOSTING': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGBOOST': XGBRegressor(),
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1)), Dense(units=1)])
    }

    eval_metrics = {}

    if chosen_model.lower() == 'all':
        chosen_models = models.keys()
    else:
        chosen_models = [chosen_model.upper()]  # Capitalize input for case insensitivity

    for model_name in chosen_models:
        if model_name not in models:
            st.warning(f"Model '{model_name}' not found. Skipping...")
            continue

        model = models[model_name]

        if model_name == 'LSTM':
            model_filename = f"Model_SELECTED_COIN_{column_index+1}/lstm_model.pkl"
            if os.path.exists(model_filename):
                model = load_model(model_filename)
                # Reshape the input data for LSTM model
                X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
                predictions = model.predict(X_test_array).flatten()
            else:
                st.error("No pre-trained LSTM model found.")
                return  # Skip the rest of the loop if LSTM model not found
        else:
            model.fit(X_train, y_train)  # Fit the model
            predictions = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)

        # Store evaluation metrics
        eval_metrics[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

    # Display evaluation metrics
    st.subheader(f"Evaluation Metrics for {coin_name}:")
    for model_name, metrics in eval_metrics.items():
        st.write(f"Evaluation metrics for {model_name}:")
        for metric_name, value in metrics.items():
            st.write(f"{metric_name}: {value}")
        st.write('---')

    # Plot evaluation metrics
    if eval_metrics:
        st.subheader("Evaluation Metric Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))

        metrics = list(eval_metrics.keys())
        mae_values = [eval_metrics[model]['MAE'] for model in metrics]
        mse_values = [eval_metrics[model]['MSE'] for model in metrics]
        rmse_values = [eval_metrics[model]['RMSE'] for model in metrics]

        bar_width = 0.15
        index = np.arange(len(metrics))

        bar1 = ax.bar(index - 2*bar_width, mae_values, bar_width, label='MAE')
        bar2 = ax.bar(index - bar_width, mse_values, bar_width, label='MSE')
        bar3 = ax.bar(index, rmse_values, bar_width, label='RMSE')

        ax.set_xlabel('Models')
        ax.set_ylabel('Metrics')
        ax.set_title(f'Evaluation Metrics for {coin_name} using {chosen_model.upper()} as the Models')
        ax.set_xticks(index)
        ax.set_xticklabels(metrics)
        ax.legend()

        # Annotate bars with values
        for bars in [bar1, bar2, bar3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        st.pyplot(fig)
    else:
        st.error("No models were evaluated.")

# 

def plot_actual_forecast_with_confidence(actual, predictions, periods, upper_bound, lower_bound):
    """
    Plot actual prices and forecasted prices with confidence intervals.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(periods, actual, label='Actual Price', color='g')
    plt.plot(periods, predictions, label='Forecasted Price', color='r')
    plt.fill_between(periods, lower_bound, upper_bound, color='b', alpha=0.2, label='95% Confidence Interval')
    plt.title("Actual and Forecasted Prices with Confidence Intervals")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    
# coin 1   
# Function to evaluate different models for the first coin price prediction
def evaluate_models_coin_1(selected_data):
    coin_name = selected_data.columns[0] 
    # Add lagged features for 1 to 3 days based on the selected coin
    for lag in range(1, 4):
        selected_data.loc[:, f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)

    # Drop rows with NaN values created due to shifting
    selected_data.dropna(inplace=True)

    # Features will be the lagged values, and the target will be the current price of the first coin
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize dictionary to hold models
    models = {
        'GBR': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGB': XGBRegressor(),  # Alias for XGBoost
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1)), Dense(units=1)])
    }
    # Load pre-trained models for SVR, XGBoost, and Gradient Boosting
    for model_name in ['SVR', 'XGBoost', 'Gradient Boosting']:
        model_filename = f"Model_{coin_name}/{model_name.lower().replace(' ', '_')}_model.pkl"
        if os.path.exists(model_filename):
            models[model_name] = joblib.load(model_filename)
        else:
            print(f"No pre-trained model found for {model_name}. Skipping...")

    # Initialize and train the selected model
    if model_choice in models:
        model = models[model_choice]
        if model_choice != 'LSTM':
            model.fit(X_train, y_train)
    elif model_choice == 'LSTM':
        model_filename = f"Model_{coin_name}/lstm_model.pkl"
        if os.path.exists(model_filename):
            model = tf.keras.models.load_model(model_filename)
            # Reshape the input data for LSTM model
            X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
            predictions = model.predict(X_test_array).flatten()
        else:
            print("No pre-trained LSTM model found.")
            return None, None, None, None
    else:
        print("Invalid model choice. Please choose from SVR, XGB, GBR, or LSTM.")
        return None, None, None, None

    return model, selected_data, X_test, y_test


def plot_predictions_1(model, selected_data, X_test, y_test):
    if model is not None:

        # Make predictions for the specified number of periods
        coin_name = selected_data.columns[0]  # Dynamically retrieve the coin name
        features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
        X_array = selected_data[features].to_numpy()
        predictions = model.predict(X_array[-num_periods:])  # Predictions for the last 'num_periods' rows

        # Get the last date in the dataset
        last_date = selected_data.index[-1]

        # Generate periods for future predictions
        if frequency == 'daily':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='D')
        elif frequency == 'weekly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='W')
        elif frequency == 'monthly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='M')
        elif frequency == 'quarterly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='Q')
        else:
            print("Invalid frequency. Please choose from 'daily', 'weekly', 'monthly', or 'quarterly'.")

        # Calculate mean squared error
        mse = mean_squared_error(y_test[-num_periods:], predictions)

        # Plot actual and forecasted prices with confidence intervals
        upper_bound = predictions + 1.96 * np.sqrt(mse)
        lower_bound = predictions - 1.96 * np.sqrt(mse)

        # Flatten arrays for fill_between
        upper_bound = upper_bound.flatten()
        lower_bound = lower_bound.flatten()

        # Create a DataFrame with dates and predictions
        predictions_df = pd.DataFrame({'Date': periods, 'Predictions': predictions.flatten()})


        # Display the DataFrame in Streamlit
        st.subheader("Predictions with Dates:")
        st.dataframe(predictions_df)

        # Call the function for plotting
        plot_actual_forecast_with_confidence(y_test[-num_periods:], predictions, periods, upper_bound, lower_bound)

        # Plot the time series plot with averages and confidence intervals using Streamlit's plotting functions
        st.subheader(f"Predicted Prices and Confidence Intervals for {coin_name} by {frequency}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(periods, predictions, label='Predicted Price')
        ax.fill_between(periods, lower_bound, upper_bound, color='b', alpha=0.2, label='95% Confidence Interval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
# coin 2
# Function to evaluate different models for the second coin price prediction
def evaluate_models_coin_2(selected_data):
    coin_name = selected_data.columns[1]  # Dynamically get the name of the first column
    # Add lagged features for 1 to 3 days
    for lag in range(1, 4):
        selected_data.loc[:, f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)

    # Drop rows with NaN values created due to shifting
    selected_data.dropna(inplace=True)

    # Features will be the lagged values, and the target will be the current price of the first coin
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize dictionary to hold models
    models = {
        'GBR': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGB': XGBRegressor(),  # Alias for XGBoost
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1)), Dense(units=1)])
    }
    # Load pre-trained models for SVR, XGBoost, and Gradient Boosting
    for model_name in ['SVR', 'XGBoost', 'Gradient Boosting']:
        model_filename = f"Model_SELECTED_COIN_2/{model_name.lower().replace(' ', '_')}_model.pkl"
        if os.path.exists(model_filename):
            models[model_name] = joblib.load(model_filename)
        else:
            print(f"No pre-trained model found for {model_name}. Skipping...")

    # User input for selecting the model
    model_choice = st.sidebar.selectbox("Choose the model you want to evaluate:", ['Gradient Boosting', 'SVR', 'XGBoost', 'LSTM'])

    # Initialize and train the selected model
    if model_choice in models:
        model = models[model_choice]
        if model_choice != 'LSTM':
            model.fit(X_train, y_train)
    elif model_choice == 'LSTM':
        model_filename = "Model_SELECTED_COIN_2/lstm_model.pkl"
        if os.path.exists(model_filename):
            model = tf.keras.models.load_model(model_filename)
            # Reshape the input data for LSTM model
            X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
            predictions = model.predict(X_test_array).flatten()
        else:
            print("No pre-trained LSTM model found.")
            return None, None, None, None
    else:
        print("Invalid model choice. Please choose from SVR, XGB, GBR, or LSTM.")
        return None, None, None, None

    return model, selected_data, X_test, y_test

# Function to plot predictions and confidence intervals
def plot_predictions_2(model, selected_data, X_test, y_test):
    if model is not None:
        # User input for frequency and number of periods (weeks, months, or quarters)
        frequency = input("Enter the frequency (daily, weekly, monthly, quarterly): ").lower()
        num_periods = int(input("Enter the number of periods: "))

        # Make predictions for the specified number of periods
        coin_name = selected_data.columns[1]  # Dynamically retrieve the coin name
        features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
        X_array = selected_data[features].to_numpy()
        predictions = model.predict(X_array[-num_periods:])  # Predictions for the last 'num_periods' rows

        # Get the last date in the dataset
        last_date = selected_data.index[-1]

        # Generate periods for future predictions
        if frequency == 'daily':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='D')
        elif frequency == 'weekly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='W')
        elif frequency == 'monthly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='M')
        elif frequency == 'quarterly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='Q')
        else:
            print("Invalid frequency. Please choose from 'daily', 'weekly', 'monthly', or 'quarterly'.")

        # Calculate mean squared error
        mse = mean_squared_error(y_test[-num_periods:], predictions)

        # Plot actual and forecasted prices with confidence intervals
        upper_bound = predictions + 1.96 * np.sqrt(mse)
        lower_bound = predictions - 1.96 * np.sqrt(mse)

        # Flatten arrays for fill_between
        upper_bound = upper_bound.flatten()
        lower_bound = lower_bound.flatten()

        # Create a DataFrame with dates and predictions
        predictions_df = pd.DataFrame({'Date': periods, 'Predictions': predictions.flatten()})

        # Display the DataFrame in Streamlit
        st.subheader("Predictions with Dates:")
        st.dataframe(predictions_df)

        # Call the function for plotting
        plot_actual_forecast_with_confidence(y_test[-num_periods:], predictions, periods, upper_bound, lower_bound)

        # Plot the time series plot with averages and confidence intervals using Streamlit's plotting functions
        st.subheader(f"Predicted Prices and Confidence Intervals for {coin_name} by {frequency}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(periods, predictions, label='Predicted Price')
        ax.fill_between(periods, lower_bound, upper_bound, color='b', alpha=0.2, label='95% Confidence Interval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# coin 3
# Function to evaluate different models for the third coin price prediction
def evaluate_models_coin_3(selected_data):
    coin_name = selected_data.columns[2]  # Dynamically get the name of the first column
    # Add lagged features for 1 to 3 days
    for lag in range(1, 4):
        selected_data.loc[:, f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)

    # Drop rows with NaN values created due to shifting
    selected_data.dropna(inplace=True)

    # Features will be the lagged values, and the target will be the current price of the first coin
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize dictionary to hold models
    models = {
        'GBR': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGB': XGBRegressor(),  # Alias for XGBoost
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1)), Dense(units=1)])
    }
    # Load pre-trained models for SVR, XGBoost, and Gradient Boosting
    for model_name in ['SVR', 'XGBoost', 'Gradient Boosting']:
        model_filename = f"Model_SELECTED_COIN_3/{model_name.lower().replace(' ', '_')}_model.pkl"
        if os.path.exists(model_filename):
            models[model_name] = joblib.load(model_filename)
        else:
            print(f"No pre-trained model found for {model_name}. Skipping...")

    # User input for selecting the model
    model_choice = st.sidebar.selectbox("Choose the model you want to evaluate:", ['Gradient Boosting', 'SVR', 'XGBoost', 'LSTM'])

    # Initialize and train the selected model
    if model_choice in models:
        model = models[model_choice]
        if model_choice != 'LSTM':
            model.fit(X_train, y_train)
    elif model_choice == 'LSTM':
        model_filename = "Model_SELECTED_COIN_3/lstm_model.pkl"
        if os.path.exists(model_filename):
            model = tf.keras.models.load_model(model_filename)
            # Reshape the input data for LSTM model
            X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
            predictions = model.predict(X_test_array).flatten()
        else:
            print("No pre-trained LSTM model found.")
            return None, None, None, None
    else:
        print("Invalid model choice. Please choose from SVR, XGB, GBR, or LSTM.")
        return None, None, None, None

    return model, selected_data, X_test, y_test

# Function to plot predictions and confidence intervals
def plot_predictions_3(model, selected_data, X_test, y_test):
    if model is not None:
        # User input for frequency and number of periods (weeks, months, or quarters)
        frequency = input("Enter the frequency (daily, weekly, monthly, quarterly): ").lower()
        num_periods = int(input("Enter the number of periods: "))

        # Make predictions for the specified number of periods
        coin_name = selected_data.columns[2]  # Dynamically retrieve the coin name
        features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
        X_array = selected_data[features].to_numpy()
        predictions = model.predict(X_array[-num_periods:])  # Predictions for the last 'num_periods' rows

        # Get the last date in the dataset
        last_date = selected_data.index[-1]

        # Generate periods for future predictions
        if frequency == 'daily':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='D')
        elif frequency == 'weekly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='W')
        elif frequency == 'monthly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='M')
        elif frequency == 'quarterly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='Q')
        else:
            print("Invalid frequency. Please choose from 'daily', 'weekly', 'monthly', or 'quarterly'.")

        # Calculate mean squared error
        mse = mean_squared_error(y_test[-num_periods:], predictions)

        # Plot actual and forecasted prices with confidence intervals
        upper_bound = predictions + 1.96 * np.sqrt(mse)
        lower_bound = predictions - 1.96 * np.sqrt(mse)

        # Flatten arrays for fill_between
        upper_bound = upper_bound.flatten()
        lower_bound = lower_bound.flatten()

        # Create a DataFrame with dates and predictions
        predictions_df = pd.DataFrame({'Date': periods, 'Predictions': predictions.flatten()})

        # Display the DataFrame in Streamlit
        st.subheader("Predictions with Dates:")
        st.dataframe(predictions_df)

        # Call the function for plotting
        plot_actual_forecast_with_confidence(y_test[-num_periods:], predictions, periods, upper_bound, lower_bound)

        # Plot the time series plot with averages and confidence intervals using Streamlit's plotting functions
        st.subheader(f"Predicted Prices and Confidence Intervals for {coin_name} by {frequency}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(periods, predictions, label='Predicted Price')
        ax.fill_between(periods, lower_bound, upper_bound, color='b', alpha=0.2, label='95% Confidence Interval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# coin 4 
# Function to evaluate different models for the first coin price prediction
def evaluate_models_coin_4(selected_data):
    coin_name = selected_data.columns[3]  # Dynamically get the name of the first column
    # Add lagged features for 1 to 3 days
    for lag in range(1, 4):
        selected_data.loc[:, f'{coin_name}_lag_{lag}'] = selected_data[coin_name].shift(lag)

    # Drop rows with NaN values created due to shifting
    selected_data.dropna(inplace=True)

    # Features will be the lagged values, and the target will be the current price of the first coin
    features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[coin_name]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize dictionary to hold models
    models = {
        'GBR': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGB': XGBRegressor(),  # Alias for XGBoost
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1)), Dense(units=1)])
    }
    # Load pre-trained models for SVR, XGBoost, and Gradient Boosting
    for model_name in ['SVR', 'XGBoost', 'Gradient Boosting']:
        model_filename = f"Model_SELECTED_COIN_4/{model_name.lower().replace(' ', '_')}_model.pkl"
        if os.path.exists(model_filename):
            models[model_name] = joblib.load(model_filename)
        else:
            print(f"No pre-trained model found for {model_name}. Skipping...")

    # User input for selecting the model
    model_choice = st.sidebar.selectbox("Choose the model you want to evaluate:", ['Gradient Boosting', 'SVR', 'XGBoost', 'LSTM'])

    # Initialize and train the selected model
    if model_choice in models:
        model = models[model_choice]
        if model_choice != 'LSTM':
            model.fit(X_train, y_train)
    elif model_choice == 'LSTM':
        model_filename = "Model_SELECTED_COIN_4/lstm_model.pkl"
        if os.path.exists(model_filename):
            model = tf.keras.models.load_model(model_filename)
            # Reshape the input data for LSTM model
            X_test_array = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)
            predictions = model.predict(X_test_array).flatten()
        else:
            print("No pre-trained LSTM model found.")
            return None, None, None, None
    else:
        print("Invalid model choice. Please choose from SVR, XGB, GBR, or LSTM.")
        return None, None, None, None

    return model, selected_data, X_test, y_test

# Function to plot predictions and confidence intervals
def plot_predictions_4(model, selected_data, X_test, y_test):
    if model is not None:
        

        # Make predictions for the specified number of periods
        coin_name = selected_data.columns[3]  # Dynamically retrieve the coin name
        features = [f'{coin_name}_lag_{lag}' for lag in range(1, 4)]
        X_array = selected_data[features].to_numpy()
        predictions = model.predict(X_array[-num_periods:])  # Predictions for the last 'num_periods' rows

        # Get the last date in the dataset
        last_date = selected_data.index[-1]

        # Generate periods for future predictions
        if frequency == 'daily':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='D')
        elif frequency == 'weekly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='W')
        elif frequency == 'monthly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='M')
        elif frequency == 'quarterly':
            periods = pd.date_range(start=last_date, periods=num_periods, freq='Q')
        else:
            print("Invalid frequency. Please choose from 'daily', 'weekly', 'monthly', or 'quarterly'.")

        # Calculate mean squared error
        mse = mean_squared_error(y_test[-num_periods:], predictions)

        # Plot actual and forecasted prices with confidence intervals
        upper_bound = predictions + 1.96 * np.sqrt(mse)
        lower_bound = predictions - 1.96 * np.sqrt(mse)

        # Flatten arrays for fill_between
        upper_bound = upper_bound.flatten()
        lower_bound = lower_bound.flatten()

        # Create a DataFrame with dates and predictions
        predictions_df = pd.DataFrame({'Date': periods, 'Predictions': predictions.flatten()})


        # Display the DataFrame in Streamlit
        st.subheader("Predictions with Dates:")
        st.dataframe(predictions_df)

        # Call the function for plotting within Streamlit context
        plot_actual_forecast_with_confidence(y_test[-num_periods:], predictions, periods, upper_bound, lower_bound)


        # Plot the time series plot with averages and confidence intervals using Streamlit's plotting functions
        st.subheader(f"Predicted Prices and Confidence Intervals for {coin_name} by {frequency}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(periods, predictions, label='Predicted Price')
        ax.fill_between(periods, lower_bound, upper_bound, color='b', alpha=0.2, label='95% Confidence Interval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


# predicting Buy and selling
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import SMAIndicator
from datetime import datetime, timedelta

# Function to apply MA Trading Strategy and display chart
def determine_best_time_to_trade_future(selected_data, chosen_coin, num_days):
    # Apply MA trading strategy
    total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)

    # Forecast future price
    future_price, future_date = forecast_price(selected_data, chosen_coin, num_days)

    if future_price is not None:
        action = "Buy" if future_price > selected_data[chosen_coin].iloc[-1] else "Sell"
        st.write(f"Recommended action: {action}")
        st.write(f"Forecasted price for {future_date.date()}: {future_price}")
    else:
        st.write("Unable to forecast price for the future.")

    return total_profit_loss

# Function to apply MA Trading Strategy and display chart
def apply_ma_trading_strategy(selected_data, chosen_coin):
    # Drop rows with missing 'Close' prices
    selected_data.dropna(subset=[chosen_coin], inplace=True)

    # Convert index to offset-naive datetime index
    selected_data.index = selected_data.index.tz_localize(None)

    # Calculate moving averages
    ma_7 = 7  # 7-day moving average
    ma_14 = 14  # 14-day moving average
    selected_data[f'MA_{ma_7}'] = SMAIndicator(close=selected_data[chosen_coin], window=ma_7).sma_indicator()
    selected_data[f'MA_{ma_14}'] = SMAIndicator(close=selected_data[chosen_coin], window=ma_14).sma_indicator()

    # Generate buy and sell signals based on moving averages
    selected_data['Buy_Signal'] = np.where(selected_data[f'MA_{ma_7}'] > selected_data[f'MA_{ma_14}'].shift(1), 1, 0)
    selected_data['Sell_Signal'] = np.where(selected_data[f'MA_{ma_7}'] < selected_data[f'MA_{ma_14}'].shift(1), -1, 0)

    # Create plotly figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[chosen_coin], name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_7}'], name=f'{ma_7}-day MA', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_14}'], name=f'{ma_14}-day MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Buy_Signal'] == 1].index, y=selected_data[selected_data['Buy_Signal'] == 1][chosen_coin],
                             mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Sell_Signal'] == -1].index, y=selected_data[selected_data['Sell_Signal'] == -1][chosen_coin],
                             mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'))
    
    # Current price line
    current_price = selected_data[chosen_coin].iloc[-1]
    fig.add_trace(go.Scatter(x=[selected_data.index[0], selected_data.index[-1]], y=[current_price, current_price],
                             mode='lines', line=dict(color='gray', dash='dash'), name='Current Price'))

    # Update layout
    fig.update_layout(
        title=f'Moving Average Trading Strategy for {chosen_coin}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1000,
        height=600
    )

    # Display plotly figure
    st.plotly_chart(fig)

    st.write(f"Current Price of {chosen_coin} is {current_price}")

    return selected_data

def forecast_price(selected_data, chosen_coin, num_days):
    future_date = datetime.now() + timedelta(days=num_days)
    future_date = future_date.replace(tzinfo=None)  # Remove timezone information

    # Convert index datetime to naive datetime
    index_datetime = selected_data.index[-1].to_pydatetime().replace(tzinfo=None)

    if future_date > index_datetime:
        # Extend the index
        new_index = pd.date_range(start=selected_data.index[0], periods=len(selected_data) + num_days, freq='D')
        extended_data = selected_data.reindex(new_index, method='ffill')

        # Recalculate MAs for the extended period
        ma_7 = 7
        ma_14 = 14
        extended_data[f'MA_{ma_7}'] = SMAIndicator(close=extended_data[chosen_coin], window=ma_7).sma_indicator()
        extended_data[f'MA_{ma_14}'] = SMAIndicator(close=extended_data[chosen_coin], window=ma_14).sma_indicator()

        # Forecast using the latest MA values
        future_price = (extended_data[f'MA_{ma_7}'].iloc[-1] + extended_data[f'MA_{ma_14}'].iloc[-1]) / 2
        return future_price, future_date
    else:
        st.write("Future date is within the available data range.")
        return None, None





# ML on trading strategy

import streamlit as st
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
import joblib
import os
from datetime import datetime, timedelta
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

# Function to apply MA Trading Strategy and display chart
def apply_ma_trading_strategy(selected_data, chosen_coin):
    # Drop rows with missing 'Close' prices
    selected_data.dropna(subset=[chosen_coin], inplace=True)

    # Calculate moving averages
    ma_7 = 7
    ma_14 = 14
    selected_data[f'MA_{ma_7}'] = SMAIndicator(close=selected_data[chosen_coin], window=ma_7).sma_indicator()
    selected_data[f'MA_{ma_14}'] = SMAIndicator(close=selected_data[chosen_coin], window=ma_14).sma_indicator()

    # Generate buy and sell signals based on moving averages
    selected_data['Buy_Signal'] = np.where(selected_data[f'MA_{ma_7}'] > selected_data[f'MA_{ma_14}'].shift(1), 1, 0)
    selected_data['Sell_Signal'] = np.where(selected_data[f'MA_{ma_7}'] < selected_data[f'MA_{ma_14}'].shift(1), -1, 0)

    # Display plotly figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[chosen_coin], name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_7}'], name=f'{ma_7}-day MA', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data[f'MA_{ma_14}'], name=f'{ma_14}-day MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Buy_Signal'] == 1].index, y=selected_data[selected_data['Buy_Signal'] == 1][chosen_coin],
                             mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=selected_data[selected_data['Sell_Signal'] == -1].index, y=selected_data[selected_data['Sell_Signal'] == -1][chosen_coin],
                             mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'))
    
    # Current price line
    current_price = selected_data[chosen_coin].iloc[-1]
    fig.add_trace(go.Scatter(x=[selected_data.index[0], selected_data.index[-1]], y=[current_price, current_price],
                             mode='lines', line=dict(color='gray', dash='dash'), name='Current Price'))

    # Update layout
    fig.update_layout(
        title=f'Moving Average Trading Strategy for {chosen_coin}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=1000,
        height=600
    )

    # Display plotly figure
    st.plotly_chart(fig)

    st.write(f"Current Price of {chosen_coin} is {current_price}")

    return selected_data

# Function to forecast future price using the SVR model
def forecast_price_svr(selected_data, chosen_coin, num_days):
    coin_index = selected_data.columns.get_loc(chosen_coin) + 1
    model_filename = f"Model_SELECTED_COIN_{coin_index}/svr_model.pkl"
    if not os.path.exists(model_filename):
        st.write(f"No pre-trained SVR model found for {chosen_coin}.")
        return None, None

    # Load the SVR model
    model = joblib.load(model_filename)

    # Prepare input data for prediction
    features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
    X_array = selected_data[features].to_numpy()
    X_today = X_array[-1].reshape(1, -1)

    # Predict future price using the SVR model
    future_price = model.predict(X_today)[0]
    future_date = datetime.now() + timedelta(days=num_days)

    return future_price, future_date

# Function to forecast future price using the GBR model
def forecast_price_gbr(selected_data, chosen_coin, num_days):
    coin_index = selected_data.columns.get_loc(chosen_coin) + 1
    model_filename = f"Model_SELECTED_COIN_{coin_index}/gradient_boosting_model.pkl"
    if not os.path.exists(model_filename):
        st.write(f"No pre-trained GBR model found for {chosen_coin}.")
        return None, None

    # Load the GBR model
    model = joblib.load(model_filename)

    # Prepare input data for prediction
    features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
    X_array = selected_data[features].to_numpy()
    X_today = X_array[-1].reshape(1, -1)

    # Predict future price using the GBR model
    future_price = model.predict(X_today)[0]
    future_date = datetime.now() + timedelta(days=num_days)

    return future_price, future_date

# Function to forecast future price using the XGBoost model
def forecast_price_xgboost(selected_data, chosen_coin, num_days):
    coin_index = selected_data.columns.get_loc(chosen_coin) + 1
    model_filename = f"Model_SELECTED_COIN_{coin_index}/xgboost_model.pkl"
    if not os.path.exists(model_filename):
        st.write(f"No pre-trained XGBoost model found for {chosen_coin}.")
        return None, None

    # Load the XGBoost model
    model = joblib.load(model_filename)

    # Prepare input data for prediction
    features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
    X_array = selected_data[features].to_numpy()
    X_today = X_array[-1].reshape(1, -1)

    # Predict future price using the XGBoost model
    future_price = model.predict(X_today)[0]
    future_date = datetime.now() + timedelta(days=num_days)

    return future_price, future_date

# Function to forecast future price using the LSTM model
def forecast_price_lstm(selected_data, chosen_coin, num_days):
    coin_index = selected_data.columns.get_loc(chosen_coin) + 1
    model_filename = f"Model_SELECTED_COIN_{coin_index}/lstm_model.hpkl"
    if not os.path.exists(model_filename):
        st.write(f"No pre-trained LSTM model found for {chosen_coin}.")
        return None, None

    # Load the LSTM model
    model = load_model(model_filename)

    # Prepare input data for prediction
    features = [f'{chosen_coin}_lag_{lag}' for lag in range(1, 4)]
    X_array = selected_data[features].to_numpy()
    X_today = X_array[-1].reshape(1, 3, 1)  # LSTM expects input shape (batch_size, timesteps, input_dim)

    # Predict future price using the LSTM model
    future_price = model.predict(X_today)[0][0]
    future_date = datetime.now() + timedelta(days=num_days)

    return future_price, future_date

# Function to determine the best time to trade based on model predictions
def determine_best_time_to_trade(selected_data, chosen_coin, num_days, model):
    if model == "SVR":
        total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)
        future_price, future_date = forecast_price_svr(selected_data, chosen_coin, num_days)
    elif model == "GBR":
        total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)
        future_price, future_date = forecast_price_gbr(selected_data, chosen_coin, num_days)
    elif model == "XGBoost":
        total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)
        future_price, future_date = forecast_price_xgboost(selected_data, chosen_coin, num_days)
    elif model == "LSTM":
        total_profit_loss = apply_ma_trading_strategy(selected_data, chosen_coin)
        future_price, future_date = forecast_price_lstm(selected_data, chosen_coin, num_days)
    else:
        st.write("Invalid model selection.")
        return None

    if future_price is not None:
        action = "Buy" if future_price > selected_data[chosen_coin].iloc[-1] else "Sell"
        st.write(f"Recommended action for {chosen_coin}: {action}")
        st.write(f"Forecasted price for {future_date.date()}: {future_price}")
    else:
        st.write(f"Unable to forecast price for {chosen_coin} in the future.")

    return total_profit_loss


# prediction coin based on profit and num of days
# import os
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

def find_best_coins(selected_data_path):
    # Load selected data
    selected_data = pd.read_csv(selected_data_path, index_col="Date")

    # Extract coins from column names
    coins = selected_data.columns[:]  # Assuming the first column is not a coin name

    # Define model folders
    model_folders = {
        "xgboost": "XGB_models",
        "gradient_bossting": "GBR_models",
        "lstm": "LSTM_models",
        "svr": "SVR_models"
    }

    # Collect user input for model type
    model_type = st.selectbox("Select the model type:", list(model_folders.keys()))

    # Load the saved models for the selected model type
    models = {}
    for i, coin in enumerate(coins, start=1):
        model_folder = model_folders[model_type]
        model_file = os.path.join(model_folder, f"{model_type.lower()}_model_{i}.pkl")
        models[coin] = joblib.load(model_file)

    # Collect user input for profit and number of days
    desired_profit = st.number_input("Enter the desired profit amount:")
    num_days = st.number_input("Enter the number of days:", value=1, step=1)

    # Initialize variables to track the closest and next best coins and profits
    closest_coin = None
    closest_profit = None
    next_best_coin = None
    next_best_profit = None

    # Iterate through the coins
    for coin, model in models.items():
        # Reshape the input data to match the shape of the training data
        input_data = np.array([[num_days, 0, 0]])  # Assuming the second and third features are placeholders
        # Predict the price change for the specified number of days
        price_change = model.predict(input_data)[0]
        # Calculate the potential profit
        potential_profit = price_change * desired_profit
        
        # Update closest and next best coins and profits
        if closest_coin is None or abs(potential_profit - desired_profit) < abs(closest_profit - desired_profit):
            next_best_coin = closest_coin
            next_best_profit = closest_profit
            closest_coin = coin
            closest_profit = potential_profit
        elif next_best_coin is None or abs(potential_profit - desired_profit) < abs(next_best_profit - desired_profit):
            next_best_coin = coin
            next_best_profit = potential_profit

    
    # # Display the closest and next best coins and their potential profits
    # st.subheader("Results:")
    # st.write(f"The closest coin to yield the desired profit of {desired_profit} in {num_days} days is: {closest_coin}")
    # st.write(f"The potential profit for {closest_coin} is: {closest_profit}")
    # st.write(f"The next best coin is: {next_best_coin}")
    # st.write(f"The potential profit for {next_best_coin} is: {next_best_profit}")
    
    # Display the closest and next best coins and their potential profits
    st.subheader("Results:")
    if closest_coin:
        st.write(f"The closest coin to yield the desired profit of {desired_profit} GBP in {num_days} days is: {closest_coin}")
        st.write(f"The potential profit for {closest_coin} is: {closest_profit}")
    else:
        st.write("No results found for the given parameters.")
    if next_best_coin:
        st.write(f"The next best coin is: {next_best_coin}")
        st.write(f"The potential profit for {next_best_coin} is: {next_best_profit}")




# App layout
    
# Sidebar navigation
side_bars = st.sidebar.radio("Navigation", ["Home", "About Us", "Dataset","Coin Correlation","Moving Average", "Visualizations","Predictions","NEWS"])


# Condition for sidebar navigation
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
    st.header(Moving Average Analysis)
    st.write("Interpretation:\n1.**Short MA:** Provides insights into short-term price trends.\n2. **Medium MA:** Offers indications of intermediate-term price movements.\ 3' **Long MA:** Helps identify long-term trends in the cryptocurrency market.
")
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
        crypto_data = pd.read_csv("Cleaned_combined_crypto_data.csv", index_col='Date')
        # User input for coin and period
        available_coins = crypto_data['Crypto'].unique()
        coin = st.selectbox("Select coin", available_coins)
        period_labels = ("Daily", "Weekly", "Monthly")
        period_values = ('D', 'W', 'M')
        period_index = st.radio("Select period", period_labels)
        period = period_values[period_labels.index(period_index)]

        plot_candlestick_chart(coin, period)      
    elif visualization_option == 'Market State Visualization':
        visualize_market_state(data)
        
    elif visualization_option == "Predicted Highs and Lows":
        predicted_highs, predicted_lows = predict_highs_lows(data)
        
elif side_bars == "Predictions":
    prediction_selection = st.sidebar.radio('Selection:', ["Dataset", "Training Model Metrics", "Prediction Graphs","Buy and Sell Prediction","Predict coin by Profit"])
    st.header("Prediction of Cryptocurrency Price")
    if prediction_selection == "Dataset":
        st.subheader("About the Prediction Data")
        st.write("The prediction data is from performing PCA to reduce dimensionality of the data with n_component of 10 and clustering the data with K-means into four clusters and selecting the best from each cluster using the centroid (You can visualize the selected data below)")
        # Display selected data
        if selected_data is not None:
            if st.button("Display Selected Data"):
                st.dataframe(selected_data)
                # plot_coin_scatter(selected_data)
            else:
                st.write("Click the button above to display the selected data.")
        else:
            st.write("Selected data is not available. Please check the file path and data format.")
        st.write("Here you can make prediction and visualize forecast for the selected coins using one or all of the available models:\n 1. SVR\n 2. XGBoost\n 3. Gradient Boosting\n 4. LSTM")
        display_selected(selected_data)
        # Display the scatter plot in Streamlit
        st.title("Coin Scatter Plot")
        plot_coin_scatter(selected_data)
    elif prediction_selection == "Training Model Metrics":
        st.write("### Model Selection:/n Choose a machine learning model from the sidebar and enter the required parameters.")

        st.subheader("Available Models: ")
        st.write("1.**Support Vector Regression (SVR)**\n2. **Gradient Boosting Regression (GBR)**\n3. **XGBoost**\n4. **Long Short-Term Memory (LSTM)**")
        # Load the selected data from a CSV file
        selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')
        # Streamlit app
        st.title("Model Evaluation for Selected Coins")
        # Select the coins to evaluate
        coins = st.multiselect("Choose the coins you want to evaluate:", selected_data.columns)
        # Select the model
        chosen_model = st.selectbox("Choose the model you want to evaluate:", ['all', 'Gradient Boosting', 'SVR', 'XGBoost', 'LSTM'])
        # Loop through selected coins and evaluate models
        for coin in coins:
            st.subheader(f"Evaluation for {coin}")
            coin_index = selected_data.columns.get_loc(coin)
            evaluate_models_selected_coin(selected_data, coin_index, chosen_model)
     
    elif prediction_selection == "Prediction Graphs":
        st.title('Cryptocurrency Price Prediction')

        # Read data
        selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')

        # Select coin for prediction
        selected_coin = st.selectbox('Select a coin for prediction:', selected_data.columns.tolist())

        # User input for selecting the model
        model_choice = st.sidebar.selectbox("Choose the model you want to evaluate:", ['GBR', 'SVR', 'XGB', 'LSTM'])
        # User input for frequency and number of periods
        frequency = st.sidebar.selectbox("Select Frequency", ['daily', 'weekly', 'monthly', 'quarterly'])
        num_periods = st.sidebar.number_input("Enter Number of Periods", min_value=1, value=20)

        # Button to trigger prediction
        if st.button('Load Model and Predict'):
            # Evaluate the model based on the selected coin
            if selected_coin == selected_data.columns[0]:
                model, selected_data, X_test, y_test = evaluate_models_coin_1(selected_data)
                plot_predictions_1(model, selected_data, X_test, y_test)
                
            elif selected_coin == selected_data.columns[1]:
                model, selected_data, X_test, y_test = evaluate_models_coin_2(selected_data)
                plot_predictions_2(model, selected_data, X_test, y_test)
                
            elif selected_coin == selected_data.columns[2]:
                model, selected_data, X_test, y_test = evaluate_models_coin_3(selected_data)
                plot_predictions_3(model, selected_data, X_test, y_test)
        
            elif selected_coin == selected_data.columns[3]:
                model, selected_data, X_test, y_test = evaluate_models_coin_4(selected_data)
                plot_predictions_4(model, selected_data, X_test, y_test)
                
            else:
                st.error("Invalid selection. Please choose a valid coin.")


            # Plot predictions and confidence intervals
            fig = plt.figure(figsize=(10, 6))
            # plot_predictions(model, selected_data, X_test, y_test)
            st.pyplot(fig)
        else:
            st.write("Click the button above to display the visualization of the prediction of the selected coin.")
    
    elif prediction_selection == "Buy and Sell Prediction":
        
        buy_and_sell = st.sidebar.radio("Selection: ",["Moving Averages","Models"])
        if buy_and_sell == "Moving Averages":
            st.title('Crypto Trading Advisor')
            st.title("Using Moving Average Trading Strategy")
            
            st.write("Here, you can explore predictive models that aim to forecast optimal times to buy and sell assets in financial markets.")
            # Explanation of Buy/Sell Signal
            st.subheader("Explanation of Buy/Sell Signal:")
            st.write("**Buy Signal**: Indicators or conditions suggesting it may be a good time to purchase an asset, "
                     "anticipating its price will increase.")
            st.write("**Sell Signal**: Indicators or conditions suggesting it may be a good time to sell an asset, "
                     "anticipating its price will decrease.")

            # Sidebar for selecting coin and input for number of days

            chosen_coin = st.sidebar.selectbox("Choose the coins you want to evaluate:", selected_data.columns)
            num_days = st.sidebar.number_input("Enter the number of days for forecasting ahead:", min_value=1, step=1)

            # Button to apply MA Trading Strategy
            if st.sidebar.button("Apply MA Trading Strategy"):
                # Call function to apply MA Trading Strategy and display chart
                result_data = determine_best_time_to_trade_future(selected_data, chosen_coin, num_days)

                st.subheader("Buy/Sell Prediction Data")
                st.write(result_data)   
        elif buy_and_sell == "Models":
            # Load data
            selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')

            # Streamlit app
            st.title("Crypto Trading Analysis")

            # Sidebar - Model selection and user input
            model_selection = st.sidebar.radio("Select Model", ("Support Vector Regression (SVR)", "Gradient Boosting Regression (GBR)", "XGBoost", "LSTM"))
            chosen_coin = st.sidebar.text_input("Enter the coin you want to analyze:")
            num_days = st.sidebar.number_input("Enter the number of days for forecasting:", min_value=1, value=10)
            # Generate lagged features for each coin
            for coin_column in selected_data.columns[:]:
                for lag in range(1, 4):
                    selected_data[f'{coin_column}_lag_{lag}'] = selected_data[coin_column].shift(lag)

            if st.button("Run Analysis"):
                if model_selection == "Support Vector Regression (SVR)":
                    determine_best_time_to_trade(selected_data, chosen_coin, num_days, "SVR")
                elif model_selection == "Gradient Boosting Regression (GBR)":
                    determine_best_time_to_trade(selected_data, chosen_coin, num_days, "GBR")
                elif model_selection == "XGBoost":
                    determine_best_time_to_trade(selected_data, chosen_coin, num_days, "XGBoost")
                elif model_selection == "LSTM":
                    determine_best_time_to_trade(selected_data, chosen_coin, num_days, "LSTM")
    elif prediction_selection == "Predict coin by Profit":
        st.title("Profit Prediction Tool")
        st.header("How it Works")
        st.write("""
        1. **Select Model Type**: Choose the type of model you want to use for prediction. You can choose from XGBoost, Gradient Boosting, LSTM, or SVR.

        2. **Enter Parameters**: Input the desired profit amount and the number of days you're willing to wait for the profit.

        3. **Get Results**: The tool will analyze the data and provide you with the closest coin to reach your desired profit within the specified time frame. It will also suggest the next best coin for investment.
        """)
        selected_data_path = "Selected_coins.csv"
        find_best_coins(selected_data_path)


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

