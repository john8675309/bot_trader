#!/usr/bin/env python3
import os
import yfinance as yf
import pandas as pd
import time
import threading
import alpaca_trade_api as tradeapi
import cmd
from datetime import datetime, timedelta
import asyncio
import sqlite3
import appdirs
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Alpaca API credentials
API_KEY = ''
API_SECRET = ''
BASE_URL = 'https://paper-api.alpaca.markets'


# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

analyzer = SentimentIntensityAnalyzer()

# Event to control the news thread
stop_event = threading.Event()

# Determine the user's config directory
appname = "TradingBot"
appauthor = "John Hass"
config_dir = appdirs.user_config_dir(appname, appauthor)

# Ensure the config directory exists
os.makedirs(config_dir, exist_ok=True)

# Path to the SQLite database
db_path = os.path.join(config_dir, 'trading_bot.db')

# Function to initialize the SQLite database
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a table for storing transactions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            ticker TEXT,
            basis REAL,
            quantity INTEGER,
            purchase_date TEXT,
            status TEXT
        )
    ''')
    
    # Create a table for monitoring positions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            basis REAL,
            quantity INTEGER,
            last_checked_price REAL,
            last_checked_date TEXT
        )
    ''')

    # Create a table for storing recommendations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            ticker TEXT PRIMARY KEY,
            recommendation TEXT,
            price REAL,
            change_percent REAL,
            volume INTEGER
        )
    ''')

    # Create a table for storing symbols
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS symbols (
            cik INTEGER PRIMARY KEY,
            ticker TEXT,
            title TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()
def is_asset_active(ticker):
    try:
        asset = api.get_asset(ticker)
        return asset.tradable
    except Exception as e:
        print(f"Error checking asset status for {ticker}: {e}")
        return False

def get_open_orders():
    try:
        orders = api.list_orders(status='open')
        open_tickers = {order.symbol for order in orders}
        print(f"Open orders: {open_tickers}")  # Debug print
        return open_tickers
    except Exception as e:
        print(f"Error fetching open orders: {e}")
        return set()



def get_market_status():
    try:
        clock = api.get_clock()
        return clock.timestamp, clock.is_open
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")
        return None, None


def get_current_time():
    try:
        clock = api.get_clock()
        return clock.timestamp
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")
        return None


def download_symbols():
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for item in data.values():
            cursor.execute('''
                INSERT OR REPLACE INTO symbols (cik, ticker, title)
                VALUES (?, ?, ?)
            ''', (item['cik_str'], item['ticker'], item['title']))
        conn.commit()
        conn.close()
        print("Symbols downloaded and stored successfully.")
    else:
        print(f"Failed to download symbols. HTTP Status Code: {response.status_code}")



def download_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        data['Symbol'] = ticker
        data['CompanyName'] = stock.info.get('longName', ticker)
        return data
    except Exception as e:
        print(f"Could not download data for {ticker}: {e}")
        return None



# Function to calculate moving averages
def calculate_moving_averages(df, window_sizes=[20, 50, 200]):
    moving_averages = {}
    if not df.empty:
        for window in window_sizes:
            moving_averages[f'{window}_day'] = df['Close'].rolling(window=window).mean()
    return moving_averages

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def identify_trends(ticker, df, moving_averages, rsi, macd, signal):
    recommendations = []
    buy_signals = 0
    sell_signals = 0

    if moving_averages:
        try:
            company_name = df['CompanyName'].iloc[0]
            current_price = df['Close'].iloc[-1]
            change_percent = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100

            recommendations.append(f"Company: {company_name}")
            recommendations.append(f"Price: ${current_price:.2f}")
            recommendations.append(f"% Change: {change_percent:.2f}%")
            
            # Moving averages
            if moving_averages['20_day'].iloc[-1] > moving_averages['50_day'].iloc[-1]:
                recommendations.append('20-day MA is above 50-day MA (uptrend)')
                buy_signals += 1
            if moving_averages['50_day'].iloc[-1] > moving_averages['200_day'].iloc[-1]:
                recommendations.append('50-day MA is above 200-day MA (uptrend)')
                buy_signals += 1
            if moving_averages['20_day'].iloc[-1] > moving_averages['200_day'].iloc[-1]:
                recommendations.append('20-day MA is above 200-day MA (uptrend)')
                buy_signals += 1
            if moving_averages['20_day'].iloc[-1] < moving_averages['50_day'].iloc[-1]:
                recommendations.append('20-day MA is below 50-day MA (downtrend)')
                sell_signals += 1
            if moving_averages['50_day'].iloc[-1] < moving_averages['200_day'].iloc[-1]:
                recommendations.append('50-day MA is below 200-day MA (downtrend)')
                sell_signals += 1
            if moving_averages['20_day'].iloc[-1] < moving_averages['200_day'].iloc[-1]:
                recommendations.append('20-day MA is below 200-day MA (downtrend)')
                sell_signals += 1

            # RSI
            if rsi.iloc[-1] < 30:
                recommendations.append('RSI is below 30 (oversold)')
                buy_signals += 1
            elif rsi.iloc[-1] > 70:
                recommendations.append('RSI is above 70 (overbought)')
                sell_signals += 1

            # MACD
            if macd.iloc[-1] > signal.iloc[-1]:
                recommendations.append('MACD is above the Signal Line (bullish)')
                buy_signals += 1
            elif macd.iloc[-1] < signal.iloc[-1]:
                recommendations.append('MACD is below the Signal Line (bearish)')
                sell_signals += 1

            # News Sentiment Analysis
            news_sentiment = get_news_sentiment(ticker)
            if news_sentiment:
                recommendations.append(f"News Sentiment: {news_sentiment}")
                if news_sentiment == 'Positive':
                    buy_signals += 1
                elif news_sentiment == 'Negative':
                    sell_signals += 1

            # Final recommendation based on stricter criteria
            if buy_signals >= 4 and sell_signals == 0:
                recommendations.append('Strong Buy')
            elif buy_signals > sell_signals:
                recommendations.append('Buy')
            else:
                recommendations.append('No Buy')

        except Exception as e:
            print(f"Error identifying trends: {e}")
    return recommendations
    return recommendations
def get_strong_buy_recommendations():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT ticker, price FROM recommendations WHERE recommendation = "Strong Buy"')
    recommendations = cursor.fetchall()
    conn.close()
    return recommendations

def create_best_limit_orders():
    balance = get_account_balance()
    if balance is None:
        print("Could not fetch account balance.")
        return

    recommendations = get_strong_buy_recommendations()
    if not recommendations:
        print("No Strong Buy recommendations found.")
        return

    open_tickers = get_open_orders()
    if open_tickers:
        print(f"Skipping tickers with open orders: {', '.join(open_tickers)}")

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    analyzed_recommendations = []
    for ticker, _ in recommendations:
        if ticker in open_tickers:
            print(f"Skipping ticker with open order: {ticker}")  # Debug print
            continue  # Skip tickers with open orders
        data = download_data(ticker, start_date, end_date)

        if data is not None and not data.empty:
            moving_averages = calculate_moving_averages(data)
            rsi = calculate_rsi(data)
            macd, signal = calculate_macd(data)
            recs = identify_trends(ticker, data, moving_averages, rsi, macd, signal)
            if recs and recs[-1] == 'Strong Buy':
                analyzed_recommendations.append((ticker, data['Close'].iloc[-1], recs))

    analyzed_recommendations.sort(key=lambda x: x[1])  # Sort by price ascending

    selected_recommendations = []
    total_spent = 0
    max_stocks_to_buy = len(analyzed_recommendations)
    if max_stocks_to_buy > 0:
        amount_per_stock = balance / max_stocks_to_buy

    for ticker, price, recs in analyzed_recommendations:
        quantity = max(1, int(amount_per_stock / price))  # Ensure at least 1 share is bought
        if quantity > 0 and (total_spent + (quantity * price)) <= balance:
            selected_recommendations.append((ticker, round(price, 2), quantity))  # Round the price to 2 decimal places
            total_spent += quantity * price
    for ticker, price, quantity in selected_recommendations:
        if not is_asset_active(ticker):
            print(f"Skipping inactive asset: {ticker}")
            continue
        try:
            api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='buy',
                type='limit',
                time_in_force='gtc',
                limit_price=price
            )
            print(f"Submitted limit order for {quantity} shares of {ticker} at ${price:.2f}")
        except Exception as e:
            print(f"Error creating order for {ticker}: {e}")


def get_news_sentiment(ticker):
    try:
        news = api.get_news(ticker)
        #print(news)
        if news:
            sentiments = [analyzer.polarity_scores(article.headline)['compound'] for article in news]
            avg_sentiment = sum(sentiments) / len(sentiments)
            if avg_sentiment > 0.05:
                return 'Positive'
            elif avg_sentiment < -0.05:
                return 'Negative'
            else:
                return 'Neutral'
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
    return None


# Function to get the account balance
def get_account_balance():
    try:
        account = api.get_account()
        return float(account.cash)
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")
        return None

# Function to get current positions
def get_positions():
    try:
        positions = api.list_positions()
        return positions
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")
        return []

# Function to check if the ticker exists
def ticker_exists(ticker):
    try:
        asset = api.get_asset(ticker)
        return asset is not None and asset.tradable
    except tradeapi.rest.APIError:
        return False

# Function to execute buy orders and store the transaction details
def execute_buy_order(ticker, quantity, order_type='limit'):
    if not ticker_exists(ticker):
        print(f"Error: Ticker '{ticker}' not found or not tradable")
        return
    
    try:
        if order_type == 'market':
            order = api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            price = api.get_latest_trade(ticker).price
        else:
            price = api.get_latest_trade(ticker).price
            order = api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='buy',
                type='limit',
                time_in_force='gtc',
                limit_price=price
            )
        
        # Store the transaction details in the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (id, ticker, basis, quantity, purchase_date, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (order.id, ticker, price, quantity, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'submitted'))
        conn.commit()
        conn.close()
        
        print(f"Bought {quantity} shares of {ticker} at ${price:.2f} ({order_type} order)")
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")

# Function to execute sell orders
def execute_sell_order(ticker, quantity, order_type='limit'):
    if not ticker_exists(ticker):
        print(f"Error: Ticker '{ticker}' not found or not tradable")
        return

    try:
        if order_type == 'market':
            order = api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            price = api.get_latest_trade(ticker).price
        else:
            price = api.get_latest_trade(ticker).price
            order = api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='sell',
                type='limit',
                time_in_force='gtc',
                limit_price=price
            )

        print(f"Sold {quantity} shares of {ticker} at ${price:.2f} ({order_type} order)")
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")

# Function to update orders and positions at startup
def update_orders_and_positions():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Update orders
    try:
        orders = api.list_orders(status='open')
        for order in orders:
            cursor.execute('SELECT * FROM transactions WHERE id = ?', (order.id,))
            if cursor.fetchone() is None:
                cursor.execute('''
                    INSERT INTO transactions (id, ticker, basis, quantity, purchase_date, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (order.id, order.symbol, None, order.qty, order.submitted_at.isoformat(), order.status))
        conn.commit()
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")
    
    # Update positions
    try:
        positions = api.list_positions()
        for position in positions:
            cursor.execute('SELECT * FROM positions WHERE ticker = ?', (position.symbol,))
            if cursor.fetchone() is None:
                cursor.execute('''
                    INSERT INTO positions (ticker, basis, quantity, last_checked_price, last_checked_date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (position.symbol, position.avg_entry_price, position.qty, position.current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")
    
    conn.close()

# Function to monitor positions and update the database
def monitor_positions():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all positions from the database
    cursor.execute('SELECT * FROM positions')
    positions = cursor.fetchall()
    
    for position in positions:
        ticker = position[1]
        basis = position[2]
        quantity = position[3]
        
        # Get the current price
        current_price = api.get_latest_trade(ticker).price
        
        # Update the last checked price and date
        cursor.execute('''
            UPDATE positions
            SET last_checked_price = ?, last_checked_date = ?
            WHERE ticker = ?
        ''', (current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker))
        
        # Check if there is a significant change in price (e.g., 5% increase or decrease)
        change = (current_price - basis) / basis * 100
        if abs(change) >= 5:
            print(f"{ticker} has changed by {change:.2f}% since purchase")
    
    conn.commit()
    conn.close()

# Function to show orders
def show_orders():
    try:
        orders = api.list_orders(status='all')
        for order in orders:
            print(f"ID: {order.id}, Symbol: {order.symbol}, Qty: {order.qty}, Type: {order.order_type}, Status: {order.status}, Submitted At: {order.submitted_at}")
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")

# Function to show positions
def show_positions():
    try:
        positions = api.list_positions()
        for position in positions:
            print(f"Symbol: {position.symbol}, Qty: {position.qty}, Avg Entry Price: {position.avg_entry_price}, Current Price: {position.current_price}, Market Value: {position.market_value}, Unrealized PL: {position.unrealized_pl}")
    except tradeapi.rest.APIError as e:
        print(f"API Error: {e}")

# Function to monitor orders and update their status
def monitor_orders():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    while not stop_event.is_set():
        cursor.execute('SELECT * FROM transactions WHERE status != "filled"')
        orders = cursor.fetchall()
        
        for order in orders:
            order_id = order[0]
            ticker = order[1]
            
            try:
                alpaca_order = api.get_order(order_id)
                if alpaca_order.status == 'filled':
                    cursor.execute('UPDATE transactions SET status = ? WHERE id = ?', ('filled', order_id))
                    cursor.execute('''
                        INSERT INTO positions (ticker, basis, quantity, last_checked_price, last_checked_date)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (ticker, alpaca_order.filled_avg_price, alpaca_order.qty, alpaca_order.filled_avg_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    conn.commit()
                    print(f"Order {order_id} for {ticker} filled at ${alpaca_order.filled_avg_price}")
            except tradeapi.rest.APIError as e:
                print(f"API Error: {e}")
        
        time.sleep(1)  # Check more frequently to allow quick shutdown

    conn.close()

# Function to handle news updates
async def handle_news(conn):
    @conn.on(r'news')
    async def on_news(conn, channel, data):
        print(f"News received: {data}")

    while not stop_event.is_set():
        await asyncio.sleep(1)

# Function to run the news thread
def run_news_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    conn = tradeapi.Stream(API_KEY, API_SECRET, BASE_URL)
    conn.subscribe_news(handle_news)
    
    async def run_forever():
        while not stop_event.is_set():
            await asyncio.sleep(1)

    try:
        loop.run_until_complete(run_forever())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.stop()
        loop.close()

class TradingBotCLI(cmd.Cmd):
    intro = 'Welcome to the trading bot CLI. Type help or ? to list commands.\n'
    prompt = '(trading_bot) '

    def do_analyze(self, arg):
        'Analyze the given stock ticker: analyze [ticker]'
        ticker = arg.strip().upper()
        if ticker:
            print(f"Analyzing {ticker}...")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            data = download_data(ticker, start_date, end_date)
            if data is not None:
                moving_averages = calculate_moving_averages(data)
                rsi = calculate_rsi(data)
                macd, signal = calculate_macd(data)
                recommendations = identify_trends(ticker, data, moving_averages, rsi, macd, signal)
                if recommendations:
                    for rec in recommendations:
                        print(rec)
                else:
                    print(f"{ticker} does not show a clear trend.")
            else:
                print(f"Failed to analyze {ticker}")
        else:
            print("Please provide a ticker symbol")

    def do_readtickers(self, arg):
        'Read tickers from a CSV file or database and output recommendations to another CSV file or database: readtickers <input_file> <output_file>'
        args = arg.split()
        if len(args) != 2:
            print("Usage: readtickers <input_file> <output_file>")
            return

        input_file, output_file = args
        try:
            if input_file.lower() == 'db':
                # Read tickers from the database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT ticker FROM symbols')
                tickers = cursor.fetchall()
                conn.close()
                tickers_df = pd.DataFrame(tickers, columns=['Symbol'])
            else:
                # Read tickers from the CSV file
                tickers_df = pd.read_csv(input_file)
            
            recommendations = []

            for _, row in tickers_df.iterrows():
                ticker = row['Symbol'].upper()
                print(f"Analyzing {ticker}...")
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                try:
                    data = download_data(ticker, start_date, end_date)
                    if data is not None and not data.empty:
                        moving_averages = calculate_moving_averages(data)
                        rsi = calculate_rsi(data)
                        macd, signal = calculate_macd(data)
                        recs = identify_trends(ticker, data, moving_averages, rsi, macd, signal)
                        recommendation = recs[-1] if recs else 'No clear trend'
                        current_price = data['Close'].iloc[-1]
                        change_percent = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                        volume = data['Volume'].iloc[-1]
                    else:
                        recommendation = 'Failed to analyze'
                        current_price = None
                        change_percent = None
                        volume = None
                except Exception as e:
                    recommendation = f"Error: {e}"
                    current_price = None
                    change_percent = None
                    volume = None
                recommendations.append([ticker, recommendation, current_price, change_percent, volume])

            if output_file.lower() == 'db':
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                for rec in recommendations:
                    ticker, recommendation, current_price, change_percent, volume = rec
                    cursor.execute('''
                        INSERT INTO recommendations (ticker, recommendation, price, change_percent, volume) 
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(ticker) DO UPDATE SET 
                        recommendation=excluded.recommendation,
                        price=excluded.price,
                        change_percent=excluded.change_percent,
                        volume=excluded.volume
                    ''', (ticker, recommendation, current_price, change_percent, volume))
                conn.commit()
                conn.close()
                print("Recommendations saved to database")
            else:
                recommendations_df = pd.DataFrame(recommendations, columns=['Ticker', 'Recommendation', 'Price', 'Change Percent', 'Volume'])
                recommendations_df.to_csv(output_file, index=False)
                print(f"Recommendations saved to {output_file}")

        except Exception as e:
            print(f"Error reading or writing files: {e}")

    def do_buy(self, arg):
        'Execute a buy order: buy [ticker] [quantity] [market (optional)]'
        args = arg.split()
        if len(args) < 2 or len(args) > 3:
            print("Usage: buy [ticker] [quantity] [market (optional)]")
            return

        ticker = args[0].strip().upper()
        try:
            quantity = int(args[1])
            order_type = 'market' if len(args) == 3 and args[2].lower() == 'market' else 'limit'
            execute_buy_order(ticker, quantity, order_type)
        except ValueError:
            print("Quantity must be an integer")

    def do_sell(self, arg):
        'Execute a sell order: sell [ticker] [quantity] [market (optional)]'
        args = arg.split()
        if len(args) < 2 or len(args) > 3:
            print("Usage: sell [ticker] [quantity] [market (optional)]")
            return

        ticker = args[0].strip().upper()
        try:
            quantity = int(args[1])
            order_type = 'market' if len(args) == 3 and args[2].lower() == 'market' else 'limit'
            execute_sell_order(ticker, quantity, order_type)
        except ValueError:
            print("Quantity must be an integer")

    def do_monitor(self, arg):
        'Monitor positions for significant changes'
        monitor_positions()

    def do_getbalance(self, arg):
        'Get the current account balance: getbalance'
        balance = get_account_balance()
        if balance is not None:
            print(f"Current account balance: ${balance:.2f}")

    def do_showorders(self, arg):
        'Show all orders: showorders'
        show_orders()

    def do_showpositions(self, arg):
        'Show all positions: showpositions'
        show_positions()

    def do_downloadsymbols(self, arg):
        'Download and store symbols from SEC: downloadsymbols'
        download_symbols()

    def do_showrecommendations(self, arg):
        'Show recommendations (Strong Buy, Buy, No Buy) from the database: showrecommendations [type]'
        recommendation_type = arg.strip().title()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if recommendation_type:
            cursor.execute('SELECT ticker, recommendation FROM recommendations WHERE recommendation = ?', (recommendation_type,))
        else:
            cursor.execute('SELECT ticker, recommendation FROM recommendations')

        recommendations = cursor.fetchall()
        conn.close()

        if recommendation_type:
            print(f"{recommendation_type}:")
            for rec in recommendations:
                print(f"  {rec[0]}")
        else:
            strong_buy = [rec for rec in recommendations if rec[1] == 'Strong Buy']
            buy = [rec for rec in recommendations if rec[1] == 'Buy']
            no_buy = [rec for rec in recommendations if rec[1] == 'No Buy']

            print("Strong Buy:")
            for rec in strong_buy:
                print(f"  {rec[0]}")

            print("\nBuy:")
            for rec in buy:
                print(f"  {rec[0]}")

            print("\nNo Buy:")
            for rec in no_buy:
                print(f"  {rec[0]}")

    def do_gettime(self, arg):
        'Get the current time from Alpaca: gettime'
        current_time, _ = get_market_status()
        if current_time:
            print(f"Current Alpaca time: {current_time}")
        else:
            print("Failed to fetch the current time from Alpaca")

    def do_marketstatus(self, arg):
        'Get the current market status from Alpaca: marketstatus'
        current_time, is_open = get_market_status()
        if current_time:
            status = "open" if is_open else "closed"
            print(f"Current Alpaca time: {current_time}, Market is {status}")
        else:
            print("Failed to fetch the market status from Alpaca")

    def do_createlimitorders(self, arg):
        'Create limit orders for Strong Buy recommendations: createlimitorders'
        create_best_limit_orders()

    def do_exit(self, arg):
        'Exit the CLI'
        print("Exiting...")
        stop_event.set()  # Signal the news thread to stop
        return True

    def emptyline(self):
        pass

    def postcmd(self, stop, line):
        global news_thread
        global order_monitor_thread
        if stop:
            print("Shutting down threads...")
            if news_thread and news_thread.is_alive():
                news_thread.join()
            if order_monitor_thread and order_monitor_thread.is_alive():
                order_monitor_thread.join()
            print("Threads terminated. Exiting CLI.")
        return stop

if __name__ == '__main__':
    # Global variable to hold the thread references
    global news_thread
    global order_monitor_thread

    # Initialize the SQLite database
    init_db()

    # Update orders and positions at startup
    update_orders_and_positions()

    # Start monitoring orders in a separate thread
    order_monitor_thread = threading.Thread(target=monitor_orders)
    order_monitor_thread.start()

    # Start monitoring news in a separate thread
    news_thread = threading.Thread(target=run_news_thread)
    news_thread.start()

    # Create an instance of the CLI and run it
    TradingBotCLI().cmdloop()

    # Wait for the threads to finish
    order_monitor_thread.join()
    news_thread.join()
