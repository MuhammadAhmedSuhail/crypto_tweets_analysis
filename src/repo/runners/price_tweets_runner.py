from datetime import datetime
import json
import os
import pandas as pd
from modules.coin_analysis.coin_price import convert_timestamp_to_unix, export_to_csv, fetch_coin_dataset
from modules.coin_analysis.coin_price import fetch_price_data, fetching_market_data, get_earliest_trade_for_coin_with_time, get_symbol_by_name
from modules.coin_analysis.coin_tweets import extract_tweets, fetch_coin_details, parse_tweet_data
from core.config import settings


def get_coin_dataset(coin_name, start_time_string, end_time_string):
    # Preprocessing
    all_coins = fetch_coin_dataset()
    market_data = fetching_market_data()

    symbol = get_symbol_by_name(all_coins, coin_name)
    start_unix = convert_timestamp_to_unix(start_time_string)
    end_unix = convert_timestamp_to_unix(end_time_string)

    # Fetching price and trade
    earliest_trade = get_earliest_trade_for_coin_with_time(symbol, start_unix, end_unix, market_data)
    price_data = fetch_price_data(earliest_trade, start_unix, end_unix)

    export_to_csv(price_data)


def get_tweets(coin_name, start_time_string, end_time_string):
    start_date = datetime.strptime(start_time_string, "%Y-%m-%d %H:%M:%S").date()
    end_date = datetime.strptime(end_time_string, "%Y-%m-%d %H:%M:%S").date()

    coin_dataset = fetch_coin_dataset()
    symbol, handle = fetch_coin_details(coin_name, coin_dataset)
    extracted_tweets = extract_tweets(coin_name, symbol, handle, start_date, end_date)
    
    filtered_tweets = []
    for i in range(len(extracted_tweets)):
        filtered_tweets.extend(parse_tweet_data(extracted_tweets[i]))

    df = pd.DataFrame(filtered_tweets)

    # Convert to list of dicts
    json_data = df.to_dict(orient="records")

    folder_path = os.path.join(settings.DATA_FOLDER, "coin_tweets", settings.COIN_NAME)
    os.makedirs(folder_path, exist_ok=True)
    # Write as pretty JSON
    with open(os.path.join(settings.DATA_FOLDER, f"coin_tweets/{coin_name}/{coin_name}_tweets.json"), "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)