import os
import json
import requests 
import pandas as pd
from datetime import datetime, timezone
from core.config import settings


def fetch_coin_dataset():
    
    tenx_df = pd.read_csv(settings.TENX_DATA_FOLDER)
    not_tenx_df = pd.read_csv(settings.NOT_TENX_DATA_FOLDER)

    combined_df = pd.concat([tenx_df, not_tenx_df], ignore_index=True)

    return combined_df


def fetching_market_data():
    # response = requests.get(
    #     'https://data-api.coindesk.com/spot/v1/markets/instruments',
    #     params={"instrument_status": "ACTIVE", "api_key": settings.COIN_DESK_API_KEY},
    #     headers={"Content-type": "application/json; charset=UTF-8"},
    # )

    # json_response = response.json()
    
    # folder_path = os.path.join(settings.DATA_FOLDER, "coin_price", settings.COIN_NAME)
    # os.makedirs(folder_path, exist_ok=True)

    # with open(os.path.join(settings.DATA_FOLDER, f"coin_price/{settings.COIN_NAME}/{settings.COIN_NAME}_market_data.json"), "w") as f:
    #     f.write(json.dumps(json_response))

    file_path = os.path.join(
        settings.DATA_FOLDER,
        f"coin_price/market_data.json"
    )

    with open(file_path, "r") as f:
        json_data = json.load(f)

    return json_data


def get_symbol_by_name(coin_dataset, coin_name):
    coin_name = coin_name.lower()

    # Lower both sides for comparison
    result = coin_dataset[coin_dataset['name'].str.lower().str.contains(coin_name, na=False)]

    if not result.empty:
        return result.iloc[0]['symbol']
    else:
        return "Coin name not found"
    

# Helper Function
def convert_timestamp_to_unix(time_string):
    dt = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    
    # Set the timezone to UTC if the input is in UTC
    dt = dt.replace(tzinfo=timezone.utc)

    # Convert to Unix timestamp
    unix_timestamp = int(dt.timestamp())
    return unix_timestamp


def convert_timestamp_date(timestamp):
    utc_time = datetime.fromtimestamp(timestamp)
    return utc_time.strftime('%Y-%m-%d %H:%M:%S')


def get_earliest_trade_for_coin_with_time(coin_symbol, start_timestamp, end_timestamp, market_data):
    coin_symbol = coin_symbol.lower()

    earliest_market = None  
    earliest_instrument = None  
    earliest_timestamp = float("inf")

    for market, exchange_data in market_data["Data"].items(): 
        instruments = exchange_data.get("instruments", {})

        for instrument, info in instruments.items():  
            instrument_lower = instrument.lower()

            # Check if the instrument matches the coin and has a valid suffix (USDT, USD, or USDC)
            if instrument_lower == coin_symbol + "-usdt" or instrument_lower == coin_symbol + "-usd" or instrument_lower == coin_symbol + "-usdc":
                timestamp = info.get("FIRST_TRADE_SPOT_TIMESTAMP")
                
                # Check if the timestamp is within the specified range
                if timestamp is not None:
                    # Update earliest market if this timestamp is earlier than the current one
                    if timestamp < earliest_timestamp:
                        earliest_timestamp = timestamp
                        earliest_market = market
                        earliest_instrument = instrument

    if earliest_timestamp > start_timestamp:
        raise Exception(f"Start timestamp {datetime.fromtimestamp(start_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} is before earliest recorded timestamp which is {datetime.fromtimestamp(earliest_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    elif earliest_instrument:
        return {
            "market": earliest_market,
            "instrument": earliest_instrument,
            "timestamp": earliest_timestamp
        }
    else:
        raise Exception(f"No matching market found for coin '{coin_symbol.upper()}' with USD, USDT, or USDC pairs within the specified time range.")
    

def fetch_price_data(earliest_trade, start_unix, end_unix):
    price_data = []
    
    response = requests.get(
        'https://data-api.coindesk.com/spot/v1/historical/hours',
        params={
            "market": earliest_trade["market"],
            "instrument": earliest_trade["instrument"],
            "limit": 2000,
            "aggregate": 1,
            "fill": "true",
            "apply_mapping": "true",
            "response_format": "JSON",
            "to_ts": end_unix,
            "api_key": settings.COIN_DESK_API_KEY
        },
        headers={"Content-type": "application/json; charset=UTF-8"}
    )
    price_response = response.json()

    price_data.append(price_response)

    earliest_timestamp = price_response["Data"][0]["TIMESTAMP"]

    if price_response["Err"]:
        print(price_response)

    while True:

        if earliest_timestamp < start_unix:
            break

        response = requests.get(
            'https://data-api.coindesk.com/spot/v1/historical/hours',
            params={
                "market": earliest_trade["market"],
                "instrument": earliest_trade["instrument"],
                "limit": 2000,
                "aggregate": 1,
                "fill": "true",
                "apply_mapping": "true",
                "response_format": "JSON",
                "to_ts": earliest_timestamp,
                "api_key": settings.COIN_DESK_API_KEY
            },
            headers={"Content-type": "application/json; charset=UTF-8"}
        )
        price_response = response.json()

        if price_response["Err"]:
            break

        price_data.append(price_response)
        
        # No later timestamp is available for that day
        if earliest_timestamp == price_response["Data"][0]["TIMESTAMP"]:
            earliest_timestamp = price_response["Data"][0]["TIMESTAMP"] - 86400
        else:
            earliest_timestamp = price_response["Data"][0]["TIMESTAMP"]
    
    return price_data


def export_to_csv(price_data):
    desired_columns = [
        "TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE",
        "TOTAL_TRADES", "TOTAL_TRADES_BUY", "TOTAL_TRADES_SELL", "TOTAL_TRADES_UNKNOWN",
        "VOLUME", "QUOTE_VOLUME", "VOLUME_BUY", "QUOTE_VOLUME_BUY",
        "VOLUME_SELL", "QUOTE_VOLUME_SELL", "FIRST_TRADE_PRICE", "HIGH_TRADE_PRICE",
        "LOW_TRADE_PRICE", "LAST_TRADE_PRICE"
    ]

    # Flatten and filter
    flattened_filtered_data = []
    for item in price_data:
        for row in item.get("Data", []):
            filtered_row = {key: row.get(key, None) for key in desired_columns}
            flattened_filtered_data.append(filtered_row)

    # Convert to DataFrame
    df = pd.DataFrame(flattened_filtered_data)
    df['TIMESTAMP'] = df['TIMESTAMP'].apply(convert_timestamp_date)

    df.reset_index(drop=True, inplace=True)
    # Build the full directory path
    dir_path = os.path.join(settings.DATA_FOLDER, f"coin_price/{settings.COIN_NAME}")

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Save the file
    df.to_csv(os.path.join(dir_path, f"{settings.COIN_NAME}_price_dataset.csv"), index=False)
