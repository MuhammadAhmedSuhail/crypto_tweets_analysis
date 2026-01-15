import os
import pandas as pd
from ..utils import extract_coin_mentions, extract_username
from core.config import settings


def preprocess_tweets():
    tweets = pd.read_json(os.path.join(settings.DATA_FOLDER, f"coin_tweets/{settings.COIN_NAME}/{settings.COIN_NAME}_tweets.json"))

    tweets["createdAt"] = pd.to_datetime(
        tweets['createdAt'],
        format="%a %b %d %H:%M:%S +0000 %Y",
        errors='coerce',
        utc=True
    )

    # tweets['author'] = tweets['author'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return tweets

    # pendle_tweets.to_csv(output_path, index=False)


def merge_tweet_analysis():
    tweets = preprocess_tweets()
    
    tweets_llm = pd.read_csv(os.path.join(settings.DATA_FOLDER, f"coin_tweets/{settings.COIN_NAME}/{settings.COIN_NAME}_llm_tweets.csv"))
    
    tweet_analysis_merged = pd.merge(tweets, tweets_llm, on="id")

    tweet_analysis_merged["userName"] = tweet_analysis_merged["author"].apply(extract_username)

    return tweet_analysis_merged


def fetch_coin_dataset():
    tenx_df = pd.read_csv(os.path.join(settings.DATA_FOLDER, "assets/10x_coins.csv"))
    non_tenx_df = pd.read_csv(os.path.join(settings.DATA_FOLDER, "assets/not_10x_coins.csv"))

    tenx_df = tenx_df.dropna(subset=["name", "id", "symbol", "screen_name"])
    non_tenx_df = non_tenx_df.dropna(subset=["name", "id", "symbol", "screen_name"])

    return tenx_df, non_tenx_df


def coin_only_tweets():

    df = merge_tweet_analysis()
    tenx_df, non_tenx_df = fetch_coin_dataset()
    df = extract_coin_mentions(df, tenx_df, non_tenx_df)
    

    df = df[df["coin_mentions"].apply(lambda x: settings.COIN_NAME in x)]

    return df


def pendle_coin_dataset():
    df = pd.read_csv(os.path.join(settings.DATA_FOLDER, f"coin_price/{settings.COIN_NAME}/{settings.COIN_NAME}_price_dataset.csv"))

    return df


def pendle_mentioned_df():
    df = coin_only_tweets()

    return df