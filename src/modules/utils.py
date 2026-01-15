import ahocorasick
import pandas as pd
import re
import jieba  # For Chinese text segmentation
import unicodedata
import json


# Get the list of unique authors
def get_unique_author_names(df):
    usernames = df['author'].apply(lambda x: x.get('userName') if isinstance(x, dict) else None)
    unique_usernames = usernames.dropna().unique()
    
    return set(unique_usernames)


def unique_author_dataset(df):
    # Safely convert stringified dicts into real dicts
    # df['author_dict'] = df['author'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Now normalize properly
    author_data = pd.json_normalize(df['author'])

    # Drop duplicates based on screen_name
    unique_users_df = author_data.drop_duplicates(subset='userName', keep='first')

    return unique_users_df


def grouping_tweets(df):
    df['userName'] = df['author'].apply(lambda x: x.get('userName') if isinstance(x, dict) else None)
    grouped_tweets = df.groupby('userName').apply(lambda x: x.to_dict(orient='records')).to_dict()

    return df, grouped_tweets


def get_cutoff_date(df):

    max_date = df['createdAt'].max().date()  # Get max date only
    cutoff_date = df[df['createdAt'].dt.date == max_date]['createdAt'].max()
    cutoff_date = cutoff_date.tz_localize(None)  # Remove timezone if needed

    return cutoff_date


def extract_username(author_obj):
    try:
        if isinstance(author_obj, str):
            author_obj = json.loads(author_obj)
        return author_obj.get("userName", None)
    except (AttributeError, ValueError, SyntaxError, json.JSONDecodeError):
        return None
    
def extract_coin_mentions(df, tenx_df, non_tenx_df):
    """
    Extracts coin mentions from a DataFrame containing tweet text and associates each mention with its corresponding coin ID.

    The function processes the tweet text, identifies mentions of coins (via hashtags, mentions, dollar signs, or direct text matches), 
    and maps them to corresponding coin IDs from two datasets: `tenx_df` and `non_tenx_df`. Coin mentions are matched in priority order:
    1. 10x coins are searched first (screen_name > symbol > name)
    2. non-10x coins are searched next (screen_name > symbol > name)
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing tweet data with a `fullText` column where coin mentions will be extracted.
    tenx_df (pd.DataFrame): The DataFrame containing information about tenx coins (with columns `name`, `id`, `symbol`, and `screen_name`).
    non_tenx_df (pd.DataFrame): The DataFrame containing information about non-tenx coins (with columns `name`, `id`, `symbol`, and `screen_name`).

    Returns:
    pd.DataFrame: A DataFrame with an additional column `coin_mentions`, which lists the coin IDs mentioned in each tweet.
    """

    def normalize(text):
        return str(text).lower().strip()

    # Preprocess and normalize input data
    tenx_df = tenx_df.dropna(subset=["name", "id", "symbol", "screen_name"]).copy()
    non_tenx_df = non_tenx_df.dropna(subset=["name", "id", "symbol", "screen_name"]).copy()

    tenx_df["name_norm"] = tenx_df["name"].apply(normalize)
    tenx_df["symbol_norm"] = tenx_df["symbol"].apply(normalize)
    tenx_df["screen_name_norm"] = "@" + tenx_df["screen_name"].apply(normalize)

    non_tenx_df["name_norm"] = non_tenx_df["name"].apply(normalize)
    non_tenx_df["symbol_norm"] = non_tenx_df["symbol"].apply(normalize)
    non_tenx_df["screen_name_norm"] = "@" + non_tenx_df["screen_name"].apply(normalize)

    # Priority ordered mapping: 10x first
    ordered_mappings = []
    for df_source in [tenx_df, non_tenx_df]:
        for col in ["screen_name_norm", "symbol_norm", "name_norm"]:
            ordered_mappings.extend([(k, v) for k, v in zip(df_source[col], df_source["id"])])

    # Filter out common/short tokens
    common_words = {
        "about", "again", "all", "an", "and", "any", "are", "as", "at", "bad", "be", "big", "but", "by", "can", 
        "different", "do", "early", "every", "for", "from", "good", "has", "high", "how", "if", "in", "is", "it", 
        "just", "late", "like", "long", "low", "me", "more", "most", "much", "my", "new", "not", "now", "old", "on", 
        "one", "only", "or", "other", "out", "over", "own", "short", "so", "that", "the", "this", "to", "under", "up", 
        "way", "we", "well", "what", "when", "where", "why", "will", "with", "would", "you", "young", "your"
    }

    filtered_mapping = {}
    for k, v in ordered_mappings:
        if k not in common_words and len(k) > 2 and k not in filtered_mapping:
            filtered_mapping[k] = v

    # Define regex patterns
    hashtag_pattern = re.compile(r'#([A-Za-z0-9]+)')
    mention_pattern = re.compile(r'@([A-Za-z0-9_]+)')
    dollar_pattern = re.compile(r'\$([A-Za-z0-9]+)')

    def contains_non_english(text):
        """
        Detects if the text contains non-English characters.
        """
        return any(unicodedata.category(char)[0] not in ('L', 'N') for char in text)

    def tokenize_text(text):
        """
        Tokenizes text using jieba for non-English or regex for English.
        """
        if contains_non_english(text):
            return jieba.lcut(text)
        return re.findall(r'\b[a-zA-Z0-9-]+\b', text)

    def extract_coin_ids(text):
        """
        Extracts matching coin IDs from a given tweet text.
        """
        coin_ids = set()
        text_lower = text.lower()

        # Extract hashtags
        for match in hashtag_pattern.findall(text_lower):
            tag = normalize(match)
            if tag in filtered_mapping:
                coin_ids.add(filtered_mapping[tag])

        # Extract mentions
        for match in mention_pattern.findall(text_lower):
            tag = "@" + normalize(match)
            if tag in filtered_mapping:
                coin_ids.add(filtered_mapping[tag])

        # Extract tickers
        for match in dollar_pattern.findall(text_lower):
            tag = normalize(match)
            if tag in filtered_mapping:
                coin_ids.add(filtered_mapping[tag])

        # Tokenized word matches
        for word in tokenize_text(text_lower):
            norm_word = normalize(word)
            if norm_word in filtered_mapping:
                coin_ids.add(filtered_mapping[norm_word])

        return list(coin_ids)

    df["coin_mentions"] = df["fullText"].astype(str).apply(extract_coin_ids)
    return df