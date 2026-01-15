import numpy as np
import pandas as pd

from modules.utils import extract_coin_mentions
from ..preprocessing.tweet_preprocessing import fetch_coin_dataset, pendle_coin_dataset, pendle_mentioned_df, coin_only_tweets
from collections import defaultdict


def account_engagement_ratio(df):
    """
    Calculates the engagement ratio for Twitter accounts based on total interactions and view counts.

    Parameters:
    df (pandas.DataFrame): DataFrame containing Twitter data with columns like 'retweetCount', 'likeCount', etc.

    Returns:
    dict: A dictionary with the average engagement ratio for each user.
    """
    df['view_count'] = df['view_count'].fillna(1)

    interaction_cols = ['retweet_count', 'reply_count', 'like_count', 'quote_count', 'bookmark_count']
    df[interaction_cols] = df[interaction_cols].fillna(0)

    # Calculate total interactions
    df['totalInteractions'] = df[interaction_cols].sum(axis=1)

    # Extract follower count
    df['followers'] = df['author'].apply(lambda x: x.get('followers', 1) if isinstance(x, dict) else 1)

    # Avoid division by zero and calculate engagement ratio
    df['engagementRatio'] = df['totalInteractions'] / df['view_count'].replace(0, 1)

    # Group by user and calculate average engagement ratio
    engagement_ratios = df.groupby('userName')['engagementRatio'].mean().to_dict()

    return engagement_ratios


def account_age(df, cutoff_date):
    """
    Calculates the account age score for Twitter users based on their account creation date.

    Parameters:
    df (pandas.DataFrame): DataFrame containing Twitter account data with 'createdAt' column.
    cutoff_date (datetime): The date used as the reference point for calculating account age.

    Returns:
    dict: A dictionary with the account age score for each user.
    """
    df['accountCreatedAt'] = pd.to_datetime(df['createdAt'], errors='coerce').dt.tz_localize(None)

    # Ensure cutoff_date is a full timestamp
    cutoff_date = cutoff_date.tz_localize(None) if cutoff_date.tz is not None else cutoff_date

    # Calculate account age in days (use total_seconds for accuracy)
    df['accountAgeDays'] = (cutoff_date - df['accountCreatedAt']).dt.total_seconds() / (24 * 3600)

    # Ensure non-negative values (in case of future dates)
    df['accountAgeDays'] = df['accountAgeDays'].clip(lower=0)

    # Calculate total days (avoid division by zero)
    total_days = df['accountAgeDays'].max() if pd.notnull(df['accountAgeDays'].max()) else 1

    # Calculate the age score
    df['ageScore'] = df['accountAgeDays'].div(total_days).clip(upper=1).fillna(0)

    # Return the scores as a dictionary
    account_ages = df.set_index('userName')['ageScore'].to_dict()

    return account_ages


def calculate_profile_completeness(df):
    """
    Calculates a profile completeness score based on profile attributes such as pictures and description.

    Parameters:
    df (pandas.DataFrame): DataFrame containing user profile information with columns like 'profilePicture', 'coverPicture', etc.

    Returns:
    dict: A dictionary with the completeness score for each user.
    """

    # Calculate completeness score with adjusted weightages
    df['completeness_score'] = (
        df['profile_picture'].notnull().astype(int) * 0.1 +
        df['cover_picture'].notnull().astype(int) * 0.1 +
        df['description'].notnull().astype(int) * 0.1 +
        df['can_dm'].astype(int) * 0.2 +
        df['is_verified'].astype(int) * 0.5
    )
    
    # Return the scores as a dictionary
    profile_scores = df.set_index('userName')['completeness_score'].to_dict()
    
    return profile_scores


def calculate_media_status_ratio(df):
    """
    Calculates the media status ratio based on media count and tweet status count.

    Parameters:
    df (pandas.DataFrame): DataFrame containing user tweet data with columns like 'mediaCount' and 'statusesCount'.

    Returns:
    dict: A dictionary with the media status ratio for each user.
    """

    df['mediaStatusRatio'] = df['media_count'].fillna(0) / (df['statuses_count'].fillna(0) + 1)
    media_status_ratios = df.set_index('userName')['mediaStatusRatio'].to_dict()
    return media_status_ratios


def calculate_tweet_frequencies(df):
    """
    Calculates tweet frequencies (daily, weekly, monthly) and average rate of tweets for each user.

    Parameters:
    df (pandas.DataFrame): DataFrame containing tweet data with 'createdAt' column.

    Returns:
    dict: A dictionary with tweet frequencies and average rate for each user.
    """
     
    # Parse createdAt to datetime
    df['tweetCreatedAt'] = pd.to_datetime(df['createdAt'], format="%a %b %d %H:%M:%S %z %Y", errors='coerce')
    
    # Group by user
    user_frequencies = {}
    for user, group in df.groupby('userName'):
        if group.empty:
            continue
        
        total_tweets = len(group)
        unique_days = group['tweetCreatedAt'].dt.date.nunique()
        unique_weeks = group['tweetCreatedAt'].dt.isocalendar().week.nunique()
        unique_months = group['tweetCreatedAt'].dt.to_period('M').nunique()
        
        min_date, max_date = group['tweetCreatedAt'].min(), group['tweetCreatedAt'].max()
        total_days = (max_date - min_date).days + 1 if pd.notnull(min_date) and pd.notnull(max_date) else 1
        
        # Compute frequencies
        daily_frequency = total_tweets / unique_days if unique_days else 0
        weekly_frequency = total_tweets / unique_weeks if unique_weeks else 0
        monthly_frequency = total_tweets / unique_months if unique_months else 0
        avg_rate_of_tweets = total_tweets / total_days if total_days else 0
        
        # Store the result
        user_frequencies[user] = (
            daily_frequency,
            weekly_frequency,
            monthly_frequency,
            avg_rate_of_tweets
        )
    
    return user_frequencies


def analyze_user_tweets(user_tweet_dict):
    """
    Analyzes user tweets, calculating the first and last tweet dates, active days, and total tweets.

    Parameters:
    user_tweet_dict (dict): Dictionary containing tweet data for each user.

    Returns:
    pandas.DataFrame: A DataFrame with analysis results like first tweet date, last tweet date, and total tweets.
    """

    result = []

    for user, tweets in user_tweet_dict.items():
        if not tweets:
            continue
        
        # Extract createdAt dates and convert to datetime
        dates = [tweet['createdAt'] for tweet in tweets]
        
        # Calculate first and last tweet dates
        first_date = min(dates)
        last_date = max(dates)
        
        # Unique active days
        unique_active_days = len(set(date.date() for date in dates))
        
        # Total tweets count
        total_tweets = len(tweets)
        
        # Append results
        result.append({
            'user_name': user,
            'first_tweet_date': first_date,
            'last_tweet_date': last_date,
            'days_active': unique_active_days,
            'total_tweets': total_tweets  # New column for tweet count
        })
    
    # Create DataFrame
    df = pd.DataFrame(result)

    # Calculate additional metrics
    df['time_span'] = (df['last_tweet_date'] - df['first_tweet_date']).dt.days.replace(0, 1)  # Avoid division by zero
    df['active_days_ratio'] = df['days_active'] / df['time_span']
    df['tweets_per_active_day'] = df['total_tweets'] / df['days_active']
    
    return df


def cal_activity_persistence_score(df):
    """
    Calculates an activity persistence score based on tweet activity and engagement over time.

    Parameters:
    df (pandas.DataFrame): DataFrame containing user tweet data with 'first_tweet_date', 'last_tweet_date', etc.

    Returns:
    pandas.DataFrame: A DataFrame with the activity persistence score for each user.
    """

    # Convert date columns to datetime format
    df['first_tweet_date'] = pd.to_datetime(df['first_tweet_date'])
    df['last_tweet_date'] = pd.to_datetime(df['last_tweet_date'])

    # Calculate the time span of tweets for each account
    df['time_span'] = (df['last_tweet_date'] - df['first_tweet_date']).dt.days
    df['time_span'] = df['time_span'].replace(0, 1)  # Avoid division by zero

    # Calculate persistence score
    total_days_in_dataset = (df['last_tweet_date'].max() - df['first_tweet_date'].min()).days + 1
    df['persistence_score'] = df['days_active'] / total_days_in_dataset

    # Calculate tweets per active day (avoid division by zero)
    df['tweets_per_active_day'] = df['total_tweets'] / df['days_active']
    df['tweets_per_active_day'] = df['tweets_per_active_day'].replace([float('inf'), -float('inf')], 0)

    # Calculate tweets per day ratio based on total dataset duration
    df['tweets_per_day_ratio'] = df['total_tweets'] / total_days_in_dataset

    # Compute the new activity score based on weighted values
    df['activity_score'] = (
        (df['tweets_per_active_day'] * 0.8) +
        (df['tweets_per_day_ratio'] * 0.2)
    )

    # Normalize the calculated activity score for comparison while preventing values from going to zero
    df['activity_score'] = ((df['activity_score'] - df['activity_score'].min()) / (
        df['activity_score'].max() - df['activity_score'].min()
    )) * 0.9 + 0.1

    return df


def avg_time_tweets(df):
    """
    Calculates the average time between tweets for each user.

    Parameters:
    df (pandas.DataFrame): DataFrame containing tweet data with 'createdAt' column.

    Returns:
    dict: A dictionary with the average time between tweets for each user.
    """

    # Sort by user and timestamp
    sorted_df = df.sort_values(by=['userName', 'createdAt'])

    # Calculate the time difference between consecutive tweets
    sorted_df['timeDiff'] = sorted_df.groupby('userName')['createdAt'].diff().dt.total_seconds() / 3600  # In hours

    # Calculate the average time difference per author (ignoring NaNs)
    avg_time_per_author = sorted_df.groupby('userName')['timeDiff'].mean().reset_index(name='avgTimeBetweenTweets')
    # For single tweet users the average time will be 0
    avg_time_per_author['avgTimeBetweenTweets'] = avg_time_per_author['avgTimeBetweenTweets'].fillna(0)

    avg_time_dict = avg_time_per_author.set_index('userName')['avgTimeBetweenTweets'].to_dict()

    return avg_time_dict


def calculate_originality_ratio(df):
    """
    Calculates the originality ratio of tweets based on the tweet type (e.g., retweets, replies, quotes).

    Parameters:
    df (pandas.DataFrame): DataFrame containing tweet data with columns like 'isRetweet', 'isReply', etc.

    Returns:
    dict: A dictionary with the normalized originality ratio for each user.
    """

    def get_tweet_score(row):
        if row['is_retweet']:
            score = 0
        elif row['is_reply']:
            score = 1
        elif row['is_quote']:
            score = 2
        elif not (row['is_retweet'] or row['is_reply'] or row['is_quote']):
            score = 3
        
        # Add media bonus
        if row['media_urls']:
            score += 0.5
        
        return score

    df['tweetScore'] = df.apply(get_tweet_score, axis=1)

    # Group by username
    grouped = df.groupby('userName').agg(
        totalScore=('tweetScore', 'sum'),
        totalTweets=('tweetScore', 'count')
    ).reset_index()

    # Calculate originality ratio
    grouped['originalityRatio'] = grouped['totalScore'] / grouped['totalTweets']
    
    # Normalize the ratio between 0 and 1
    min_ratio = grouped['originalityRatio'].min()
    max_ratio = grouped['originalityRatio'].max()
    
    grouped['normalizedOriginalityRatio'] = (
        (grouped['originalityRatio'] - min_ratio) / (max_ratio - min_ratio)
    ).fillna(0)

    result_dict = grouped.set_index('userName')['normalizedOriginalityRatio'].to_dict()

    return result_dict


def calculate_human_device_ratio(df):
    """
    Calculates the ratio of tweets from human devices (e.g., mobile, desktop) compared to total tweets.

    Parameters:
    df (pandas.DataFrame): DataFrame containing tweet data with the 'source' column.

    Returns:
    dict: A dictionary with the human device ratio for each user.
    """

    possible_human_source = [
        "Twitter Web App",
        "Twitter for Android",
        "Twitter for iPhone",
        "Twitter for iPad",
        "Twitter for Mac",
        "TweetDeck",
        "TweetDeck Web App"
    ]

    human_device_ratio = {}
    
    for username, group in df.groupby('userName'):
        total_tweets = len(group)
        human_tweets = group['source'].isin(possible_human_source).sum()
        
        ratio = human_tweets / total_tweets if total_tweets > 0 else 0
        
        human_device_ratio[username] = ratio
    
    return human_device_ratio


def avg_reach(df):
    """
    Calculates the average tweet reach (view count) for each user and normalizes the value.
    """
    avg_reach_series = df.groupby("userName")["view_count"].mean().fillna(0)

    min_val = avg_reach_series.min()
    max_val = avg_reach_series.max()

    if max_val == min_val:
        # Avoid division by zero — all users have the same average reach
        avg_reach_normalized = pd.Series(0.0, index=avg_reach_series.index)
    else:
        avg_reach_normalized = (avg_reach_series - min_val) / (max_val - min_val)

    return avg_reach_normalized.to_dict()

def follower_to_following_ratio(df):
    """
    Calculates the follower-to-following ratio for each user and normalizes the value.

    Parameters:
    df (pandas.DataFrame): DataFrame containing user data with 'followers' and 'following' columns.

    Returns:
    dict: A dictionary with the normalized follower-to-following ratio for each user.
    """

    df["follower_following_ratio"] = df["followers"] / (df["following"] + 1)

    df.set_index('userName')['follower_following_ratio'].to_dict()

    df["follower_following_ratio_normalized"] = (
        (df["follower_following_ratio"] - df["follower_following_ratio"].min()) / 
        (df["follower_following_ratio"].max() - df["follower_following_ratio"].min())
    )

    follower_to_following_ratio_normalized = df.set_index("userName")["follower_following_ratio_normalized"].to_dict()

    return follower_to_following_ratio_normalized


def is_single_day_user(df):
    """
    Identifies users who tweeted only on a single day and calculates their activity score.

    Parameters:
    df (pandas.DataFrame): DataFrame containing tweet data with 'first_tweet_date' and 'last_tweet_date'.

    Returns:
    dict: A dictionary with the account activity score for each user.
    """

    df["single_day_user"] = df["first_tweet_date"].dt.date == df["last_tweet_date"].dt.date

    df["account_activity"] = (
        df["activity_score"] * 0.75 +
        df["persistence_score"] * 0.25 -
        (df["single_day_user"] * 0.05)
    )

    account_activity_dict = df.set_index("user_name")[["account_activity"]].apply(tuple, axis=1).to_dict()

    return account_activity_dict


def blueCheck(df):
    """
    Checks if a user has a blue verified check and returns the result.

    Parameters:
    df (pandas.DataFrame): DataFrame containing user profile data with 'isBlueVerified' column.

    Returns:
    dict: A dictionary with the blue check status for each user.
    """
    blueCheck_dict = df.set_index("userName")[["is_verified"]].apply(tuple, axis=1).to_dict()

    return blueCheck_dict


def compute_author_ratios(df):
    """
    Computes the ratio of 'statistical' vs 'emotional' tweet types for each user.

    Parameters:
    df (pandas.DataFrame): DataFrame containing tweet data with 'tweet_type' column.

    Returns:
    dict: A dictionary with the 'statistical' vs 'emotional' tweet ratio for each user.
    """

    # Group by userName and count tweet types
    tweet_counts = df.groupby("userName")["tweet_type"].value_counts().unstack(fill_value=0)

    tweet_counts["total_tweets"] = tweet_counts.sum(axis=1)

    tweet_counts["stat_vs_emot_ratio"] = tweet_counts["statistical"] / tweet_counts["total_tweets"]

    ratio_dict = tweet_counts["stat_vs_emot_ratio"].to_dict()
    return ratio_dict


def compute_hist_cmp(df):
    """
    Computes the ratio of historical comparison ('present' vs 'past') tweets for each user.

    Parameters:
    df (pandas.DataFrame): DataFrame containing tweet data with 'historical_comparison' column.

    Returns:
    dict: A dictionary with the historical comparison ratio for each user.
    """

    # Group by userName and count tweet types
    tweet_counts = df.groupby("userName")["historical_comparison"].value_counts().unstack(fill_value=0)

    tweet_counts["total_tweets"] = tweet_counts.sum(axis=1)

    tweet_counts["historical_comparison_ratio"] = tweet_counts.get("present", 0) / tweet_counts["total_tweets"]

    ratio_dict = tweet_counts["historical_comparison_ratio"].to_dict()
    
    return ratio_dict


def compute_market_hint_ratio(df):
    """
    Computes the ratio of market hint ('signal') tweets for each user.
    Returns 0 for users where 'market_hint' or 'signal' is missing.
    """

    # Check for required columns
    if 'userName' not in df.columns or 'market_hint' not in df.columns:
        return defaultdict(lambda: 0)

    # Count tweet types per user
    tweet_counts = df.groupby("userName")["market_hint"].value_counts().unstack(fill_value=0)

    # Ensure the 'signal' column exists
    if 'signal' not in tweet_counts.columns:
        tweet_counts['signal'] = 0

    # Calculate total tweets
    tweet_counts["total_tweets"] = tweet_counts.sum(axis=1)

    # Compute ratio safely
    tweet_counts["market_hint_ratio"] = tweet_counts.apply(
        lambda row: row["signal"] / row["total_tweets"] if row["total_tweets"] > 0 else 0,
        axis=1
    )

    # Return safe dictionary
    return defaultdict(lambda: 0, tweet_counts["market_hint_ratio"].to_dict())



def signal_classification_ratio(df):
    """
    Computes the ratio of signal classifications (bearish, bullish, normal) for each user and calculates a score.

    Parameters:
        df (pandas.DataFrame): DataFrame containing tweet data with 'signal_classification' column.

    Returns:
        dict: A dictionary with the signal classification score for each user.
    """
    category_columns = ["signal_classification"]
    category_ratios_list = []

    for col in category_columns:
        temp_df = df.groupby("userName")[col].value_counts(normalize=True).unstack(fill_value=0)
        temp_df.columns = [f"{col}_{val}_ratio" for val in temp_df.columns]
        category_ratios_list.append(temp_df)

    final_ratios = pd.concat(category_ratios_list, axis=1).reset_index()

    final_ratios["signal_classification_score"] = (
        final_ratios.get("signal_classification_bearish_ratio", 0) * 0 +
        final_ratios.get("signal_classification_bullish_ratio", 0) * 1 +
        final_ratios.get("signal_classification_normal_ratio", 0) * 0.5
    )

    ratio_dict = dict(zip(final_ratios["userName"], final_ratios["signal_classification_score"]))

    return ratio_dict


def cta_ratio(df):
    """
    Calculates a call-to-action (CTA) ratio for each user based on their activity in the 'call_to_action' column.
    The function calculates normalized ratios for each CTA category (buy, sell, none, hold) and computes a score 
    based on weighted values.

    Args:
        df (pd.DataFrame): DataFrame containing user tweets and their respective CTA actions.

    Returns:
        dict: A dictionary where keys are user names and values are their corresponding CTA scores.
    """

    category_columns = ["call_to_action"]
    expected_cta_types = ["buy", "sell", "none", "hold"]
    
    category_ratios_list = []

    for col in category_columns:
        temp_df = df.groupby("userName")[col].value_counts(normalize=True).unstack(fill_value=0)
        temp_df.columns = [f"{col}_{val}_ratio" for val in temp_df.columns]
        
        # Add missing ratio columns with 0
        for val in expected_cta_types:
            col_name = f"{col}_{val}_ratio"
            if col_name not in temp_df.columns:
                temp_df[col_name] = 0
        
        category_ratios_list.append(temp_df)

    final_ratios = pd.concat(category_ratios_list, axis=1).reset_index()

    final_ratios["call_to_action_score"] = (
        final_ratios["call_to_action_buy_ratio"] * 1 +
        final_ratios["call_to_action_sell_ratio"] * 0.2 +
        final_ratios["call_to_action_none_ratio"] * 0 +
        final_ratios["call_to_action_hold_ratio"] * 0.5
    )

    return dict(zip(final_ratios["userName"], final_ratios["call_to_action_score"]))


def cal_hype_classification(df):
    """
    Calculates a hype classification score for each user based on the 'hype_classification' column.
    The function computes normalized ratios for each hype classification category (high, normal, low) and 
    assigns a score to each user.

    Args:
        df (pd.DataFrame): DataFrame containing user tweets and their respective hype classifications.

    Returns:
        dict: A dictionary where keys are user names and values are their corresponding hype classification scores.
    """
    category_columns = ["hype_classification"]
    category_ratios_list = []

    for col in category_columns:
        temp_df = df.groupby("userName")[col].value_counts(normalize=True).unstack(fill_value=0)
        temp_df.columns = [f"{col}_{val}_ratio" for val in temp_df.columns]
        category_ratios_list.append(temp_df)

    final_ratios = pd.concat(category_ratios_list, axis=1).reset_index()

    final_ratios["hype_classification_score"] = (
        final_ratios.get("hype_classification_high_ratio", 0) * 0 +
        final_ratios.get("hype_classification_normal_ratio", 0) * 0.5 +
        final_ratios.get("hype_classification_low_ratio", 0) * 1
    )

    ratio_dict = dict(zip(final_ratios["userName"], final_ratios["hype_classification_score"]))

    return ratio_dict


def absense_crypto_manipulation(df):
    """
    Calculates the normalized count of false occurrences in the 'crypto_manipulative_words' column per user,
    which indicates the absence of manipulative behavior. 

    Args:
        df (pd.DataFrame): DataFrame containing user tweets and the 'crypto_manipulative_words' indicator.

    Returns:
        dict: A dictionary where keys are user names and values are their normalized absence counts of manipulative behavior.
    """
    # Count occurrences of False in the 'crypto_manipulative_words' column per user
    false_counts = df.groupby("userName")["crypto_manipulative_words"].apply(lambda x: (~x.astype(bool)).sum()).reset_index()

    # Rename the column for clarity
    false_counts.rename(columns={"crypto_manipulative_words": "false_count_crypto_manipulative_words"}, inplace=True)

    # Normalize false_count_crypto_manipulative_words (Min-Max Scaling)
    min_val = false_counts["false_count_crypto_manipulative_words"].min()
    max_val = false_counts["false_count_crypto_manipulative_words"].max()

    # Avoid division by zero if all values are the same
    if min_val != max_val:
        false_counts["normalized_false_count"] = (false_counts["false_count_crypto_manipulative_words"] - min_val) / (max_val - min_val)
    else:
        false_counts["normalized_false_count"] = 1  # If all values are the same, set them to 1

    false_count_dict = false_counts.set_index("userName")[["normalized_false_count"]].apply(tuple, axis=1).to_dict()

    return false_count_dict


def cal_urgency_level_score(df):
    """
    Calculates an urgency level score for each user based on the 'urgency_level' column. The function computes 
    normalized ratios for each urgency level category (high, medium, low) and assigns a weighted score to each user.

    Args:
        df (pd.DataFrame): DataFrame containing user tweets and their respective urgency levels.

    Returns:
        dict: A dictionary where keys are user names and values are their corresponding urgency level scores.
    """
    category_columns = ["urgency_level"]
    category_ratios_list = []

    for col in category_columns:
        temp_df = df.groupby("userName")[col].value_counts(normalize=True).unstack(fill_value=0)
        temp_df.columns = [f"{col}_{val}_ratio" for val in temp_df.columns]
        category_ratios_list.append(temp_df)

    final_ratios = pd.concat(category_ratios_list, axis=1).reset_index()

    final_ratios["urgency_level_score"] = (
        final_ratios.get("urgency_level_high_ratio", 0) * 1 +
        final_ratios.get("urgency_level_medium_ratio", 0) * 0.3 +
        final_ratios.get("urgency_level_low_ratio", 0) * 0
    )
    
    ratio_dict = dict(zip(final_ratios["userName"], final_ratios["urgency_level_score"]))

    return ratio_dict


def prediction_ratio(df, tenx_df, non_tenx_df):
    """
    Calculates the prediction ratio for each user based on their coin mentions and compares them with the 
    corresponding all-time-high (ATH) dates of coins. It counts the total and successful predictions (before ATH).

    Args:
        df (pd.DataFrame): DataFrame containing user tweets and coin mentions.
        tenx_df (pd.DataFrame): DataFrame containing coin details with ATH dates.
        non_tenx_df (pd.DataFrame): DataFrame containing non-10x coin data.

    Returns:
        pd.DataFrame: A DataFrame with additional columns for total predictions and successful predictions for each user.
    """
    import ast

    df = extract_coin_mentions(df, tenx_df, non_tenx_df)

    # Convert timestamps to datetime objects
    df['tweet_date'] = pd.to_datetime(df['createdAt'], errors='coerce')
    tenx_df['all_time_high_date'] = pd.to_datetime(tenx_df['ath_date'], errors='coerce')

    # Create a dictionary of coin name to ATH date for faster lookup
    coin_ath_dict = dict(zip(tenx_df['name'].str.lower(), tenx_df['all_time_high_date']))

    # Function to analyze predictions
    def analyze_predictions(row):
        try:
            if isinstance(row['coin_mentions'], list):
                coins = row['coin_mentions']
            else:
                coins = ast.literal_eval(row['coin_mentions'])
            
            # Count total predictions
            total_predictions = len(coins)
            
            # Count successful predictions (coins mentioned before ATH)
            successful_predictions = sum(
                1 for coin in coins 
                if row['tweet_date'] < coin_ath_dict.get(coin.lower(), pd.NaT)
            )
            
            return pd.Series([total_predictions, successful_predictions])
        except Exception:
            return pd.Series([0, 0])

    # Apply the function to create new columns
    df[['total_predictions', 'successful_predictions']] = df.apply(analyze_predictions, axis=1)

    return df


def successful_10x_ratio(df):
    """
    Calculates the ratio of successful 10x predictions for each user. The function uses the prediction ratio 
    and computes a normalized success ratio based on the user's successful predictions.

    Args:
        df (pd.DataFrame): DataFrame containing user tweets and their prediction data.

    Returns:
        dict: A dictionary where keys are user names and values are their corresponding successful 10x prediction ratios.
    """
    tenx_df, non_tenx_df = fetch_coin_dataset()
    df = prediction_ratio(df, tenx_df, non_tenx_df)

    user_stats = df.groupby('userName').agg({
        'total_predictions': 'sum',
        'successful_predictions': 'sum',
        'id': 'count'  # Count of tweets per user
    }).reset_index()

    # Calculate success ratio
    user_stats['success_ratio'] = user_stats['successful_predictions'] * 2 / user_stats['total_predictions'] * 5
    user_stats['successful_10x_predictions_ratio'] = user_stats['success_ratio'].fillna(0)  # Handle division by zero

    min_val = user_stats["successful_10x_predictions_ratio"].min()
    max_val = user_stats["successful_10x_predictions_ratio"].max()

    if min_val == max_val:
        user_stats["successful_10x_predictions_ratio_normalized"] = 1.0  # or 0.5, depending on preference
    else:
        user_stats["successful_10x_predictions_ratio_normalized"] = (
            (user_stats["successful_10x_predictions_ratio"] - min_val) /
            (max_val - min_val)
        )

    # Rename columns for clarity
    user_stats = user_stats.rename(columns={'id': 'tweet_count'})

    return dict(zip(user_stats['userName'], user_stats['successful_10x_predictions_ratio_normalized']))


def longest_tweet_streak(df):
    """
    Calculates the longest streak of non-zero predictions (successful predictions) for each user and normalizes it.

    Args:
        df (pd.DataFrame): DataFrame containing user tweets and their successful prediction status.

    Returns:
        dict: A dictionary where keys are user names and values are the normalized longest streak of successful predictions.
    """
    # Convert createdAt to datetime
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Step 1: Group by username and sort by createdAt
    df = df.sort_values(['userName', 'createdAt'])

    # Step 2: Identify streaks using cumsum
    df['streak_id'] = df.groupby('userName')['successful_predictions'].transform(lambda x: (x == 0).cumsum())

    # Filter out zeros (breaking points)
    streaks = df[df['successful_predictions'] != 0].groupby(['userName', 'streak_id'])

    # Calculate longest streak
    streak_lengths = streaks['successful_predictions'].count().reset_index(name='streak_length')
    max_streaks = streak_lengths.groupby('userName')['streak_length'].max().reset_index(name='longest_tweets_streak')

    # Manual Min-Max Normalization
    min_val = max_streaks['longest_tweets_streak'].min()
    max_val = max_streaks['longest_tweets_streak'].max()

    if max_val == min_val:
        # Avoid division by zero; assign 1.0 to all if all values are same
        max_streaks['normalized_streak'] = 1.0
    else:
        max_streaks['normalized_streak'] = (max_streaks['longest_tweets_streak'] - min_val) / (max_val - min_val)

    return max_streaks.set_index('userName')["normalized_streak"].to_dict()


def incorrect_buy_signals(df):
    """
    Identifies and calculates the inverse of incorrect buy signals for each user. This function analyzes 
    tweets with buy signals that did not result in successful predictions and normalizes the inverse success rate.

    Args:
        df (pd.DataFrame): DataFrame containing user tweets, buy signals, and prediction results.

    Returns:
        dict: A dictionary where keys are user names and values are their normalized inverse incorrect buy signal rates.
    """
    tenx_df, non_tenx_df = fetch_coin_dataset()
    df = prediction_ratio(df, tenx_df, non_tenx_df)
    
    buy_tweets_analysis = df.copy()
    buy_tweets_analysis[(buy_tweets_analysis["call_to_action"] == "buy")].head()

    buy_tweets_ids = buy_tweets_analysis["id"].to_list()
    buy_tweets_df = df[df["id"].isin(buy_tweets_ids)]

    buy_tweets_df = buy_tweets_df[buy_tweets_df["successful_predictions"] == 0]

    bullish_tweets_analysis = df.copy()
    bullish_tweets_analysis[(bullish_tweets_analysis["signal_classification"] == "bullish")].head()

    bullish_tweets_ids = bullish_tweets_analysis["id"].to_list()
    bullish_tweets_df = df[df["id"].isin(bullish_tweets_ids)]

    incorrect_buy_df = pd.concat([buy_tweets_df, bullish_tweets_df])

    incorrect_buy_df["incorrect_buy_signal"] = incorrect_buy_df["successful_predictions"].apply(lambda x: 1 if x == 0 else 0)

    false_prediction_df = incorrect_buy_df.groupby('userName').agg({
        'incorrect_buy_signal': 'sum',
        'id': 'count'  # Count of tweets per user
    }).reset_index()

    # Calculate success ratio
    false_prediction_df['incorrect_buy_signal_inverse'] = 1 - (false_prediction_df['incorrect_buy_signal'] / false_prediction_df['id'])
    false_prediction_df['incorrect_buy_signal_inverse'] = false_prediction_df['incorrect_buy_signal_inverse'].fillna(0)  # Handle division by zero

    # Rename columns for clarity
    false_prediction_df = false_prediction_df.rename(columns={'id': 'tweet_count'})

    # Sort by success ratio in descending order
    false_prediction_df = false_prediction_df.sort_values('incorrect_buy_signal_inverse', ascending=False)

    return false_prediction_df.set_index("userName")["incorrect_buy_signal_inverse"].to_dict()


def cal_time_to_5x_movement():
    """
    Calculates the time it took for Pendle's coin price to reach 2x, 3x, and 5x movement based on the user's first tweet 
    and the coin's closing price. The function computes a final weighted score based on the normalized times for each user.

    Returns:
        dict: A dictionary where keys are user names and values are their corresponding time-to-5x movement scores.
    """
    df = coin_only_tweets()
    pendle_coin_df = pendle_coin_dataset()
    first_tweets = df.sort_values("createdAt").groupby("userName").first().reset_index()
    first_tweets["rounded_timestamp"] = first_tweets["createdAt"].dt.floor("30T")

    first_tweets["rounded_timestamp"] = pd.to_datetime(first_tweets["rounded_timestamp"], utc=True)

    pendle_coin_df["TIMESTAMP"] = pd.to_datetime(pendle_coin_df["TIMESTAMP"], utc=True)

    merged_data = first_tweets.merge(
        pendle_coin_df, 
        left_on="rounded_timestamp", 
        right_on="TIMESTAMP", 
        how="left"
    )[["userName", "createdAt", "rounded_timestamp", "CLOSE"]]

    merged_data["CLOSE"].fillna(0.4, inplace=True)

    merged_data["createdAt"] = pd.to_datetime(merged_data["createdAt"])
    pendle_coin_df["TIMESTAMP"] = pd.to_datetime(pendle_coin_df["TIMESTAMP"])

    # Sort price data for efficient searching
    pendle_coin_df = pendle_coin_df.sort_values(by=["TIMESTAMP"])

    # Convert to numpy arrays for fast vectorized operations
    timestamps = pendle_coin_df["TIMESTAMP"].values
    prices = pendle_coin_df["CLOSE"].values

    # Vectorized function to find days to reach 2x, 3x, and 5x
    def find_days_to_targets_fast(created_at, start_price):
        try:
            if hasattr(created_at, 'tzinfo') and created_at.tzinfo is not None:
                created_at = created_at.tz_convert(None)

            if start_price <= 0:
                return [0, 0, 0]

            target_prices = [start_price * 2, start_price * 3, start_price * 5]
            days = [0, 0, 0]
            found = [False, False, False]

            for ts, price in zip(timestamps, prices):
                if ts >= np.datetime64(created_at):
                    for i, target in enumerate(target_prices):
                        if not found[i] and price >= target:
                            diff = (ts - np.datetime64(created_at)).astype('timedelta64[D]').astype(int)
                            days[i] = diff
                            found[i] = True
                    if all(found):
                        break

            return days

        except Exception as e:
            return [0, 0, 0]


    # def safe_find_days(row):
    #     try:
    #         if pd.isnull(row["rounded_timestamp"]) or pd.isnull(row["CLOSE"]):
    #             return [0, 0, 0]
    #         res = find_days_to_targets_fast(row["rounded_timestamp"], row["CLOSE"])
    #         # Force correct format
    #         if isinstance(res, (list, np.ndarray)) and len(res) == 3:
    #             return list(map(int, res))
    #         else:
    #             return [0, 0, 0]
    #     except Exception as e:
    #         print(f"Safe fail for {row.get('userName', 'unknown')}: {e}")
    #         return [0, 0, 0]


    # # Apply with pd.Series for safe unpacking
    # merged_data[["Days_to_2x", "Days_to_3x", "Days_to_5x"]] = merged_data.apply(
    #     lambda row: pd.Series(safe_find_days(row)), axis=1
    # )

    # print(merged_data.apply(lambda row: find_days_to_targets_fast(row["rounded_timestamp"], row["CLOSE"]), axis=1))

    # Apply function efficiently
    merged_data[["Days_to_2x", "Days_to_3x", "Days_to_5x"]] = np.vstack(
        merged_data.apply(lambda row: find_days_to_targets_fast(row["rounded_timestamp"], row["CLOSE"]), axis=1)
    )

    for col in ["Days_to_2x", "Days_to_3x", "Days_to_5x"]:
        col_min = merged_data[col].min()
        col_max = merged_data[col].max()

        if col_max != col_min:
            merged_data[col + "_norm"] = 1 - (merged_data[col] - col_min) / (col_max - col_min)
        else:
            merged_data[col + "_norm"] = 1  # or 0 or 1 or np.nan, depending on your logic

    # Calculate the final weighted score
    merged_data["time_to_5x_movement"] = (
        0.6 * merged_data["Days_to_5x_norm"] + 
        0.25 * merged_data["Days_to_3x_norm"] + 
        0.15 * merged_data["Days_to_2x_norm"]
    )
    
    return merged_data.set_index("userName")["time_to_5x_movement"].to_dict()


def find_surges(pendle_coin_df):
    """
    Identifies surge periods in Pendle's coin data based on changes in price and volume. A surge is defined 
    when both the price and volume experience a significant increase.

    Args:
        pendle_coin_df (pd.DataFrame): DataFrame containing coin price and volume data.

    Returns:
        list: A list of tuples, each containing the start and end timestamps of a surge period.
    """
    pendle_coin_df['TIMESTAMP'] = pd.to_datetime(pendle_coin_df['TIMESTAMP'])
    pendle_coin_df.set_index('TIMESTAMP', inplace=True)
    # Ensure datetime index
    pendle_coin_df.index = pd.to_datetime(pendle_coin_df.index)

    # Calculate percentage changes for CLOSE and VOLUME
    pendle_coin_df['CLOSE_pct_change'] = pendle_coin_df['CLOSE'].pct_change() * 100
    pendle_coin_df['VOLUME_pct_change'] = pendle_coin_df['VOLUME'].pct_change() * 100

    # Compute relaxed surge thresholds (80th percentile)
    close_threshold_80 = pendle_coin_df['CLOSE_pct_change'].quantile(0.80)
    volume_threshold_80 = pendle_coin_df['VOLUME_pct_change'].quantile(0.80)

    # Identify surges using relaxed thresholds
    pendle_coin_df['surge'] = (
        (pendle_coin_df['CLOSE_pct_change'] >= close_threshold_80) &
        (pendle_coin_df['VOLUME_pct_change'] >= volume_threshold_80)
    )

    # Find contiguous surge periods
    pendle_coin_df['surge_shift'] = pendle_coin_df['surge'].shift(1, fill_value=False)
    pendle_coin_df['start'] = pendle_coin_df['surge'] & ~pendle_coin_df['surge_shift']  # Start of a new surge
    pendle_coin_df['end'] = ~pendle_coin_df['surge'] & pendle_coin_df['surge_shift']    # End of a surge

    # Extract start and end timestamps, ensuring minimum surge duration
    surge_periods = []
    start_time = None
    min_surge_duration = pd.Timedelta(hours=1)  # Minimum surge duration

    for timestamp, row in pendle_coin_df.iterrows():
        if row['start']:
            start_time = timestamp
        if row['end'] and start_time is not None:
            duration = timestamp - start_time
            if duration >= min_surge_duration:
                surge_periods.append((start_time, timestamp))
            start_time = None

    # Handle case where last surge doesn't have an end yet
    if start_time is not None and pendle_coin_df['surge'].iloc[-1]:
        end_time = pendle_coin_df.index[-1]
        duration = end_time - start_time
        if duration >= min_surge_duration:
            surge_periods.append((start_time, end_time))

    return surge_periods


def lead_time_before_price_surges(bullish_tweets_df, coin_df):
    """
    Calculates the average lead time (in days) before a price surge for each user who tweeted bullish signals. 
    The function compares the timestamp of the user's last tweet before a surge with the surge's start time.

    Args:
        bullish_tweets_df (pd.DataFrame): DataFrame containing bullish tweets with timestamps.
        coin_df (pd.DataFrame): DataFrame containing coin price data with surge periods.

    Returns:
        dict: A dictionary where keys are user names and values are their corresponding average days before a price surge.
    """
    surge_periods = find_surges(coin_df)
    # Convert 'createdAt' to datetime and sort
    bullish_tweets_df['createdAt'] = pd.to_datetime(bullish_tweets_df['createdAt'])
    bullish_tweets_df = bullish_tweets_df.sort_values(by=['userName', 'createdAt'])

    # Convert surge periods to DataFrame with proper datetime format
    surge_df = pd.DataFrame(surge_periods, columns=['surge_start', 'surge_end'])
    surge_df['surge_start'] = pd.to_datetime(surge_df['surge_start'])
    surge_df['surge_end'] = pd.to_datetime(surge_df['surge_end'])

    # Find the last tweet before each surge for each user using a more efficient approach
    results = []

    # Create a dictionary for quick lookup of last tweets before surge
    user_tweet_dict = {}
    for username, group in bullish_tweets_df.groupby('userName'):
        user_tweet_dict[username] = group['createdAt'].values

    # Process each surge period
    for _, surge_row in surge_df.iterrows():
        surge_start = surge_row['surge_start']
        
        # Process each user for this surge period
        for username, tweet_times in user_tweet_dict.items():
            # Find last tweet before surge using numpy which is much faster
            mask = tweet_times < surge_start
            if mask.any():
                last_tweet = tweet_times[mask].max()
                days_before = (surge_start - pd.Timestamp(last_tweet)).days
            else:
                days_before = 0
            
            results.append({'userName': username, 'days_before_surge': days_before})

    # Convert results to DataFrame and calculate average
    df_days = pd.DataFrame(results)
    avg_days_df = df_days.groupby('userName', as_index=False)['days_before_surge'].mean()

    return avg_days_df.set_index("userName")["days_before_surge"].to_dict()


def success_rate_during_surge(pendle_llm_analysis_df):
    """
    Computes the success rate of 'buy' tweets during surge periods.

    The function calculates the ratio of 'buy' tweets relative to all tweets for each user
    before and during surge periods. The ratios are then normalized for comparison.

    Parameters:
        pendle_llm_analysis_df (pd.DataFrame): DataFrame containing tweet data with user actions.

    Returns:
        dict: A dictionary mapping each user to their normalized 'buy' tweet ratio during surge periods.
    """
    coin_df = pendle_coin_dataset()
    surge_periods = find_surges(coin_df)
    pendle_mentioned_df = coin_only_tweets()
    # Define lookback period for pre-surge tweets
    lookback_period = pd.Timedelta(hours=72)

    # Getting IDs of all Buy tweets
    buy_tweets_ids = pendle_llm_analysis_df[pendle_llm_analysis_df["call_to_action"] == "buy"]["id"].to_list()

    # Function to compute buy tweet ratio
    def compute_buy_tweet_ratio(df):
        total_tweets = df.groupby("userName")["id"].count()
        buy_tweets = df[df["id"].isin(buy_tweets_ids)].groupby("userName")["id"].count()
        return (buy_tweets / total_tweets).fillna(0)

    # Temporary storage for merging all results
    all_pre_surge_ratios = []
    all_in_surge_ratios = []

    # Calculate buy tweet ratio before and during surges
    for start, end in surge_periods:
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)
        # Filter tweets during surge
        surge_tweets = pendle_mentioned_df[(pendle_mentioned_df['createdAt'] >= start) & (pendle_mentioned_df['createdAt'] <= end)]
        
        # Filter tweets 72 hours before surge
        # Filter tweets 72 hours before surge
        pre_surge_tweets = pendle_mentioned_df[
            (pendle_mentioned_df['createdAt'] >= start - lookback_period) &
            (pendle_mentioned_df['createdAt'] < start)
        ]

        # Compute buy tweet ratios
        pre_surge_buy_ratio = compute_buy_tweet_ratio(pre_surge_tweets)
        in_surge_buy_ratio = compute_buy_tweet_ratio(surge_tweets)
        
        # Store results for later aggregation
        all_pre_surge_ratios.append(pre_surge_buy_ratio)
        all_in_surge_ratios.append(in_surge_buy_ratio)

    # Combine all periods and sum them across users
    pre_surge_buy_ratios = pd.concat(all_pre_surge_ratios, axis=1).sum(axis=1)
    in_surge_buy_ratios = pd.concat(all_in_surge_ratios, axis=1).sum(axis=1)

    # Merge into final DataFrame
    buy_ratio_result = pd.DataFrame({
        "userName": pre_surge_buy_ratios.index.union(in_surge_buy_ratios.index),
        "presurge_buy_tweets_ratio": pre_surge_buy_ratios,
        "insurge_buy_tweets_ratio": in_surge_buy_ratios
    }).fillna(0)

    # Sort by In-Surge Buy Tweets Ratio (highest first)
    buy_ratio_result = buy_ratio_result.sort_values(by="insurge_buy_tweets_ratio", ascending=False)
    buy_ratio_result.reset_index(drop=True, inplace=True)

    # Normalize the in-surge buy tweet ratio (e.g., Min-Max scaling)
    max_val = buy_ratio_result["insurge_buy_tweets_ratio"].max()
    min_val = buy_ratio_result["insurge_buy_tweets_ratio"].min()
    if max_val > min_val:
        buy_ratio_result["normalized_insurge_buy_ratio"] = (
            (buy_ratio_result["insurge_buy_tweets_ratio"] - min_val) / (max_val - min_val)
        )
    else:
        buy_ratio_result["normalized_insurge_buy_ratio"] = 0  # or 1, depending on your logic

    # Convert to dictionary
    result_dict = dict(
        zip(buy_ratio_result["userName"], buy_ratio_result["normalized_insurge_buy_ratio"])
    )

    return result_dict


def lead_time_during_surge():
    """
    Calculates the lead time (tweet frequency) before and during surge periods.

    The function computes the frequency of tweets during surge periods and compares them with tweet frequencies
    in the 72 hours preceding the surge. The frequencies are normalized for comparison.

    Returns:
        dict: A dictionary mapping each user to their normalized tweet frequency during surge periods.
    """
    coin_df = pendle_coin_dataset()
    surge_periods = find_surges(coin_df)
    pendle_mentioned_df = coin_only_tweets()
    # Create an empty DataFrame to store tweet frequencies
    tweet_frequencies = []

    # Define lookback period for pre-surge tweets
    lookback_period = pd.Timedelta(hours=72)

    # Process each surge period
    for start, end in surge_periods:
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)

        # Filter tweets during surge
        surge_tweets = pendle_mentioned_df[(pendle_mentioned_df['createdAt'] >= start) & (pendle_mentioned_df['createdAt'] <= end)]
        surge_counts = surge_tweets.groupby("userName").size().reset_index(name="insurge_tweets_frequency")

        # Filter tweets 72 hours before surge
        pre_surge_tweets = pendle_mentioned_df[
            (pendle_mentioned_df['createdAt'] >= start - lookback_period) & 
            (pendle_mentioned_df['createdAt'] < start)
        ]
        pre_surge_counts = pre_surge_tweets.groupby("userName").size().reset_index(name="presurge_tweets_frequency")

        # Merge results to ensure all usernames are considered
        tweet_frequency = pd.merge(pre_surge_counts, surge_counts, on="userName", how="outer").fillna(0)

        # Append results
        tweet_frequencies.append(tweet_frequency)

    # Combine results into a single DataFrame
    tweet_frequency_result = pd.concat(tweet_frequencies, ignore_index=True)

    # Aggregate by username to get total tweets across all surge periods
    tweet_frequency_result = tweet_frequency_result.groupby("userName", as_index=False).sum()

    # Normalize the in-surge tweet frequency
    max_val = tweet_frequency_result["insurge_tweets_frequency"].max()
    min_val = tweet_frequency_result["insurge_tweets_frequency"].min()
    if max_val > min_val:
        tweet_frequency_result["normalized_insurge_frequency"] = (
            (tweet_frequency_result["insurge_tweets_frequency"] - min_val) / (max_val - min_val)
        )
    else:
        tweet_frequency_result["normalized_insurge_frequency"] = 0

    # Convert to dictionary
    result_dict = dict(
        zip(tweet_frequency_result["userName"], tweet_frequency_result["normalized_insurge_frequency"])
    )

    return result_dict


def tweet_frequency_ratio():
    """
    Computes the tweet frequency ratio before and during surge periods.

    This function calculates the ratio of tweet frequency before surge to tweet frequency during surge
    for each user. The ratio is then normalized for comparison.

    Returns:
        dict: A dictionary mapping each user to their normalized tweet frequency ratio.
    """
    coin_df = pendle_coin_dataset()
    surge_periods = find_surges(coin_df)
    pendle_mentioned_df = coin_only_tweets()
    tweet_frequencies = []

    lookback_period = pd.Timedelta(hours=72)

    for start, end in surge_periods:
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)

        surge_tweets = pendle_mentioned_df[
            (pendle_mentioned_df['createdAt'] >= start) & (pendle_mentioned_df['createdAt'] <= end)
        ]
        surge_counts = surge_tweets.groupby("userName").size().reset_index(name="insurge_tweets_frequency")

        pre_surge_tweets = pendle_mentioned_df[
            (pendle_mentioned_df['createdAt'] >= start - lookback_period) & (pendle_mentioned_df['createdAt'] < start)
        ]
        pre_surge_counts = pre_surge_tweets.groupby("userName").size().reset_index(name="presurge_tweets_frequency")

        tweet_frequency = pd.merge(pre_surge_counts, surge_counts, on="userName", how="outer").fillna(0)
        tweet_frequencies.append(tweet_frequency)

    tweet_frequency_result = pd.concat(tweet_frequencies, ignore_index=True)

    tweet_frequency_result = tweet_frequency_result.groupby("userName", as_index=False).sum()

    # Compute raw ratio (presurge / insurge), avoid division by zero
    tweet_frequency_result["frequency_ratio"] = tweet_frequency_result.apply(
        lambda row: row["presurge_tweets_frequency"] / row["insurge_tweets_frequency"]
        if row["insurge_tweets_frequency"] > 0 else 0,
        axis=1
    )

    # Normalize the frequency ratio
    max_val = tweet_frequency_result["frequency_ratio"].max()
    min_val = tweet_frequency_result["frequency_ratio"].min()
    if max_val > min_val:
        tweet_frequency_result["normalized_frequency_ratio"] = (
            (tweet_frequency_result["frequency_ratio"] - min_val) / (max_val - min_val)
        )
    else:
        tweet_frequency_result["normalized_frequency_ratio"] = 0

    # Convert to dictionary
    result_dict = dict(
        zip(tweet_frequency_result["userName"], tweet_frequency_result["normalized_frequency_ratio"])
    )

    return result_dict


def false_positive_rate_non_surge():
    """
    Calculates the tweet frequency before surge periods (as a proxy for false positives).

    The function computes the frequency of tweets in the 72 hours before surge periods
    and normalizes the values for comparison.

    Returns:
        dict: A dictionary mapping each user to their normalized pre-surge tweet frequency.
    """
    coin_df = pendle_coin_dataset()
    surge_periods = find_surges(coin_df)
    pendle_mentioned_df = coin_only_tweets()
    tweet_frequencies = []

    lookback_period = pd.Timedelta(hours=72)

    for start, end in surge_periods:
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)

        surge_tweets = pendle_mentioned_df[
            (pendle_mentioned_df['createdAt'] >= start) & (pendle_mentioned_df['createdAt'] <= end)
        ]
        surge_counts = surge_tweets.groupby("userName").size().reset_index(name="insurge_tweets_frequency")

        pre_surge_tweets = pendle_mentioned_df[
            (pendle_mentioned_df['createdAt'] >= start - lookback_period) & (pendle_mentioned_df['createdAt'] < start)
        ]
        pre_surge_counts = pre_surge_tweets.groupby("userName").size().reset_index(name="presurge_tweets_frequency")

        tweet_frequency = pd.merge(pre_surge_counts, surge_counts, on="userName", how="outer").fillna(0)
        tweet_frequencies.append(tweet_frequency)

    tweet_frequency_result = pd.concat(tweet_frequencies, ignore_index=True)

    tweet_frequency_result = tweet_frequency_result.groupby("userName", as_index=False).sum()

    # Normalize the presurge tweet frequency
    max_val = tweet_frequency_result["presurge_tweets_frequency"].max()
    min_val = tweet_frequency_result["presurge_tweets_frequency"].min()
    if max_val > min_val:
        tweet_frequency_result["normalized_presurge_frequency"] = (
            (tweet_frequency_result["presurge_tweets_frequency"] - min_val) / (max_val - min_val)
        )
    else:
        tweet_frequency_result["normalized_presurge_frequency"] = 0

    # Convert to dictionary
    result_dict = dict(
        zip(tweet_frequency_result["userName"], tweet_frequency_result["normalized_presurge_frequency"])
    )

    return result_dict


def hype_score_ratio():
    """
    Computes the hype score ratio during surge and non-surge periods.

    This function calculates the ratio of hype scores during surge periods to non-surge periods,
    normalizing the results for comparison. Hype scores are based on tweet classifications.

    Returns:
        dict: A dictionary mapping each user to their normalized hype score ratio.
    """
    tweets_analysis_merged_df = pendle_mentioned_df()
    surge_periods = find_surges(pendle_coin_dataset())

    # Ensure all 'createdAt' timestamps are timezone-aware
    tweets_analysis_merged_df['createdAt'] = pd.to_datetime(tweets_analysis_merged_df['createdAt'], utc=True)

    # Ensure all surge periods are timezone-aware
    surge_periods = [(pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)) for start, end in surge_periods]

    # Tag each tweet with whether it's in a surge period
    tweets_analysis_merged_df['in_surge_period'] = tweets_analysis_merged_df['createdAt'].apply(
        lambda x: any(start <= x <= end for start, end in surge_periods)
    )

    # Separate into surge and non-surge
    surge_tweets = tweets_analysis_merged_df[tweets_analysis_merged_df['in_surge_period']]
    non_surge_tweets = tweets_analysis_merged_df[~tweets_analysis_merged_df['in_surge_period']]

    # Define scoring criteria
    hype_scores = {"low": 0, "normal": 1, "high": 2}

    # Map to numerical scores
    surge_tweets["hype_score"] = surge_tweets["hype_classification"].map(hype_scores)
    non_surge_tweets["hype_score"] = non_surge_tweets["hype_classification"].map(hype_scores)

    # Aggregate scores
    surge_scores = surge_tweets.groupby("userName")["hype_score"].sum().reset_index(name="surge_score")
    non_surge_scores = non_surge_tweets.groupby("userName")["hype_score"].sum().reset_index(name="non_surge_score")

    # Merge on username
    hype_scores_merged = pd.merge(surge_scores, non_surge_scores, on="userName", how="outer").fillna(0)

    # Calculate the ratio (surge / non-surge)
    hype_scores_merged["hype_ratio"] = hype_scores_merged.apply(
        lambda row: row["surge_score"] / row["non_surge_score"] if row["non_surge_score"] > 0 else 0,
        axis=1
    )

    # Normalize the ratio
    max_val = hype_scores_merged["hype_ratio"].max()
    min_val = hype_scores_merged["hype_ratio"].min()
    if max_val > min_val:
        hype_scores_merged["normalized_hype_ratio"] = (
            (hype_scores_merged["hype_ratio"] - min_val) / (max_val - min_val)
        )
    else:
        hype_scores_merged["normalized_hype_ratio"] = 0

    # Convert to dict
    result_dict = dict(
        zip(hype_scores_merged["userName"], hype_scores_merged["normalized_hype_ratio"])
    )

    return result_dict


def manipulative_language_surge():
    """
    Calculates the ratio of manipulative language used during surge periods.

    The function computes the ratio of tweets containing manipulative language to total tweets
    during surge periods for each user.

    Returns:
        dict: A dictionary mapping each user to their manipulative tweet ratio during surge periods.
    """
    tweets_analysis_merged_df = pendle_mentioned_df()

    # Ensure 'createdAt' and 'in_surge_period' are set properly
    tweets_analysis_merged_df['createdAt'] = pd.to_datetime(tweets_analysis_merged_df['createdAt'], utc=True)
    surge_periods = find_surges(pendle_coin_dataset())
    surge_periods = [(pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)) for start, end in surge_periods]
    tweets_analysis_merged_df['in_surge_period'] = tweets_analysis_merged_df['createdAt'].apply(
        lambda x: any(start <= x <= end for start, end in surge_periods)
    )

    # Filter for tweets during surge
    surge_tweets = tweets_analysis_merged_df[tweets_analysis_merged_df['in_surge_period']]

    # Count total tweets and manipulative tweets per username
    manipulative_ratio = surge_tweets.groupby("userName").agg(
        total_tweets=("id", "count"),
        manipulative_tweets=("crypto_manipulative_words", "sum")
    )

    # Compute the ratio
    manipulative_ratio["manipulative_tweets_ratio"] = (
        manipulative_ratio["manipulative_tweets"] / manipulative_ratio["total_tweets"]
    )

    # Reset and convert to dict
    manipulative_ratio = manipulative_ratio.reset_index()
    result_dict = dict(
        zip(manipulative_ratio["userName"], manipulative_ratio["manipulative_tweets_ratio"])
    )

    return result_dict