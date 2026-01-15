import pandas as pd
from ..preprocessing.tweet_preprocessing import merge_tweet_analysis, pendle_coin_dataset, coin_only_tweets
from ..utils import get_unique_author_names, get_cutoff_date, unique_author_dataset, grouping_tweets
from ..feature_engineering.feature_engineering import absense_crypto_manipulation, account_age, account_engagement_ratio, analyze_user_tweets
from ..feature_engineering.feature_engineering import avg_reach, blueCheck, cal_activity_persistence_score, cal_hype_classification
from ..feature_engineering.feature_engineering import cal_time_to_5x_movement, cal_urgency_level_score, calculate_human_device_ratio
from ..feature_engineering.feature_engineering import calculate_media_status_ratio, calculate_originality_ratio
from ..feature_engineering.feature_engineering import calculate_profile_completeness, compute_author_ratios, compute_hist_cmp
from ..feature_engineering.feature_engineering import compute_market_hint_ratio, cta_ratio, false_positive_rate_non_surge, find_surges
from ..feature_engineering.feature_engineering import follower_to_following_ratio, hype_score_ratio, incorrect_buy_signals
from ..feature_engineering.feature_engineering import lead_time_before_price_surges, lead_time_during_surge, longest_tweet_streak
from ..feature_engineering.feature_engineering import manipulative_language_surge, signal_classification_ratio, success_rate_during_surge
from ..feature_engineering.feature_engineering import successful_10x_ratio, tweet_frequency_ratio


def calculate_verification_trust(df):
    """
    Calculates a trust score for each Twitter user based on various factors such as blue verification badge,
    account age, profile completeness, media status ratio, tweet frequency, content originality, and human-device ratio.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated trust score.
    """
    unique_usernames = get_unique_author_names(df)
    unique_author_df = unique_author_dataset(df)
    df_with_username, grouped_tweets = grouping_tweets(df)

    cutoff_date = get_cutoff_date(df)
    account_age_dict = account_age(unique_author_df, cutoff_date)
    profile_score_dict = calculate_profile_completeness(unique_author_df)
    media_dict = calculate_media_status_ratio(unique_author_df)

    result_df = analyze_user_tweets(grouped_tweets)
    presistence_activity_df = cal_activity_persistence_score(result_df)

    content_originality_ratio_dict = calculate_originality_ratio(df_with_username)
    human_device_ratio_dict = calculate_human_device_ratio(df_with_username)

    blueCheck_dict = blueCheck(unique_author_df)

    verification_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'blue_verification_badge': [blueCheck_dict.get(user, 0)[0] for user in unique_usernames],
        'account_age': [account_age_dict.get(user, 0) for user in unique_usernames],
        'profile_completeness': [profile_score_dict.get(user, 0) for user in unique_usernames],
        'media_status_ratio': [media_dict.get(user, 0) for user in unique_usernames],
        'tweets_frequency': [presistence_activity_df.get(user, (0, 0, 0, 0))[0] for user in unique_usernames],
        'content_originality_ratio': [content_originality_ratio_dict.get(user, 0) for user in unique_usernames],
        'human_source_device_ratio': [human_device_ratio_dict.get(user, 0) for user in unique_usernames]
    })

    verification_df["verification_trust"] = (
        verification_df["blue_verification_badge"].astype(int) * 0.20 +
        verification_df["account_age"] * 0.20 +
        verification_df["profile_completeness"] * 0.10 +
        verification_df["media_status_ratio"] * 0.10 +
        verification_df["tweets_frequency"] * 0.10 +
        verification_df["content_originality_ratio"] * 0.20 +
        verification_df["human_source_device_ratio"] * 0.10
    )

    trust_dict = verification_df.set_index("userName")["verification_trust"].to_dict()
    return trust_dict


def calculate_follower_quality(df):
    """
    Determines the quality of a user's followers using metrics like follower-to-following ratio,
    engagement ratio, and average reach.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated follower quality score.
    """
    unique_usernames = get_unique_author_names(df)
    unique_author_df = unique_author_dataset(df)
    df_with_username, grouped_tweets = grouping_tweets(df)

    account_engagement_dict = account_engagement_ratio(df_with_username)
    avg_reach_dict = avg_reach(df_with_username)

    follow_following_dict = follower_to_following_ratio(unique_author_df)

    follower_quality_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'follower_to_following_ratio': [follow_following_dict.get(user, 0) for user in unique_usernames],
        'engagement_ratio': [account_engagement_dict.get(user, 0) for user in unique_usernames],
        'avg_reach': [avg_reach_dict.get(user, 0) for user in unique_usernames]
    })

    follower_quality_df["follower_quality_raw"] = (
        follower_quality_df["follower_to_following_ratio"] * 0.35 +
        follower_quality_df["engagement_ratio"] * 0.50 +
        follower_quality_df["avg_reach"] * 0.15
    )

        # Apply Min-Max Normalization
    min_val = follower_quality_df["follower_quality_raw"].min()
    max_val = follower_quality_df["follower_quality_raw"].max()

    if min_val == max_val:
        # Avoid division by zero — assign 0 to all if constant
        follower_quality_df["follower_quality"] = 0.0
    else:
        follower_quality_df["follower_quality"] = (
            (follower_quality_df["follower_quality_raw"] - min_val) /
            (max_val - min_val)
        )

    # Convert to dictionary
    follower_dict = follower_quality_df.set_index("userName")["follower_quality"].to_dict()
    return follower_dict


def cal_data_driven_content(df):
    """
    Evaluates whether the user’s content is data-driven by analyzing the statistical vs emotional ratio
    and historical comparison presence ratio.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated data-driven content score.
    """
    unique_usernames = get_unique_author_names(df)
    author_dict = compute_author_ratios(df)
    hist_dict = compute_hist_cmp(df)

    data_driven_content_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'statistical_vs_emotional_ratio': [author_dict.get(user, 0) for user in unique_usernames],
        'historical_comparison_presence_ratio': [hist_dict.get(user, 0) for user in unique_usernames],
    })

    data_driven_content_df["data_driven_content"] = (
        data_driven_content_df["statistical_vs_emotional_ratio"] * 0.60 +
        data_driven_content_df["historical_comparison_presence_ratio"] * 0.40
    )

    data_driven_content_dict = data_driven_content_df.set_index("userName")["data_driven_content"].to_dict()
    
    return data_driven_content_dict


def cal_signal_clarity(df):
    """
    Computes a signal clarity score based on market hints, signal classification score, and call-to-action effectiveness.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated signal clarity score.
    """
    unique_usernames = get_unique_author_names(df)

    market_hint_dict = compute_market_hint_ratio(df)
    signal_dict = signal_classification_ratio(df)
    cta_dict = cta_ratio(df)

    signal_clarity_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'market_hint_ratio': [market_hint_dict.get(user, 0) for user in unique_usernames],
        'signal_classification_score': [signal_dict.get(user, 0) for user in unique_usernames],
        'call_to_action_score': [cta_dict.get(user, 0) for user in unique_usernames],
    })

    signal_clarity_df["signal_clarity"] = (
        signal_clarity_df["market_hint_ratio"] * 0.30 +
        signal_clarity_df["signal_classification_score"] * 0.40 +
        signal_clarity_df["call_to_action_score"] * 0.30
    )

    signal_clarity_df_dict = signal_clarity_df.set_index("userName")["signal_clarity"].to_dict()
    
    return signal_clarity_df_dict


def cal_manipulation_resistance(df):
    """
    Calculates a manipulation resistance score based on the absence of manipulative language and hype classification.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated manipulation resistance score.
    """
    unique_usernames = get_unique_author_names(df)

    absense_crypto_mani_dict = absense_crypto_manipulation(df)
    hype_dict = cal_hype_classification(df)

    manipulation_resistance_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'absence_of_manipulative_language_ratio': [absense_crypto_mani_dict.get(user, 0) for user in unique_usernames],
        'hype_classification': [hype_dict.get(user, 0) for user in unique_usernames]
    })

    manipulation_resistance_df["absence_of_manipulative_language_ratio"] = (
        manipulation_resistance_df["absence_of_manipulative_language_ratio"]
        .apply(lambda x: x[0] if isinstance(x, tuple) else x)
    )

    manipulation_resistance_df["manipulation_resistance"] = (
        manipulation_resistance_df["absence_of_manipulative_language_ratio"] * 0.5 +
        manipulation_resistance_df["hype_classification"] * 0.5
    )

    manipulation_resistance_df_dict = manipulation_resistance_df.set_index("userName")["manipulation_resistance"].to_dict()
    
    return manipulation_resistance_df_dict


def cal_urgency_sanity_check(df):
    """
    Provides a score for the urgency level associated with a user's tweets.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated urgency score.
    """
    urgency_score_dict = cal_urgency_level_score(df)

    return urgency_score_dict


def cal_prediction_success_rate(df):
    """
    Evaluates the prediction success rate based on the user's successful 10x prediction ratio
    and consecutive successful predictions.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated prediction success rate.
    """
    unique_usernames = get_unique_author_names(df)
    success_ratio_dict = successful_10x_ratio(df)
    longest_streak_dict = longest_tweet_streak(df)

    prediction_success_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'successful_10x_predictions_ratio': [success_ratio_dict.get(user, 0) for user in unique_usernames],
        'consecutive_successful_predictions': [longest_streak_dict.get(user, 0) for user in unique_usernames]
    })

    prediction_success_df["prediction_success_rate"] = (
        prediction_success_df["successful_10x_predictions_ratio"] * 0.70 +
        prediction_success_df["consecutive_successful_predictions"] * 0.30
    )

    prediction_success_df_dict = prediction_success_df.set_index("userName")["prediction_success_rate"].to_dict()
    
    return prediction_success_df_dict


def cal_false_prediction_rate(df):
    """
    Calculates the rate of false predictions, specifically for incorrect buy signals.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the false prediction rate.
    """

    return incorrect_buy_signals(df)


def cal_early_detection():
    """
    Measures early detection capability based on the time taken for a 5x price movement
    and lead time before price surges.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated early detection score.
    """
    df = coin_only_tweets()
    unique_usernames = get_unique_author_names(df)
    time_dict = cal_time_to_5x_movement()
    coin_df = pendle_coin_dataset()
    lead_dict = lead_time_before_price_surges(df, coin_df)

    prediction_success_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'time_to_5x_movement': [time_dict.get(user, 0) for user in unique_usernames],
        'lead_time_before_price_surges': [lead_dict.get(user, 0) for user in unique_usernames]
    })

    prediction_success_df["early_detection"] = (
        prediction_success_df["time_to_5x_movement"] * 0.70 +
        prediction_success_df["lead_time_before_price_surges"] * 0.30
    )

    prediction_success_df_dict = prediction_success_df.set_index("userName")["early_detection"].to_dict()
    
    return prediction_success_df_dict


def cal_consistency():
    """
    Assesses the consistency of a user's activity during surges, including tweet frequency and the ratio of buy
    and bullish tweets before and during surges.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated consistency score.
    """
    pendle_mentioned_df = coin_only_tweets()
    pendle_coin_df = pendle_coin_dataset()
    unique_users = get_unique_author_names(pendle_mentioned_df)
    surge_periods = find_surges(pendle_coin_df)

    # Lookback period for pre-surge analysis
    lookback_period = pd.Timedelta(hours=72)

    # Store tweet frequency during and before surges
    tweet_frequencies = []

    for start, end in surge_periods:
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)
        surge_tweets = pendle_mentioned_df[(pendle_mentioned_df['createdAt'] >= start) & (pendle_mentioned_df['createdAt'] <= end)]
        pre_surge_tweets = pendle_mentioned_df[
            (pendle_mentioned_df['createdAt'] >= start - lookback_period)
            & (pendle_mentioned_df['createdAt'] < start)
        ]

        surge_counts = surge_tweets.groupby("userName").size().reset_index(name="insurge_tweets_frequency")
        pre_surge_counts = pre_surge_tweets.groupby("userName").size().reset_index(name="presurge_tweets_frequency")

        combined = pd.merge(pre_surge_counts, surge_counts, on="userName", how="outer").fillna(0)
        tweet_frequencies.append(combined)

    if tweet_frequencies:
        tweet_frequency_result = pd.concat(tweet_frequencies, ignore_index=True)
        tweet_frequency_result = tweet_frequency_result.groupby("userName", as_index=False).sum()
    else:
        tweet_frequency_result = pd.DataFrame(columns=["userName", "presurge_tweets_frequency", "insurge_tweets_frequency"])

    # Get all buy tweet IDs
    buy_ids = pendle_mentioned_df[pendle_mentioned_df["call_to_action"] == "buy"]["id"].tolist()

    def compute_buy_ratio(df):
        if df.empty:
            return pd.Series(dtype='float64')
        total = df.groupby("userName")["id"].count()
        buys = df[df["id"].isin(buy_ids)].groupby("userName")["id"].count()
        ratio = (buys / total).fillna(0)
        return ratio.reindex(total.index, fill_value=0)

    all_pre_buy = []
    all_in_buy = []

    for start, end in surge_periods:
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)
        
        surge_tweets = pendle_mentioned_df[(pendle_mentioned_df['createdAt'] >= start) & (pendle_mentioned_df['createdAt'] <= end)]
        pre_tweets = pendle_mentioned_df[(pendle_mentioned_df['createdAt'] >= start - lookback_period) & (pendle_mentioned_df['createdAt'] < start)]

        all_pre_buy.append(compute_buy_ratio(pre_tweets))
        all_in_buy.append(compute_buy_ratio(surge_tweets))

    if all_pre_buy or all_in_buy:
        pre_buy_ratio = pd.concat(all_pre_buy, axis=1).sum(axis=1)
        in_buy_ratio = pd.concat(all_in_buy, axis=1).sum(axis=1)
        buy_ratio_result = pd.DataFrame({
            "userName": pre_buy_ratio.index.union(in_buy_ratio.index),
            "presurge_buy_tweets_ratio": pre_buy_ratio,
            "insurge_buy_tweets_ratio": in_buy_ratio
        }).fillna(0)
    else:
        buy_ratio_result = pd.DataFrame(columns=["userName", "presurge_buy_tweets_ratio", "insurge_buy_tweets_ratio"])

    # Same structure for bullish tweets
    bullish_ids = pendle_mentioned_df[pendle_mentioned_df["signal_classification"] == "bullish"]["id"].tolist()

    def compute_bullish_ratio(df):
        if df.empty:
            return pd.Series(dtype='float64')
        total = df.groupby("userName")["id"].count()
        bullish = df[df["id"].isin(bullish_ids)].groupby("userName")["id"].count()
        ratio = (bullish / total).fillna(0)
        return ratio.reindex(total.index, fill_value=0)

    all_pre_bull = []
    all_in_bull = []

    for start, end in surge_periods:
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)
        
        surge_tweets = pendle_mentioned_df[(pendle_mentioned_df['createdAt'] >= start) & (pendle_mentioned_df['createdAt'] <= end)]
        pre_tweets = pendle_mentioned_df[(pendle_mentioned_df['createdAt'] >= start - lookback_period) & (pendle_mentioned_df['createdAt'] < start)]

        all_pre_bull.append(compute_bullish_ratio(pre_tweets))
        all_in_bull.append(compute_bullish_ratio(surge_tweets))

    if all_pre_bull or all_in_bull:
        pre_bull_ratio = pd.concat(all_pre_bull, axis=1).sum(axis=1)
        in_bull_ratio = pd.concat(all_in_bull, axis=1).sum(axis=1)
        bullish_ratio_result = pd.DataFrame({
            "userName": pre_bull_ratio.index.union(in_bull_ratio.index),
            "presurge_bullish_tweets_ratio": pre_bull_ratio,
            "insurge_bullish_tweets_ratio": in_bull_ratio
        }).fillna(0)
    else:
        bullish_ratio_result = pd.DataFrame(columns=["userName", "presurge_bullish_tweets_ratio", "insurge_bullish_tweets_ratio"])

    for df in [tweet_frequency_result, buy_ratio_result, bullish_ratio_result]:
        if df.index.name == "userName":
            df.index.name = None
            df.reset_index(inplace=True)

    # Merge all into one final dict
    merged = pd.merge(tweet_frequency_result, buy_ratio_result, on="userName", how="outer")
    merged = pd.merge(merged, bullish_ratio_result, on="userName", how="outer").fillna(0)

    result = {}

    for _, row in merged.iterrows():
        score = (
            row["presurge_buy_tweets_ratio"] * 0.50 +
            row["insurge_tweets_frequency"] * 0.20 +
            row["presurge_bullish_tweets_ratio"] * 0.20 +
            row["insurge_bullish_tweets_ratio"] * 0.10
        )
        result[row["userName"]] = score

    for user in unique_users:
        if user not in result:
            result[user] = 0

    return result


def cal_surge_accuracy(df):
    """
    Determines the accuracy of a user's surge predictions based on success rates and lead times during surges.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the calculated surge accuracy score.
    """
    unique_usernames = get_unique_author_names(df)

    df = merge_tweet_analysis()
    success_rate_dict = success_rate_during_surge(df)
    lead_time_dict = lead_time_during_surge()

    surge_accuracy_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'success_rate_during_surge': [success_rate_dict.get(user, 0) for user in unique_usernames],
        'lead_time_during_surge': [lead_time_dict.get(user, 0) for user in unique_usernames]
    })

    surge_accuracy_df["surge_accuracy"] = (
        surge_accuracy_df["success_rate_during_surge"] * 0.60 +
        surge_accuracy_df["lead_time_during_surge"] * 0.40
    )

    surge_accuracy_df_dict = surge_accuracy_df.set_index("userName")["surge_accuracy"].to_dict()
    
    return surge_accuracy_df_dict


def cal_surge_vs_non_surge_consistency(df):
    """
    Measures consistency between surge and non-surge periods, considering tweet frequency and false positive rates
    during non-surge periods.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the surge vs non-surge consistency score.
    """
    unique_usernames = get_unique_author_names(df)

    df = merge_tweet_analysis()
    tweet_frequency_dict = tweet_frequency_ratio()
    false_positive_rate_dict = false_positive_rate_non_surge()

    surge_vs_non_surge_consistency_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'tweet_frequency_ratio': [tweet_frequency_dict.get(user, 0) for user in unique_usernames],
        'false_positive_rate_non_surge': [false_positive_rate_dict.get(user, 0) for user in unique_usernames]
    })

    surge_vs_non_surge_consistency_df["surge_vs_non_surge_consistency"] = (
        surge_vs_non_surge_consistency_df["tweet_frequency_ratio"] * 0.50 +
        surge_vs_non_surge_consistency_df["false_positive_rate_non_surge"] * 0.50
    )

    surge_vs_non_surge_consistency_df_dict = surge_vs_non_surge_consistency_df.set_index("userName")["surge_vs_non_surge_consistency"].to_dict()
    
    return surge_vs_non_surge_consistency_df_dict


def cal_hype_differential(df):
    """
    Compares hype scores between surge and non-surge periods, along with manipulative language usage during surges.
    
    Args:
    - df (DataFrame): DataFrame containing tweet data.
    
    Returns:
    - dict: Dictionary where the key is the username and the value is the hype differential score.
    """
    df = merge_tweet_analysis()
    unique_usernames = get_unique_author_names(df)

    hype_score_ratio_dict = hype_score_ratio()
    manipulative_language_surge_dict = manipulative_language_surge()

    hype_differential_df = pd.DataFrame({
        'userName': list(unique_usernames),
        'hype_score_surge_vs_non_surge': [hype_score_ratio_dict.get(user, 0) for user in unique_usernames],
        'manipulative_language_surge': [manipulative_language_surge_dict.get(user, 0) for user in unique_usernames]
    })

    hype_differential_df["surge_vs_non_surge_consistency"] = (
        hype_differential_df["hype_score_surge_vs_non_surge"] * 0.60 +
        hype_differential_df["manipulative_language_surge"] * 0.40
    )

    hype_differential_df_dict = hype_differential_df.set_index("userName")["surge_vs_non_surge_consistency"].to_dict()
    
    return hype_differential_df_dict