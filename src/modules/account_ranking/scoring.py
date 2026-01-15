import pandas as pd
from modules.account_ranking.internal_ranking import cal_consistency
from modules.utils import get_unique_author_names
from modules.weightages import scoring_config
from modules.account_ranking.internal_ranking import cal_data_driven_content, cal_early_detection, cal_false_prediction_rate
from modules.account_ranking.internal_ranking import cal_hype_differential, cal_manipulation_resistance, cal_prediction_success_rate
from modules.account_ranking.internal_ranking import cal_signal_clarity, cal_surge_accuracy, cal_surge_vs_non_surge_consistency
from modules.account_ranking.internal_ranking import cal_urgency_sanity_check, calculate_verification_trust, calculate_follower_quality


def cal_account_credibility(df):
    """
    Calculates an account credibility score for each user based on verification trust and follower quality.
    Returns a dictionary mapping user names to their credibility scores.
    """
    verification_dict = calculate_verification_trust(df)
    follower_dict = calculate_follower_quality(df)
    unique_users = get_unique_author_names(df)

    account_df = pd.DataFrame({
        'userName': list(unique_users),
        'verification_score': [verification_dict.get(user, 0) for user in unique_users],
        'follower_quality_score': [follower_dict.get(user, 0) for user in unique_users]
    })

    account_df["account_credibility"] = (
        account_df["verification_score"]
        * scoring_config["main_categories"]["account_credibility"]["subcategories"]["verification_and_trust"]["weight"]
        + account_df["follower_quality_score"]
        * scoring_config["main_categories"]["account_credibility"]["subcategories"]["follower_quality"]["weight"]
    )

    account_score_dict = account_df.set_index("userName")["follower_quality_score"].to_dict()

    return account_score_dict


def cal_signal_quality(df):
    """
    Computes a signal quality score per user by evaluating data-driven content, signal clarity,
    resistance to manipulation, and urgency sanity. Returns a dictionary of scores by user.
    """
    data_driven_content_dict = cal_data_driven_content(df)
    signal_clarity_dict = cal_signal_clarity(df)
    manipulation_resistance_dict = cal_manipulation_resistance(df)
    urgency_sanity_check_dict = cal_urgency_sanity_check(df)

    unique_users = get_unique_author_names(df)

    signal_df = pd.DataFrame({
        'userName': list(unique_users),
        'data_driven_content': [data_driven_content_dict.get(user, 0) for user in unique_users],
        'signal_clarity': [signal_clarity_dict.get(user, 0) for user in unique_users],
        'manipulation_resistance': [manipulation_resistance_dict.get(user, 0) for user in unique_users],
        'urgency_sanity_check': [urgency_sanity_check_dict.get(user, 0) for user in unique_users],
    })

    signal_df["signal_quality"] = ( 
        signal_df["data_driven_content"] * scoring_config["main_categories"]["signal_quality"]["subcategories"]["data_driven_content"]["weight"] + 
        signal_df["signal_clarity"] * scoring_config["main_categories"]["signal_quality"]["subcategories"]["signal_clarity"]["weight"] + 
        signal_df["manipulation_resistance"]*scoring_config["main_categories"]["signal_quality"]["subcategories"]["manipulation_resistance"]["weight"]
        +
        signal_df["urgency_sanity_check"] * scoring_config["main_categories"]["signal_quality"]["subcategories"]["urgency_sanity_check"]["weight"]
    )

    signal_df_dict = signal_df.set_index("userName")["signal_quality"].to_dict()

    return signal_df_dict


def cal_historical_prediction_accuracy(df):
    """
    Calculates historical prediction accuracy for each user using their prediction success rate
    and false prediction rate. Returns a dictionary mapping users to accuracy scores.
    """
    unique_users = get_unique_author_names(df)

    prediction_success_df_dict = cal_prediction_success_rate(df)
    incorrect_buy_signals_dict = cal_false_prediction_rate(df)

    historical_prediction_accuracy_df = pd.DataFrame({
        'userName': list(unique_users),
        'prediction_success_rate': [prediction_success_df_dict.get(user, 0) for user in unique_users],
        'false_prediction_rate': [incorrect_buy_signals_dict.get(user, 0) for user in unique_users],
    })

    historical_prediction_accuracy_df["historical_prediction_accuracy"] = (
        historical_prediction_accuracy_df["prediction_success_rate"]
        * scoring_config["main_categories"]["historical_prediction_accuracy"]["subcategories"]["prediction_success_rate"]["weight"]
        + historical_prediction_accuracy_df["false_prediction_rate"]
        * scoring_config["main_categories"]["historical_prediction_accuracy"]["subcategories"]["false_prediction_rate"]["weight"]
    )

    historical_prediction_accuracy_dict = historical_prediction_accuracy_df.set_index("userName")["historical_prediction_accuracy"].to_dict()

    return historical_prediction_accuracy_dict


def cal_timing_and_relevance(df):
    """
    Assesses timing and relevance of users' signals based on early detection and consistency.
    Scores are normalized and returned as a dictionary per user.
    """
    unique_users = get_unique_author_names(df)
    early_dict = cal_early_detection()
    consistency_dict = cal_consistency()

    historical_prediction_accuracy_df = pd.DataFrame({
        'userName': list(unique_users),
        'early_detection': [early_dict.get(user, 0) for user in unique_users],
        'consistency': [consistency_dict.get(user, 0) for user in unique_users],
    })

    historical_prediction_accuracy_df["timing_and_relevance"] = ( 
        historical_prediction_accuracy_df["early_detection"] * 
        scoring_config["main_categories"]["timing_and_relevance"]["subcategories"]["early_detection"]["weight"]
        + 
        historical_prediction_accuracy_df["consistency"] * 
        scoring_config["main_categories"]["timing_and_relevance"]["subcategories"]["consistency"]["weight"]
    )

    # Get the timing_and_relevance column
    timing_and_relevance_column = historical_prediction_accuracy_df["timing_and_relevance"]

    # Calculate min and max values for normalization
    min_value = timing_and_relevance_column.min()
    max_value = timing_and_relevance_column.max()

    # Normalize the 'timing_and_relevance' column
    historical_prediction_accuracy_df['normalized_timing_and_relevance'] = (timing_and_relevance_column - min_value) / (max_value - min_value)

    # Convert to dictionary
    historical_prediction_accuracy_dict = historical_prediction_accuracy_df.set_index("userName")["normalized_timing_and_relevance"].to_dict()

    return historical_prediction_accuracy_dict


def surge_performance_differential(df):
    """
    Measures users' performance during market surges by combining surge accuracy,
    consistency between surge and non-surge periods, and hype differential.
    Returns a dictionary of differential scores by user.
    """
    unique_users = get_unique_author_names(df)
    surge_dict = cal_surge_accuracy(df)
    surge_vs_non_surge_dict = cal_surge_vs_non_surge_consistency(df)
    hype_diff_dict = cal_hype_differential(df)

    surge_performance_differential_df = pd.DataFrame({
        'userName': list(unique_users),
        'surge_accuracy': [surge_dict.get(user, 0) for user in unique_users],
        'surge_vs_non_surge_consistency': [surge_vs_non_surge_dict.get(user, 0) for user in unique_users],
        'hype_differential': [hype_diff_dict.get(user, 0) for user in unique_users],
    })

    surge_performance_differential_df["surge_performance_differential"] = (
        surge_performance_differential_df["surge_accuracy"]
        * scoring_config["main_categories"]["surge_performance_differential"]["subcategories"]["surge_accuracy"]["weight"]
        + surge_performance_differential_df["surge_vs_non_surge_consistency"]
        * scoring_config["main_categories"]["surge_performance_differential"]["subcategories"]["surge_vs_non_surge_consistency"]["weight"]
        + surge_performance_differential_df["hype_differential"]
        * scoring_config["main_categories"]["surge_performance_differential"]["subcategories"]["hype_differential"]["weight"]
    )

    # Convert to dictionary
    surge_performance_differential_df_dict = surge_performance_differential_df.set_index("userName")["surge_performance_differential"].to_dict()

    return surge_performance_differential_df_dict