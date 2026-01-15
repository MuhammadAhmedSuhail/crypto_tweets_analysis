import os
import pandas as pd

from core.config import settings
from modules.account_ranking.scoring import cal_historical_prediction_accuracy, cal_signal_quality
from modules.account_ranking.scoring import cal_timing_and_relevance, surge_performance_differential, cal_account_credibility
from modules.utils import get_unique_author_names
from modules.weightages import scoring_config
import logging
from core.config import settings
logger: logging.Logger = settings.LOGGING_SERVICE.logger


def final_ranking(df):
    logger.debug("Ranking Account Credibility")
    acc_cred_dict = cal_account_credibility(df)
    logger.debug("Ranking Signal Quality")
    signal_quality = cal_signal_quality(df)
    logger.debug("Ranking Historical Prediction Accuracy")
    hist_pred_acc_dict = cal_historical_prediction_accuracy(df)
    logger.debug("Ranking Timing and Relevance")
    timing_relevance_dict = cal_timing_and_relevance(df)
    logger.debug("Ranking Surge Performance Differential")
    surge_perf_dict = surge_performance_differential(df)

    unique_users = get_unique_author_names(df)

    final_ranking_df = pd.DataFrame({
        'userName': list(unique_users),
        'account_credibility': [acc_cred_dict.get(user, 0) for user in unique_users],
        'surge_performance_differential': [surge_perf_dict.get(user, 0) for user in unique_users],
        'historical_prediction_accuracy': [hist_pred_acc_dict.get(user, 0) for user in unique_users],
        'signal_quality': [signal_quality.get(user, 0) for user in unique_users],
        'timing_and_relevance': [timing_relevance_dict.get(user, 0) for user in unique_users],
    })

    final_ranking_df.fillna(0, inplace=True)

    final_ranking_df["final_ranking"] = ( 
        final_ranking_df["account_credibility"] * scoring_config["main_categories"]["account_credibility"]["weight"] + 
        final_ranking_df["surge_performance_differential"] * scoring_config["main_categories"]["surge_performance_differential"]["weight"] +
        final_ranking_df["historical_prediction_accuracy"] * scoring_config["main_categories"]["historical_prediction_accuracy"]["weight"] +
        final_ranking_df["signal_quality"] * scoring_config["main_categories"]["signal_quality"]["weight"] +
        final_ranking_df["timing_and_relevance"] * scoring_config["main_categories"]["timing_and_relevance"]["weight"]
    )

    logger.debug("Final Ranking Completed!")

    final_ranking_df = final_ranking_df.fillna(0)
    final_ranking_df = final_ranking_df.sort_values(by="final_ranking", ascending=False)

    final_ranking_df.to_csv(
        os.path.join(settings.DATA_FOLDER, f"output/{settings.COIN_NAME}_ranking.csv"),
        index=False
    )

    logger.debug("Exported Ranking as CSV")

    return final_ranking_df