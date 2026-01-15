from core import config
from repo.runners.price_dataset_runner import coin_price_runner
from repo.runners.run_final_ranking import final_ranking_runner
from repo.runners.tweets_runner import tweets_runner
from repo.runners.llm_analyze_runner import llm_runner
# from core.config import settings
import logging

logger: logging.Logger = config.settings.LOGGING_SERVICE.logger


def complete_pipeline():
    logger.debug(f"Running Pipeline for {config.settings.COIN_NAME}")
    if config.settings.RUN_COIN_PRICE:
        coin_price_runner()
    if config.settings.RUN_TWEET_RUNNER:
        tweets_runner()
    if config.settings.RUN_LLM_ANALYSIS:
        llm_runner()
    if config.settings.RUN_FINAL_RANKING:
        return final_ranking_runner()
        

if __name__ == "__main__":
    complete_pipeline()