from modules.tweet_analysis.llm_analysis import llm_analyze_tweets
import logging
from core.config import settings
logger: logging.Logger = settings.LOGGING_SERVICE.logger


def llm_runner():

    llm_analyze_tweets()
    logger.debug("Tweets Analyzed!")