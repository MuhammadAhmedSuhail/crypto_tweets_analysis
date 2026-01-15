from modules.preprocessing.tweet_preprocessing import fetch_coin_dataset, merge_tweet_analysis
import logging
from core.config import settings
logger: logging.Logger = settings.LOGGING_SERVICE.logger


def preprocessing_runner():

    logger.debug("Preprocessing Started!")
    merge_tweet_analysis()
    fetch_coin_dataset()
    logger.debug("Preprocessing Finished!")