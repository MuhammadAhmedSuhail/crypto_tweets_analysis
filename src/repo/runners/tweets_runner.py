from repo.runners.price_tweets_runner import get_tweets
from core.config import settings
import logging
from core.config import settings
logger: logging.Logger = settings.LOGGING_SERVICE.logger


def tweets_runner():
    
    get_tweets(settings.COIN_NAME, settings.START_TIME_STRING, settings.END_TIME_STRING)
    logger.debug("Tweets Scraped!")