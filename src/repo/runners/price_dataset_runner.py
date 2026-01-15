from repo.runners.price_tweets_runner import get_coin_dataset
from core.config import settings
import logging
from core.config import settings
logger: logging.Logger = settings.LOGGING_SERVICE.logger


def coin_price_runner():

    logger.debug("Getting Coin Price Dataset")
    get_coin_dataset(settings.COIN_NAME, settings.START_TIME_STRING, settings.END_TIME_STRING)