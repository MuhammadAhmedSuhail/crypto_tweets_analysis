from modules.preprocessing.tweet_preprocessing import merge_tweet_analysis
from modules.account_ranking.ranking import final_ranking


def final_ranking_runner():
    tweets_analysis = merge_tweet_analysis()
    return final_ranking(tweets_analysis)