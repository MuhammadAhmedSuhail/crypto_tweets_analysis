import requests
import os
import re
from requests.exceptions import RequestException, JSONDecodeError
import logging
from core.config import settings
logger: logging.Logger = settings.LOGGING_SERVICE.logger

rapid_api_key = settings.RAPID_API_KEY

rapid_api_keys = rapid_api_key.split(",")


def fetch_coin_details(coin_name, coin_dataset):
    coin_name = coin_name.lower()

    # Lower both sides for comparison
    result = coin_dataset[coin_dataset['name'].str.lower().str.contains(coin_name, na=False)]

    if not result.empty:
        return result.iloc[0]['symbol'], result.iloc[0]['screen_name']
    else:
        return "Coin name not found"


url = "https://twitter241.p.rapidapi.com/search-v2"


def extract_tweets(coin, symbol, handle, start_date, end_date):
    tweet_data = []
    queries = [
        f'"{coin}" since:{start_date} until:{end_date}',
        f'"{symbol}" since:{start_date} until:{end_date}',
        f'"${symbol}" since:{start_date} until:{end_date}',
        f'"#{coin}" since:{start_date} until:{end_date}',
        f'"#{symbol}" since:{start_date} until:{end_date}',
        f'"@{handle}" since:{start_date} until:{end_date}'
    ]

    key_index = 0  # Start with the first API key

    for query in queries:
        bottom_cursor = None
        query_data = []
        while True:
            current_key = rapid_api_keys[key_index % len(rapid_api_keys)]
            key_index += 1

            headers = {
                "x-rapidapi-key": current_key,
                "x-rapidapi-host": "twitter241.p.rapidapi.com"
            }

            params = {
                "type": "Latest",
                "count": "5",
                "query": query
            }
            if bottom_cursor:
                params["cursor"] = bottom_cursor
            try:
                response = requests.get(url, headers=headers, params=params)
                data = response.json()
            except (RequestException, JSONDecodeError) as e:
                logger.exception(f"Request failed: {e}")
                continue

            try:
                entries = data["result"]["timeline"]["instructions"][0]["entries"]
            except (KeyError, IndexError):
                bottom_cursor = data.get("cursor", {}).get("bottom")
                continue

            query_data.extend(entries)

            if len(query_data) >= 1:
                tweet_data.append(query_data)
                # print("Length of tweets scraped till now: ",len(tweet_data))
                break

            bottom_cursor = data.get("cursor", {}).get("bottom")
            # print("Length of tweets scraped for query: ",len(query_data))

    return tweet_data


def parse_tweet_data(extracted_tweets):
    parsed_tweets = []
    for tweet in extracted_tweets:
        try:
            result = tweet["content"]["itemContent"]["tweet_results"]["result"]
            legacy = result.get("legacy", {})
            user = result["core"]["user_results"]["result"]
            user_legacy = user.get("legacy", {})

            tweet_data = {
                "type": result.get("__typename"),
                "id": result.get("rest_id"),
                "username": user_legacy.get("screen_name"),
                "tweet_url": f"https://x.com/{user_legacy.get('screen_name')}/status/{legacy.get('conversation_id_str')}",
                "twitter_url": f"https://twitter.com/{user_legacy.get('screen_name')}/status/{legacy.get('conversation_id_str')}",
                # "display_text_range": legacy.get("display_text_range"),
                "text": legacy.get("full_text", "")[:legacy.get("display_text_range", [0, 0])[-1]],
                "fullText": legacy.get("full_text", ""),
                "source": re.sub(r'<[^>]+>', '', result.get("source", "")),
                "retweet_count": legacy.get("retweet_count"),
                "like_count": legacy.get("favorite_count"),
                "reply_count": legacy.get("reply_count"),
                "favorite_count": legacy.get("favorite_count"),
                "quote_count": legacy.get("quote_count"),
                "view_count": result.get("views", {}).get("count"),
                "createdAt": legacy.get("created_at"),
                "lang": legacy.get("lang"),
                "bookmark_count": legacy.get("bookmark_count"),
                "conversation_id": legacy.get("conversation_id_str"),
                "is_reply": "quoted_status_result" in result,
                "is_quote": legacy.get("is_quote_status"),
                "is_retweet": legacy.get("retweeted"),
                "quoted_status": result.get("quoted_status_result", {}),
                "quoted_id": result.get("quoted_status_result", {}).get("result", {}).get("quotedRefResult", {}).get("result", {}).get("rest_id"),
                "quoted_username": (
                    result.get("quoted_status_result", {})
                    .get("result", {})
                    .get("core", {})
                    .get("user_results", {})
                    .get("result", {})
                    .get("legacy", {})
                    .get("screen_name")
                ),
                "is_pinned": "pinned_tweet_ids_str" in legacy,
                "author": {
                    "type": user.get("__typename"),
                    "is_blue_verified": user.get("is_blue_verified"),
                    "is_verified": user_legacy.get("verified"),
                    "user_id": user.get("rest_id"),
                    "affiliates_highlighted_label": user.get("affiliates_highlighted_label"),
                    "userName": user_legacy.get("screen_name"),
                    "url": f"https://x.com/{user_legacy.get('screen_name')}",
                    "twitter_url": f"https://twitter.com/{user_legacy.get('screen_name')}",
                    "name": user_legacy.get("name"),
                    "profile_picture": user_legacy.get("profile_image_url_https"),
                    "cover_picture": user_legacy.get("profile_banner_url"),
                    "description": user_legacy.get("description"),
                    "location": user_legacy.get("location"),
                    "followers": user_legacy.get("followers_count"),
                    "following": user_legacy.get("friends_count"),
                    "can_dm": user_legacy.get("can_dm"),
                    "can_media_tag": user_legacy.get("can_media_tag"),
                    "created_at": user_legacy.get("created_at"),
                    "entities": user_legacy.get("entities"),
                    "fast_followers_count": user_legacy.get("fast_followers_count"),
                    "favourites_count": user_legacy.get("favourites_count"),
                    "has_custom_timelines": user_legacy.get("has_custom_timelines"),
                    "is_translator": user_legacy.get("is_translator"),
                    "media_count": user_legacy.get("media_count"),
                    "statuses_count": user_legacy.get("statuses_count"),
                    "withheld_in_countries": user_legacy.get("withheld_in_countries"),
                    "possibly_sensitive": user_legacy.get("possibly_sensitive"),
                    "pinned_tweet_ids": user_legacy.get("pinned_tweet_ids_str"),
                    "createdAt": user_legacy.get("created_at")
                },
                "media_urls": legacy.get("entities", {}).get("urls", []),
                "is_conversation_controlled": result.get("edit_control", {}).get("is_edit_eligible")
            }

            parsed_tweets.append(tweet_data)
        except KeyError as k:
            continue
        except Exception as e:
            logger.exception(f"Request failed: {e}")
            continue

    return parsed_tweets