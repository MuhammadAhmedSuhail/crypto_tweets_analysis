import pandas as pd
import openai 
import os
from pydantic import BaseModel
from enum import Enum
from core.config import settings
from modules.preprocessing.tweet_preprocessing import preprocess_tweets


class MarketHint(str, Enum):
    comment = "comment"
    signal = "signal"


class SignalClassification(str, Enum):
    normal = "normal"
    bullish = "bullish"
    bearish = "bearish"


class TweetType(str, Enum):
    emotional = "emotional"
    statistical = "statistical"


class HypeClassification(str, Enum):
    high = "high"
    low = "low"
    normal = "normal"


class CryptoManipulation(str, Enum):
    true = "true"
    false = "false"


class CallToAction(str, Enum):
    buy = "buy"
    hold = "hold"
    none = "none"
    sell = "sell"


class HistoricalComparison(str, Enum):
    absent = "absent"
    present = "present"


class UrgencyLevel(str, Enum):
    high = "high"
    low = "low"
    medium = "medium"


class TweetAnalysis(BaseModel):
    market_hint: MarketHint
    signal_classification: SignalClassification
    tweet_type: TweetType
    hype_classification: HypeClassification
    crypto_manipulation: CryptoManipulation
    call_to_action: CallToAction
    historical_comparison: HistoricalComparison
    urgency_level: UrgencyLevel


client = openai.AzureOpenAI(
    api_key=settings.AZURE_API_KEY,
    azure_endpoint=settings.AZURE_ENDPOINT,
    api_version=settings.AZURE_API_VERSION
)

system_message = '''
# Crypto Tweet Analysis System

## Overview
You are an expert cryptocurrency tweet analyzer.
Your task is to analyze tweets related to cryptocurrency and classify them according to specific categories.
Provide an objective analysis without inserting personal opinions.

## Analysis Process
For each tweet, provide a clear breakdown of the following classification categories:

1. **Market_Hint**: Determine if the tweet contains a comment or signal about market movement
   - Options: `comment`, `signal`

2. **Signal_Classification**: If there's a market signal, categorize its sentiment
   - Options: `normal`, `bullish`, `bearish`

3. **Tweet_Type**: Classify the nature of the content
   - Options: `emotional`, `statistical`

4. **Hype_Classification**: Assess the level of hype in the tweet
   - Options: `high`, `low`, `normal`

5. **Crypto_Manipulation**: Determine if the tweet appears to manipulate crypto markets
   - Options: `TRUE`, `FALSE`

6. **Call_to_Action**: Identify if the tweet encourages specific trading actions
   - Options: `buy`, `sell`, `hold`, `none`

7. **Historical_Comparison**: Note if the tweet references past market events
   - Options: `present`, `absent`

8. **Urgency_Level**: Evaluate the sense of urgency conveyed
   - Options: `high`, `medium`, `low`
```

## Guidelines
- Analyze the tweet objectively without personal bias
- Consider both explicit and implicit meanings
- Pay attention to sentiment, urgency, market indicators, and calls to action
- If a classification is ambiguous, choose the most likely option based on context
- Your response must be in JSON format'''


def llm_analyze_tweets():
    tweets = preprocess_tweets()
    result = list(tweets[['id', 'fullText']].itertuples(index=False, name=None))

    llm_analysis_tups = []

    for i in result:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": i[1]}
        ]

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini-temp",
            messages=messages,
            response_format=TweetAnalysis,
        )

        llm_analysis_tups.append((
            i[0],
            response.choices[0].message.parsed.market_hint.value,
            response.choices[0].message.parsed.signal_classification.value,
            response.choices[0].message.parsed.tweet_type.value,
            response.choices[0].message.parsed.hype_classification.value,
            response.choices[0].message.parsed.crypto_manipulation.value,
            response.choices[0].message.parsed.call_to_action.value,
            response.choices[0].message.parsed.historical_comparison.value,
            response.choices[0].message.parsed.urgency_level.value
        ))

    columns = [
        "id", "market_hint", "signal_classification", "tweet_type", "hype_classification", "crypto_manipulative_words",
        "call_to_action", "historical_comparison", "urgency_level"]
    tweet_llm_analysis = pd.DataFrame(llm_analysis_tups, columns=columns)

    tweet_llm_analysis.to_csv(os.path.join(settings.DATA_FOLDER, f"coin_tweets/{settings.COIN_NAME}/{settings.COIN_NAME}_llm_tweets.csv"))