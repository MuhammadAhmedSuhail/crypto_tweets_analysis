# Crypto Tweets Pipeline

## Overview

The Cryptocurrency Twitter Analysis project leverages social media data to predict cryptocurrency market behavior and rank influential users. Using data processing techniques and machine learning algorithms, this project extracts insights from Twitter interactions related to cryptocurrencies, focusing on:

- Historical performance analysis
- Tweet classification and sentiment analysis
- User credibility ranking
- API development for seamless data access

## Features

- **Tweet Analysis**: Extracts cryptocurrency mentions from tweets using Aho-Corasick algorithm
- **Market Correlation**: Correlates tweets with price movements to identify prediction accuracy
- **User Ranking**: Evaluates Twitter users based on credibility, signal quality, and prediction accuracy
- **LLM Integration**: Uses OpenAI's GPT models to analyze tweet sentiment and content
- **API Access**: FastAPI endpoints for requesting analysis and retrieving results

## Project Structure

```
./
  └── requirements.txt
notebooks/                           # Analysis notebooks
src/                                 # Source code
  ├── app.py                         # FastAPI application entry point
  ├── controller.py                  # Pipeline orchestration
  └── mongoclient.py                 # MongoDB connection
src/api/                             # API endpoints
src/core/                            # Configuration
src/modules/                         # Core functionality
  ├── account_ranking/               # User ranking algorithms
  ├── coin_analysis/                 # Cryptocurrency data analysis
  ├── feature_engineering/           # Feature calculation
  ├── preprocessing/                 # Data preprocessing
  └── tweet_analysis/                # Tweet content analysis
src/repo/                            # Data repositories
  └── runners/                       # Pipeline components
src/schemas/                         # Data validation schemas
src/services/                        # Support services
```

## Key Components

### Data Collection
- Extracts tweets related to cryptocurrencies using Twitter API
- Fetches cryptocurrency price data from CoinDesk API
- Stores data in MongoDB for analysis

### Analysis Pipeline
1. **Preprocessing**: Normalizes tweet data and extracts relevant features
2. **LLM Analysis**: Classifies tweets by sentiment, urgency, and signal type
3. **Feature Engineering**: Calculates metrics for user credibility and signal quality
4. **Ranking**: Combines scores to create a final ranking of Twitter accounts

### Scoring System
The ranking algorithm evaluates users on four main categories:

1. **Account Credibility** (25%)
   - Verification status
   - Follower quality
   - Account age and profile completeness

2. **Signal Quality** (30%)
   - Data-driven content ratio
   - Signal clarity
   - Manipulation resistance
   - Urgency appropriateness

3. **Historical Prediction Accuracy** (25%)
   - Success rate for bullish/bearish predictions
   - False prediction rate
   - Prediction consistency

4. **Timing and Relevance** (20%)
   - Tweet timing relative to price movements
   - Consistency during market surge events
   - Engagement differential during price movements

## API Usage

### Request Analysis
```
POST /coin-analysis
{
  "coin_name": "bitcoin",
  "start_time_string": "2023-01-01 00:00:00",
  "end_time_string": "2023-01-31 23:59:59"
}
```

### Check Request Status
```
GET /get_request_status?request_id=<id>
```

### Retrieve Results
```
GET /get_request_results?request_id=<id>
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/crypto-cartel-pipeline.git
cd crypto-cartel-pipeline
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
# API Keys
coin_desk_api_key=""
rapid_api_key=""
AZURE_API_KEY=""
AZURE_ENDPOINT=""
AZURE_API_VERSION=""

# Analysis Configuration
coin_name=""
start_time_string=""
end_time_string=""

# Pipeline Control
RUN_COIN_PRICE=""
RUN_TWEET_RUNNER=""
RUN_LLM_ANALYSIS=""
RUN_FINAL_RANKING=""

# Database Configuration
DB_NAME=""
REQUEST_COLLECTION_NAME=""
RANKING_COLLECTION_NAME=""
DB_USERNAME=""
DB_PASSWORD=""
```

4. Run the API server
```bash
cd src
uvicorn app:app --reload
```

## Notebooks

The project includes several Jupyter notebooks for exploration and development:

- `Accounts_Historical_Prediction.ipynb`: Analyzes historical prediction accuracy
- `Classification_Tweets.ipynb`: Classifies tweets by content and signal type
- `Crypto_Cartel_EDA.ipynb`: Exploratory data analysis of Twitter accounts
- `llm_analysis.ipynb`: Tweet analysis using OpenAI GPT models
- `price.ipynb`: Cryptocurrency price data processing
- `Time_Scoring.ipynb`: Time-based analysis of tweet impact
- `twitter.ipynb`: Twitter data extraction and processing
