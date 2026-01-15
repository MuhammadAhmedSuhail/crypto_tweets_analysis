scoring_config = {
    "main_categories": {
        "historical_prediction_accuracy": {
            "weight": 0.35,
            "subcategories": {
                "prediction_success_rate": {
                    "weight": 0.70,
                    "features": {
                        "successful_10x_predictions_ratio": 0.70,
                        "consecutive_successful_predictions": 0.30
                    }
                },
                "false_prediction_rate": {
                    "weight": 0.30,
                    "features": {
                        "incorrect_buy_signals": 1.0
                    }
                }
            }
        },
        "signal_quality": {
            "weight": 0.20,
            "subcategories": {
                "data_driven_content": {
                    "weight": 0.30,
                    "features": {
                        "statistical_vs_emotional_ratio": 0.60,
                        "historical_comparison_presence_ratio": 0.40
                    }
                },
                "signal_clarity": {
                    "weight": 0.35,
                    "features": {
                        "market_hint_classification_ratio": 0.30,
                        "signal_classification_ratio": 0.40,
                        "call_to_action": 0.30
                    }
                },
                "manipulation_resistance": {
                    "weight": 0.30,
                    "features": {
                        "absence_of_manipulative_language_ratio": 0.50,
                        "hype_classification": {
                            "low": 1.0,
                            "normal": 0.5,
                            "high": 0.0
                        }
                    }
                },
                "urgency_sanity_check": {
                    "weight": 0.05,
                    "features": {
                        "urgency_level": {
                            "low": 1.0,
                            "medium": 0.3,
                            "high": 0.0
                        }
                    }
                }
            }
        },
        "timing_and_relevance": {
            "weight": 0.125,
            "subcategories": {
                "early_detection": {
                    "weight": 0.60,
                    "features": {
                        "time_to_5x_movement": 0.50,
                        "lead_time_before_price_surges": 0.50
                    }
                },
                "consistency": {
                    "weight": 0.40,
                    "features": {
                        "pre_surge_buy_signals_ratio": 0.50,
                        "tweet_frequency_during_surge_periods": 0.20,
                        "bullish_tweets_ratio_before_surge_periods": 0.20,
                        "bullish_tweets_ratio_during_surge_periods": 0.10
                    }
                }
            }
        },
        "account_credibility": {
            "weight": 0.20,
            "subcategories": {
                "verification_and_trust": {
                    "weight": 0.40,
                    "features": {
                        "blue_verification_badge": 0.20,
                        "account_age": {
                            "min(1, account_age_days / 1096)": 0.20
                        },
                        "profile_completeness": 0.10,
                        "media_status_ratio": 0.10,
                        "tweets_frequency": 0.10,
                        "content_originality_ratio": 0.20,
                        "human_source_devices_ratio": 0.10
                    }
                },
                "follower_quality": {
                    "weight": 0.60,
                    "features": {
                        "follower_to_following_ratio": 0.35,
                        "engagement_ratio": 0.50,
                        "avg_reach": 0.15
                    }
                }
            }
        },
        "surge_performance_differential": {
            "weight": 0.125,
            "subcategories": {
                "surge_accuracy": {
                    "weight": 0.40,
                    "features": {
                        "success_rate_during_surge": 0.60,
                        "lead_time_during_surge": 0.40
                    }
                },
                "surge_vs_non_surge_consistency": {
                    "weight": 0.30,
                    "features": {
                        "tweet_frequency_ratio": 0.50,
                        "false_positive_rate_non_surge": 0.50
                    }
                },
                "hype_differential": {
                    "weight": 0.30,
                    "features": {
                        "hype_score_surge_vs_non_surge": 0.60,
                        "manipulative_language_surge": 0.40
                    }
                }
            }
        }
    },
    "possible_penalty_mechanisms": {
        "hype_penalty": {
            "condition": "hype_score_during_surge < 20% above non-surge baseline",
            "penalty": 0.20
        },
        "engagement_floor": {
            "condition": "engagement_rate < 0.05%",
            "action": "exclude"
        },
        "surge_only_filter": {
            "condition": "tweets_during_surge > 80% of total_tweets",
            "penalty": 0.15
        },
        "is_single_day_user": {
            "penalty": None  # Fill this in later
        }
    },
    "calculation_methods": {
        "surge_definition": "price_increase > 5x OR volume_increase > 10x OR trade_count_increase > 10x over 7-day rolling window",
        "success_rate_during_surge": "correct_signals_before_surge / total_signals_during_surge",
        "hype_score": "1 - (high_hype_tweets / total_tweets_per_period)",
        "consistency_score": "1 - (std_dev_of_post_intervals / max_interval)",
        "follower_quality": "(followers_count / max(following_count, 1)) / highest_ratio_in_dataset",
        "final_score": '''(0.35*HistoricalPredictionAccuracy + 0.20*SignalQuality + 0.125*TimingAndRelevance + 0.20*AccountCredibility +
        0.125*SurgePerformanceDifferential) * (1 - HypePenalty - SurgeOnlyPenalty)'''
    }
}
