import pandas as pd
import os
import numpy as np
import json

# load vars from config and debug files
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "config.json"), 'r') as file:
    config = json.load(file)

with open(os.path.join(dir_path, "debug.json"), 'r') as file:
    debug_params = json.load(file)

def log_outcome(article_idx, articles_df, outcome):
    """
    Logs the outcome of a trade for a given article to the articles df.

    Also, prints the article index to the terminal if an error occurs.
    """
    if outcome is None:
        return articles_df
    elif "error" in outcome:
        print(outcome, "occurred for article idx:", article_idx)

    if type(articles_df.loc[article_idx, "outcome"]) == str:
        articles_df.loc[article_idx, "outcome"] += ","+outcome
    else:
        articles_df.loc[article_idx, "outcome"] = outcome

    return articles_df


def get_debug_articles(articles_df, debug_idx=None):
    """
    Function for accessing articles in debug mode using the debug.json file.
    """
    debug_mode = debug_params["debug_mode"] # "scored" or "all"
    num_articles = debug_params["num_articles"]

    if debug_idx is not None:
        print(f"Running in debug mode, using article idx={debug_idx} for testing.")
        if debug_idx == -1:
            articles_raw = articles_df.iloc[-1:]
        else:
            articles_raw = articles_df.iloc[debug_idx:debug_idx+1]
    else:
        print(f"Running in debug mode ({debug_mode}), using the last {num_articles} articles for testing.")
        if debug_mode == "scored":
            scored_articles_df = articles_df[articles_df['article_score'].notnull()]
            articles_raw = scored_articles_df[-num_articles:]
        else:
            articles_raw = articles_df[-num_articles:]
    return articles_raw