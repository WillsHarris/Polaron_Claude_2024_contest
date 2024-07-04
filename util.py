import pandas as pd

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