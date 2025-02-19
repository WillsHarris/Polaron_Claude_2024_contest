
import os

import sys
sys.path.append('../')
import json

import time
import threading
import argparse

import pandas as pd
import numpy as np
import openai
from util import *
from llm import *
from filter_functions import s_filter, b_filter, q_filter
from stories import *
import pytz
import traceback

# Get the directory that this script is in
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "config.json"), 'r') as file:
    config = json.load(file)

storage_path = config['storage_path']
debug = config['debug']
smart_llm = config['smart_llm']
dumb_llm = config['dumb_llm']

s_low = config['s_low']
s_high = config['s_high']

max_score = config['max_score']

assets_df = pd.read_csv(os.path.join(storage_path, "assets.csv"))
articles_df = pd.read_csv(os.path.join(storage_path, "articles.csv"))
stories_df = pd.read_csv(os.path.join(storage_path, "stories.csv"))
full_articles_df = pd.read_csv("/home/broton/storage/articles.csv")
len_articles = len(full_articles_df)

if __name__ == "__main__":
    while True:
        if debug:
            # loop through test articles:
            print("Debug mode on.")
            articles_raw = get_debug_articles(articles_df)
        else:
            articles_raw = None # get articles_raw from data source of news articles

        for article_idx in articles_raw.index:
            # q-filter:
            articles_df, q_bool = q_filter(article_idx, assets_df, articles_df, debug)
            if not q_bool and not debug: 
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                continue # reset loop

            #summarize article:
            article_sum, on_topic_bool, articles_df = llm_get_summary(article_idx, dumb_llm, assets_df, articles_df)
            if not on_topic_bool and not debug:
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                continue # reset loop

            # s-filter:
            s_bool, s_type, articles_df = s_filter(article_idx, s_low, s_high, storage_path, articles_df, debug)
            if not s_bool and not debug:
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                continue # reset loop

            # b-filter:
            b_bool, b_str, articles_df = b_filter(article_idx, dumb_llm, articles_df)
            if not b_bool and not debug:
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                continue # reset loop
            # news type filter:
            news_type, articles_df = llm_get_summary_metadata(article_idx, dumb_llm, assets_df, articles_df)
            if news_type in ['earnings', 'financial', 'macroeconomic', \
                                'market analysis', 'opinion', 'management' ,'other'] and not debug:
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                print("News type is out of scope.")
                continue # reset loop

            # get the article score:
            article_score, articles_df = llm_score_summary(article_idx,
                                                        max_score, 
                                                        smart_llm, 
                                                        assets_df,
                                                        articles_df)

            # do nothing if article score is 0:
            if article_score == 0:
                print("Article score is 0.")
            # add story to stories_df:
            stories_df = add_story(article_idx, smart_llm, stories_df, articles_df, assets_df, debug)
            if not debug:
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                stories_df.to_csv(os.path.join(storage_path, 'stories.csv'), index=False)
        if debug:
            break