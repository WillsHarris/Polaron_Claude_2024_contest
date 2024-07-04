
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
from plotNpost import post_story, post_update
from filter_functions import s_filter, b_filter, q_filter
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

assets_df = pd.read_csv(os.path.join(storage_path, "assets.csv"))
articles_df = pd.read_csv(os.path.join(storage_path, "articles.csv"))

if __name__ == "__main__":
    # loop through test articles:
    for article_idx in range(len(articles_df)):

        # q-filter:
        articles_df, q_bool = q_filter(article_idx, assets_df, articles_df, debug)

        #summarize article:
        article_sum, on_topic_bool, articles_df = llm_get_summary(article_idx, dumb_llm, assets_df, articles_df)

        # s-filter:
        s_bool, s_type, articles_df = s_filter(article_idx, s_low, s_high, storage_path, articles_df, debug)
        if not s_bool:
            print("s_filter failed")
            continue # reset loop
        if s_type == "update": # TEMP -- kill updates after extracting update data
            posts_df, articles_df = post_update(article_idx, articles_df, posts_df, dev_mode, debug)
            print("update post")
            continue # reset loop

        # b-filter:
        b_bool, b_str, llm_cost, articles_df = b_filter(article_idx, dumb_llm, articles_df)
        if not b_bool:
            print("b_filter failed")
            continue # reset loop
        # news type filter:
        news_type, articles_df = llm_get_summary_metadata(article_idx, dumb_llm, assets_df, articles_df)
        if news_type in ['earnings', 'financial', 'macroeconomic', \
                            'market analysis', 'opinion', 'management' ,'other']:
            print("News type not relevant.")
            continue # reset loop
        elif news_type == 'product': 
            articles_df = get_product_aux(article_idx, dumb_llm, articles_df, storage_path)
        elif news_type == 'acquisition (of a company)':
            articles_df = get_acquisition_aux(article_idx, dumb_llm, articles_df, storage_path)
        elif news_type == 'regulatory':
            articles_df = get_regulatory_aux(article_idx, dumb_llm, articles_df, storage_path)

        # get the article score:
        article_score, articles_df = llm_score_summary(article_idx,
                                                    max_score, 
                                                    smart_llm, 
                                                    assets_df,
                                                    articles_df)

        # do nothing if article score is 0:
        if article_score == 0:
            print("Article score is 0.")
            continue

        # social media post
        #posts_df, articles_df = post_story(article_idx, articles_df, posts_df, dev_mode, debug)