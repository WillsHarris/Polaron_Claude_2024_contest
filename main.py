
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

max_score = config['max_score']

assets_df = pd.read_csv(os.path.join(storage_path, "assets.csv"))
articles_df = pd.read_csv(os.path.join(storage_path, "articles.csv"))
full_articles_df = pd.read_csv("/home/broton/storage/articles.csv")
len_articles = len(full_articles_df)

if __name__ == "__main__":
    while True:
        full_articles_df = pd.read_csv("/home/broton/storage/articles.csv")
        # loop through test articles:
        if debug:
            print("Debug mode on.")
            articles_raw = get_debug_articles(articles_df)
        else:
            time.sleep(60) # check every 60 seconds
            # check for new articles
            if len(full_articles_df) > len_articles:
                new_articles = len(full_articles_df) - len_articles
                print(new_articles, "New articles detected.")
                new_articles_df = full_articles_df.tail(new_articles)
                new_articles_df.loc[:, ['qfactor', 'qfactor_bool', 'summary', 'sfactor', 'sfactor_bool', 'duplicate_articles', 'bfactor_str', 'article_score']] = None
                articles_df = pd.concat([articles_df, new_articles_df], ignore_index=True)
                articles_raw = articles_df.tail(new_articles)
                len_articles = len(full_articles_df) # update len_articles
            else:
                continue # reset loop


        for article_idx in articles_raw.index:
            # q-filter:
            articles_df, q_bool = q_filter(article_idx, assets_df, articles_df, debug)
            if not q_bool: 
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                continue # reset loop

            #summarize article:
            article_sum, on_topic_bool, articles_df = llm_get_summary(article_idx, dumb_llm, assets_df, articles_df)
            if not on_topic_bool:
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                continue # reset loop

            # s-filter:
            s_bool, s_type, articles_df = s_filter(article_idx, s_low, s_high, storage_path, articles_df, debug)
            if not s_bool:
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                continue # reset loop
            if s_type == "update": # update post
                #posts_df, articles_df = post_update(article_idx, articles_df, posts_df, dev_mode, debug)
                print("update post")
                continue # reset loop

            # b-filter:
            b_bool, b_str, articles_df = b_filter(article_idx, dumb_llm, articles_df)
            if not b_bool:
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
                continue # reset loop
            # news type filter:
            news_type, articles_df = llm_get_summary_metadata(article_idx, dumb_llm, assets_df, articles_df)
            if news_type in ['earnings', 'financial', 'macroeconomic', \
                                'market analysis', 'opinion', 'management' ,'other']:
                articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
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

            articles_df.to_csv(os.path.join(storage_path, 'articles.csv'), index=False)
            # social media post
            #posts_df, articles_df = post_story(article_idx, articles_df, posts_df, dev_mode, debug)
        if debug:
            break