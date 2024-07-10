import re
import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize
from wordfreq import zipf_frequency
from collections import Counter
from llm import *
from util import *
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from openai import OpenAI
import chromadb

def get_llm_q_bool(title, asset_name, llm_name):
    # define the response schemas
    response_schemas = [ResponseSchema(
        name="q_bool",
        description="Indicate if the article title is relevant, quality news.",
        type="bool")
        ]


    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template='''
    Evaluate if the following article title pertains to relevant news about {asset_name} such as product development, regulatory changes, legal action, business announcements, etc. \
    or if it's describing something that is speculative or already priced into the stock such as price action, investment advice, earnings, expectations, etc.

    {format_instructions}
    Do not include comments in the output JSON string.
    
    Here is the title: {title}  
    '''

    prompt = PromptTemplate(
        template=template,
        input_variables=["title", "asset_name"],
        partial_variables={"format_instructions": format_instructions}
    )
  
    chat = ChatAnthropic(model=llm_name, temperature=0)

    _input = prompt.format_prompt(title=title, asset_name=asset_name)
    output = chat.invoke(_input.to_messages()).content
    dict_out = output_parser.parse(output)
    q_bool = dict_out['q_bool']
    return q_bool

def get_q_bool_keywords(title):
    title = title.lower()
    title = title.replace('-', ' ')
    q_bool = True
    reject_reason = None

    reject_keywords = ['must buy', 'a buy', 'a sell', 'a "hold"', 'a hold', 'blue chip', 'why these', 
                    'growth stocks', 'more gains', 'to buy', 'is a buy', 'top pick', 'stock reach',
                    'could reach', "here's why", 'could skyrocket', 'market rebound', 'profit outlook', 
                    'dividend stocks', 'to watch', 'stock slips', 'to reach', 'year high', 'stock forecast', 
                    'to surpass', 'stock climbs', 'stock soars', 'stock up', 'to grow by', 'poised to',
                    'passive income', 'your radar', 'prepare for', 'earnings report', 'quarterly report',
                    'faang stock', 'quarterly earnings', 'filing', 'analyst', 'opinion', 'stock gains', 
                    'stock market', 'stock movers', 'bullish', 'bearish', 'how to earn', 'you may not',
                    'single-session', 'parabolic']
  
    reject_patterns = [
        r"\d+\s*(\w+\s*){0,3}stock",          # int followed by up to three words and "stock"
        r"\d+\s*(\w+\s*){0,3}companies",      # int followed by up to three words and "companies"
        r'\d+\s*(\w+\s*){0,3}picks',          # int followed by up to three words and "picks"
        r"\d+\s*(\w+\s*){0,2}reasons",        # int followed by up to two words and "reasons"
        r"\d+\s*(\w+\s*){0,1}crypto",         # int followed by up to one word and "crypto"
        r"\d+\s*(\w+\s*){0,1}countries",      # int followed by up to one word and "countries"
        r"\d+\s*best",                        # int followed by "best"
        r'\d+\s*largest',                     # int followed by "largest"
        r"can+\s*(\w+\s*){0,3}reach",         # "can" followed by up to three words and "reach"
        r"will+\s*(\w+\s*){0,3}reach",        # "will" followed by up to three words and "reach"
        r'soars\s*(\d+)%'                     # "soars" followed by a percentage 
    ]

    for key in reject_keywords:
        if key in title:
            q_bool = False
            reject_reason = 'q-filter keyword'
            break 
            
    for pattern in reject_patterns:
        if re.search(pattern, title):
            q_bool = False
            reject_reason = 'q-filter pattern'
            break  
    return q_bool, reject_reason

def get_q_bool(title, asset_name, llm_name):
    q_bool, reject_reason = get_q_bool_keywords(title)
    if q_bool: 
        q_bool = get_llm_q_bool(title, asset_name, llm_name)
        if not q_bool:
            reject_reason = "q-filter llm"
        else:
            reject_reason = None

    return q_bool, reject_reason

def q_filter(article_idx, assets_df, articles_df, debug=False):
    article_title = articles_df.loc[article_idx, 'title']
    ticker = articles_df.loc[article_idx, 'source_ticker']
    asset_name = assets_df.loc[assets_df['ticker'] == ticker, 'name'].iloc[0]
    q_bool, reject_reason = get_q_bool(article_title, asset_name, smart_llm)
    if debug:
        print("Article title for q-filter:", article_title)
    articles_df.loc[article_idx, 'qfactor_bool'] = q_bool
    article_id = articles_df.loc[article_idx, 'article_id']
    if not q_bool: 
        print("\033[31m" + "Article rejected: " + reject_reason + "\033[0m")
        articles_df = log_outcome(article_idx, articles_df, reject_reason)

    return articles_df, q_bool

def b_filter(article_idx, llm_name, articles_df):
    summary = articles_df.loc[article_idx, 'summary']
    b_str = llm_get_summary_b_index(summary, llm_name)

    if b_str == 'reflective': b_bool = False
    elif b_str == 'breaking': b_bool = True
    else:
        print(f'WARINING: r_index not recoginzed. Dismissing r-filter') 
        b_bool = False

    articles_df.loc[article_idx, 'bfactor_str'] = b_str
    article_id = articles_df.loc[article_idx, 'article_id']

    if b_bool: 
        article_title = articles_df.loc[article_idx, 'title']
        article_sum = articles_df.loc[article_idx, 'summary']
        article_date = articles_df.loc[article_idx, 'pub_date']
        # run t filter:
        #t_bool, t_dict, reject_reason, breaking_time = get_t_bool(article_title, article_sum, article_date, llm_name='auto')
        #t_dict = str(t_dict)
        #articles_df.loc[article_idx, 't_bool'] = t_bool
        #articles_df.loc[article_idx, 't_dict'] = t_dict
        #articles_df.loc[article_idx, 'breaking_time'] = breaking_time
        #articles_df = log_outcome(article_idx, articles_df, reject_reason)
    else:
        print("\033[31m" + "Article is not breaking news, no action required." + "\033[0m")
        articles_df = log_outcome(article_idx, articles_df, "not breaking")

    return b_bool, b_str, articles_df

def llm_get_summary_b_index(summary, llm_name):
    # define the response schemas
    response_schemas = [ResponseSchema(
        name="b_str",
        description="Str: 'reflective' or 'breaking'"
    )]


    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template='''
    Identify whether the text discusses breaking news (breaking), meaning it is new information that has just been released, or reflective news (reflective), meaning it is a summary or analysis of past events.
    {format_instructions}
    text: {summary}  
    '''

    prompt = PromptTemplate(
        template=template,
        input_variables=["summary"],
        partial_variables={"format_instructions": format_instructions}
    )
  
    chat = ChatAnthropic(model=llm_name, temperature=0)

    _input = prompt.format_prompt(summary=summary)
    output = chat.invoke(_input.to_messages()).content
    dict_out = output_parser.parse(output)
    b_str = dict_out['b_str']

    return b_str
    
def s_filter(article_idx, s_low, s_high, storage_path, articles_df, debug=False):

    ticker = articles_df.loc[article_idx, 'source_ticker']
    article_id = articles_df.loc[article_idx, 'article_id']
    current_sum = articles_df.loc[article_idx, 'summary']
    target_embedding = get_embedding(current_sum)

    # initialize params
    ref_id_match, s_type, s_factor_match = None, None, None
    s_bool = True 

    if debug:
        chroma_client = chromadb.PersistentClient(path=storage_path)
        sums = chroma_client.get_collection("summaries")
        # include all articles up to the current one
        sums_df = articles_df.loc[articles_df["summary"].notna()].reset_index(drop=True)
        row_idx = sums_df[sums_df["article_id"] == article_id].index[0]
        include_article_ids = sums_df.iloc[:row_idx]["article_id"].values.tolist()
        query_dict = sums.query(query_embeddings = target_embedding, 
                                n_results=5, 
                                where={'$and':[
                                        {"ticker":{"$eq": ticker}},
                                        {"article_id": {"$in": include_article_ids}}
                                    ]},
                                include = ["distances", "metadatas", "documents"])
    else:
        chroma_client = chromadb.PersistentClient(path=storage_path)
        sums = chroma_client.get_collection("summaries")
        query_dict = sums.query(query_embeddings = target_embedding, 
                                n_results=5, 
                                where={"ticker": ticker}, 
                                include = ["distances", "metadatas", "documents"])
        
    distances = np.array(query_dict["distances"][0])
    if len(distances) == 0:
        print("No similar articles found.")
        return True, 'different', articles_df
    metadata = np.array(query_dict["metadatas"][0])
    summaries = np.array(query_dict["documents"][0])
    repeat_articles_bool = np.array(query_dict["distances"][0]) < s_low
    llm_check_bool = (np.array(query_dict["distances"][0]) > s_low) & (np.array(query_dict["distances"][0]) < s_high)

    if sum(repeat_articles_bool) > 0:
        s_bool = False
        metadata_dicts = metadata[repeat_articles_bool]
        ref_ids = [d["article_id"] for d in metadata_dicts]
        matches = articles_df.loc[articles_df["article_id"].isin(ref_ids), "duplicate_articles"].values
        # break if there is a repeat article:
        ref_id_match = ref_ids[0]
        s_factor_match = distances[0]
        s_type = "same"
        match_matches = matches
        reject_reason = "s-filter same"
        articles_df = log_outcome(article_idx, articles_df, reject_reason)
        if debug:
            print("Current sum:", current_sum, "\n", "Closest match:", summaries[0], str(metadata[0]["article_id"]))
        print("\033[31m" + "Article rejected: " + reject_reason + "\033[0m")
    elif sum(llm_check_bool) > 0:
        if debug:
            print("Articles checked with s-filter LLM:", str(sum(llm_check_bool)))
        metadata_dicts = metadata[llm_check_bool]
        ref_ids = [d["article_id"] for d in metadata_dicts]
        s_factors = np.array(query_dict["distances"])[0][llm_check_bool]
        # get the corresponding reference article sums and order from newest to oldest
        ref_sums = articles_df[articles_df["article_id"].isin(ref_ids)]["summary"].values[::-1]
        # use language model to determine if the article is an update or different
        for i, ref_sum in enumerate(ref_sums):
            matches = articles_df.loc[articles_df["article_id"] == ref_ids[i], "duplicate_articles"].values
            s_bool, s_type = llm_s_factor_bool(ref_sum, current_sum, smart_llm)
            
            # break if update or same article found
            if s_type in ['update', 'same']:
                ref_id_match = ref_ids[i]
                s_factor_match = s_factors[i]
                match_matches = matches
                if s_type == 'same':
                    reject_reason = "s-filter llm"
                    articles_df = log_outcome(article_idx, articles_df, reject_reason)
                    print("\033[31m" + "Article rejected: " + reject_reason + "\033[0m")
                break
        if debug:
            print("Current sum:", current_sum, "\n", "Closest match:", summaries[0], "article_id:", str(metadata[0]["article_id"]))
    else:
        # no similar article matches
        s_type = "different"

    if debug:
        print("closest s_factor:", str(distances[0]), "s_type:", s_type)
    if ref_id_match is not None: 
        duplicate_articles_dict = str({"match_type": s_type, "match_id":ref_id_match}) #, "match_matches":match_matches})
    else:  
        duplicate_articles_dict = float('nan')

    articles_df.loc[article_idx, 'sfactor'] = s_factor_match
    articles_df.loc[article_idx, 'sfactor_bool'] = s_bool
    articles_df.loc[article_idx, 'duplicate_articles'] = duplicate_articles_dict

    # add the article summary embedding to the database
    if not debug:
        sums.add(embeddings=target_embedding, 
                documents=current_sum, 
                ids=str(sums.count()), 
                metadatas={"ticker": ticker, "article_id": article_id})

    return s_bool, s_type, articles_df

def llm_s_factor_bool(text1, text2, llm_name):
    reject_reason = None

    # Define the response schemas
    response_schemas = [ResponseSchema(
        name="event_similarity",
        description="Respond with 'same' or 'update' or 'different'. 'update' is for when the articles describe the same news but one has significant additional information"
    )]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = '''
    Are the following two article summaries likely describing the same or different news events?
    {format_instructions}
    Summary1: {text1}
    Summary2: {text2}
    Respond with 'same' or 'update' or 'different'.
    '''

    prompt = PromptTemplate(
        template=template,
        input_variables=["text1", "text2"],
        partial_variables={"format_instructions": format_instructions}
    )

    chat = ChatAnthropic(model=llm_name, temperature=0)

    _input = prompt.format_prompt(text1=text1, text2=text2)
    output = chat.invoke(_input.to_messages()).content
    #llm_cost = llm_cost_cents(str(_input), str(output), llm_name)

    try:
        dict_out = output_parser.parse(output)
        event_similarity = dict_out['event_similarity']
        if event_similarity == 'same':
            bool_out = False
        else:
            bool_out = True
        type_out = event_similarity
    except:
        bool_out = False
        type_out = output

    return bool_out, type_out
